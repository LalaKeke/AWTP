import os
import sys
import math
import copy
import socket
import glob
import ast
import subprocess
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import theseus as th

from LaneBoundaryLagrangianLoss_dual_final import (
    LaneBoundaryLagrangianLoss,
    SafetyConstraintLoss,
    SoftConstraintLambdas,
)

PNN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PNN_ROOT not in sys.path:
    sys.path.insert(0, PNN_ROOT)
PROJECT_ROOT = os.path.dirname(PNN_ROOT)

from nnc.controllers.baselines.dynamics_v4 import BicycleModel
from nnc.controllers.neural_network.nnc_controllers import NeuralNetworkController, NNCDynamics
from PCC_helpers_static_v1 import (
    normalize,
    inverse_normalize,
    StaticAwareControlNet,
)
from weight_model_v10 import (
    WeightNet,
    load_control_encoder_to_weightnet,
    freeze_pretrained_part,
)


def masked_agent_minmax(
    states: torch.Tensor,
    mask: Optional[torch.Tensor],
    feature_dim: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute agent feature min/max without counting padded agent slots."""
    states = states.reshape(-1, feature_dim)
    if mask is not None:
        valid = mask.reshape(-1).bool()
        if valid.any():
            states = states[valid]
    return states.min(0).values, states.max(0).values


def tensor_feature_minmax(
    values: torch.Tensor,
    q_low: float = 0.0,
    q_high: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feature-wise min/max, optionally using robust quantiles.

    PNN uses min/max normalization. Raw B2D tensors contain rare but very large
    lane/agent outliers; full-range min/max compresses most normal samples and
    can make the ControlNet produce unnecessarily sharp controls. Quantile
    stats keep the old behavior when q_low/q_high are 0/1, but allow robust
    normalization for Stage-2 safety/comfort experiments.
    """
    values = values.float()
    if values.numel() == 0:
        raise ValueError("Cannot compute min/max on an empty tensor.")
    if q_low <= 0.0 and q_high >= 1.0:
        lo = values.min(0).values
        hi = values.max(0).values
    else:
        lo = torch.quantile(values, float(q_low), dim=0)
        hi = torch.quantile(values, float(q_high), dim=0)
        fallback_lo = values.min(0).values
        fallback_hi = values.max(0).values
        bad = (hi - lo).abs() < 1e-6
        lo = torch.where(bad, fallback_lo, lo)
        hi = torch.where(bad, fallback_hi, hi)
    return lo, hi


def masked_agent_stats_minmax(
    states: torch.Tensor,
    mask: Optional[torch.Tensor],
    feature_dim: int = 6,
    q_low: float = 0.0,
    q_high: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    states = states.reshape(-1, feature_dim)
    if mask is not None:
        valid = mask.reshape(-1).bool()
        if valid.any():
            states = states[valid]
    return tensor_feature_minmax(states, q_low=q_low, q_high=q_high)


torch.autograd.set_detect_anomaly(False)

DT = 0.1
TRAJ_LEN = 30
WHEELBASE = 2.85
MAX_ACC = 10.0
MAX_STEER = 1.066
SAFE_MARGIN = 6.0
OFFICIAL_EGO_LENGTH = 4.084
OFFICIAL_EGO_WIDTH = 1.85
OFFICIAL_EGO_CENTER_FORWARD_OFFSET = 0.5
TTC_HORIZON = 2.0
TTC_WEIGHT = 0.8
TTC_EPS = 1e-3
COLLISION_BARRIER_SHARPNESS = 10.0
LANE_DAC_MARGIN = 0.30
LANE_HALF_WIDTH_FALLBACK = 1.90
JERK_COMFORT_LIMIT = 8.0
STEER_RATE_COMFORT_LIMIT = 0.40
PROGRESS_OVERSHOOT_WEIGHT = 0.25
REFERENCE_FORWARD_OFFSET = 0.0

COST_NAMES = [
    "acceleration",
    "jerk",
    "steering",
    "steering_change",
    "lane_xy",
    "lane_theta",
    "route_target",
    "safe",
]
NUM_COSTS = len(COST_NAMES)

MONITORED_CONTROL_LOSS_COMPONENT_NAMES = (
    "loss_track_ego",
    "loss_safety",
    "loss_lane_hard",
    "loss_lane_clearance",
    "loss_metric_lane",
    "loss_comfort_threshold",
    "loss_control_rate_ego",
    "loss_teacher_trust",
    "loss_pnn_only_hipad_anchor",
    "loss_official_frame_acr",
    "loss_official_frame_anchor",
    "loss_official_frame_lane_guard",
    "loss_safe_parent_distill",
    "loss_static_box_safety",
    "loss_static_any_hit",
    "loss_static_hipad_anchor",
    "loss_static_risk_gate",
)


class StaticNNCDynamics(nn.Module):
    """Small compatibility wrapper retaining train_v10 optimizer semantics."""

    def __init__(self, neural_network: nn.Module):
        super().__init__()
        self.neural_network = neural_network

    def forward(
        self,
        ego_state,
        ped_states,
        veh_states,
        lane_points,
        ped_mask,
        veh_mask,
        static_states,
        static_mask,
    ):
        return self.neural_network(
            ego_state,
            ped_states,
            veh_states,
            lane_points,
            ped_mask,
            veh_mask,
            static_states,
            static_mask,
        )


# ============================================================
# Basic utilities
# ============================================================
def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def linear_ramp(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))


def should_save_best_l2_checkpoint(
    l2_avg: float,
    weight_variation_score: float,
    best_l2_with_variation: float,
    min_weight_variation_score: float,
) -> bool:
    return (
        math.isfinite(float(l2_avg))
        and float(weight_variation_score) >= float(min_weight_variation_score)
        and float(l2_avg) < float(best_l2_with_variation)
    )


def run_l2_eval_for_checkpoint(
    ckpt_path: str,
    save_dir: str,
    epoch: int,
    weight_variation_score: float,
    cfg_runtime: Dict,
) -> Dict[str, float]:
    if not cfg_runtime.get("eval_each_epoch", False):
        return {}

    ckpt_path = os.path.abspath(ckpt_path)
    save_dir = os.path.abspath(save_dir)
    nnplanner_python = cfg_runtime.get("eval_nnplanner_python", sys.executable)
    pinn_python = cfg_runtime.get("eval_pinn_python", sys.executable)
    l2_script = cfg_runtime.get(
        "eval_l2_script",
        os.environ.get("PNN_L2_EVAL_SCRIPT", os.path.join(PROJECT_ROOT, "tools", "run_sparse_l2_eval.py")),
    )
    eval_gpu_visible = str(cfg_runtime.get("eval_cuda_visible_devices", "5"))
    eval_root = os.path.join(save_dir, "epoch_l2_eval", f"epoch_{epoch:04d}")
    mkdir(eval_root)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = eval_gpu_visible
    wrapper = f"""
import new_run
new_run.CKPT_PATH = {ckpt_path!r}
new_run.SAVE_BASE_DIR = {eval_root!r}
new_run.GPU_ID = 0
new_run.SAVE_IMAGES = False
new_run.SAVE_SPARSEDRIVE_STYLE = True
new_run.main()
"""
    run_cmd = [nnplanner_python, "-c", wrapper]
    print(f"[epoch {epoch}] running new_run eval on CUDA_VISIBLE_DEVICES={eval_gpu_visible}: {ckpt_path}")
    new_run_proc = subprocess.run(
        run_cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    new_run_log = os.path.join(eval_root, "new_run.log")
    with open(new_run_log, "w", encoding="utf-8") as f:
        f.write(new_run_proc.stdout)
    if new_run_proc.returncode != 0:
        print(f"[epoch {epoch}] new_run eval failed, log={new_run_log}")
        return {"eval_failed": 1.0}

    candidates = sorted(
        glob.glob(os.path.join(eval_root, "**", "all_results_*.pkl"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        print(f"[epoch {epoch}] no all_results pkl found under {eval_root}")
        return {"eval_failed": 1.0}

    all_results = candidates[0]
    converted = os.path.join(eval_root, f"results_replaced_epoch_{epoch:04d}.pkl")
    l2_log = converted + ".planning_evalv2.log"
    l2_cmd = [
        pinn_python,
        l2_script,
        "--all-results",
        all_results,
        "--converted-output",
        converted,
        "--log-path",
        l2_log,
    ]
    print(f"[epoch {epoch}] running SparseDrive L2 eval: {all_results}")
    l2_env = os.environ.copy()
    l2_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    l2_proc = subprocess.run(
        l2_cmd,
        env=l2_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    l2_stdout_log = os.path.join(eval_root, "run_sparse_l2_eval.stdout.log")
    with open(l2_stdout_log, "w", encoding="utf-8") as f:
        f.write(l2_proc.stdout)
    if l2_proc.returncode != 0:
        print(f"[epoch {epoch}] L2 eval failed, log={l2_stdout_log}")
        return {"eval_failed": 1.0}

    metric = {}
    marker = "[eval] parsed metric_dict:"
    for line in l2_proc.stdout.splitlines():
        if marker in line:
            try:
                metric = ast.literal_eval(line.split(marker, 1)[1].strip())
            except Exception:
                metric = {}
    if not metric:
        print(f"[epoch {epoch}] metric_dict parse failed, log={l2_stdout_log}")
        return {"eval_failed": 1.0}

    l2_avg = float(metric.get("L2", float("nan")))
    row = {
        "epoch": epoch,
        "checkpoint": ckpt_path,
        "all_results": all_results,
        "converted": converted,
        "log": l2_log,
        "L2": l2_avg,
        "obj_col": float(metric.get("obj_col", float("nan"))),
        "obj_box_col": float(metric.get("obj_box_col", float("nan"))),
        "weight_variation_score": weight_variation_score,
    }
    eval_csv = os.path.join(save_dir, "epoch_l2_eval", "eval_history.csv")
    pd.DataFrame([row]).to_csv(eval_csv, mode="a", header=not os.path.exists(eval_csv), index=False)
    print(
        f"[epoch {epoch}] eval L2={l2_avg:.6f} | "
        f"obj_box_col={row['obj_box_col']:.6f} | weight_variation={weight_variation_score:.6f}"
    )
    return row


def reduce_mean_scalar(value: float, device: torch.device, world_size: int) -> float:
    t = torch.tensor(float(value), device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
    return float(t.item())


def reduce_mean_tensor(x: torch.Tensor, world_size: int) -> torch.Tensor:
    y = x.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y /= world_size
    return y


def reduce_sum_tensor(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def masked_mean_per_sample(x: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return x.reshape(x.shape[0], -1).mean(dim=1)
    mask = mask.to(dtype=x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(x)
    num = (x * mask).reshape(x.shape[0], -1).sum(dim=1)
    den = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(eps)
    return num / den


def build_model_padding_mask(valid_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if valid_mask is None:
        return None
    return ~valid_mask.bool()


def plot_history(history_rows: List[Dict[str, float]], save_dir: str):
    if len(history_rows) == 0:
        return

    mkdir(save_dir)
    df = pd.DataFrame(history_rows)
    df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    fig = plt.figure(figsize=(10, 6))
    for col in [
        "control_total",
        "weight_loss",
        "weight_traj_loss",
        "weight_rule_loss",
        "weight_feedback_loss",
        "weight_rank_loss",
        "weight_sep_loss",
        "weight_entropy_band_loss",
        "weight_diversity_floor_loss",
        "weight_extreme_loss",
        "aug_term",
    ]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=200)
    plt.close(fig)

    weight_cols = [c for c in df.columns if c.startswith("costw_")]
    if len(weight_cols) > 0:
        fig = plt.figure(figsize=(10, 6))
        for c in weight_cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("epoch")
        plt.ylabel("planner weight")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "weight_curves.png"), dpi=200)
        plt.close(fig)


def _coord_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    meta = data.get("__meta__", {})
    return meta if isinstance(meta, dict) else {}


def _angle_distance(a: torch.Tensor, b: float) -> torch.Tensor:
    ref = torch.tensor(float(b), dtype=a.dtype, device=a.device)
    return torch.atan2(torch.sin(a - ref), torch.cos(a - ref)).abs()


def infer_coord_convention_from_tensor(data: Dict[str, Any]) -> str:
    """Infer the old/new coordinate convention from saved tensor metadata.

    New converted B2D files carry ``__meta__['coord_convention']``. For older
    files, fall back to the ego heading convention:
      hipad_xy: ego starts with theta ~= pi/2 because y is forward.
      pnn_xy:   ego starts with theta ~= 0 because x is forward.
    """
    meta = _coord_meta(data)
    conv = meta.get("coord_convention")
    if conv in {"hipad_xy", "pnn_xy"}:
        return str(conv)

    if "ego_state" not in data or not torch.is_tensor(data["ego_state"]):
        return "unknown"
    theta = data["ego_state"][:, 2].float()
    if theta.numel() == 0:
        return "unknown"
    sample = theta[:: max(1, theta.numel() // 20000)]
    d_pnn = _angle_distance(sample, 0.0).median().item()
    d_hipad = _angle_distance(sample, math.pi / 2).median().item()
    if d_pnn < 0.20 and d_pnn < d_hipad:
        return "pnn_xy"
    if d_hipad < 0.20 and d_hipad <= d_pnn:
        return "hipad_xy"
    return "unknown"


def validate_dataset_coord_convention(
    old_data: Dict[str, Any],
    new_data: Dict[str, Any],
    expected: Optional[str],
) -> str:
    old_conv = infer_coord_convention_from_tensor(old_data)
    new_conv = _coord_meta(new_data).get("coord_convention", old_conv)
    if new_conv not in {"hipad_xy", "pnn_xy", "unknown"}:
        new_conv = "unknown"
    if old_conv != "unknown" and new_conv != "unknown" and old_conv != new_conv:
        raise ValueError(
            "old/new coordinate convention mismatch: "
            f"old={old_conv}, new={new_conv}. Regenerate the paired .pt files together."
        )

    expected = (expected or "").strip()
    if expected:
        if expected not in {"hipad_xy", "pnn_xy"}:
            raise ValueError(f"PNN_COORD_CONVENTION must be hipad_xy or pnn_xy, got {expected!r}")
        inferred = old_conv if old_conv != "unknown" else new_conv
        if inferred != "unknown" and inferred != expected:
            raise ValueError(
                "PNN_COORD_CONVENTION does not match training tensors: "
                f"expected={expected}, data={inferred}. "
                "Use the matching dataset or regenerate with --coord-convention."
            )

    effective = expected or (old_conv if old_conv != "unknown" else new_conv)
    if effective not in {"hipad_xy", "pnn_xy"}:
        effective = "unknown"
    old_meta = _coord_meta(old_data)
    print(
        "[CoordConvention] "
        f"effective={effective}, old={old_conv}, new={new_conv}, "
        f"route_source={old_meta.get('route_source', '<unknown>')}, "
        f"lane_semantics={old_meta.get('lane_points_semantics', 'lane_points[0]=left,lane_points[1]=right')}"
    )
    return effective


def warn_if_lane_order_suspicious(data: Dict[str, Any], coord_convention: str) -> None:
    lane = data.get("lane_points")
    if not torch.is_tensor(lane) or lane.ndim != 4 or lane.size(1) < 2 or lane.size(-1) != 2:
        return
    sample = lane[:: max(1, lane.size(0) // 20000), :2].float()
    if coord_convention == "hipad_xy":
        left_lat = sample[:, 0, :, 0].median(dim=1).values
        right_lat = sample[:, 1, :, 0].median(dim=1).values
        ok = (left_lat <= right_lat).float().mean().item()
        expected = "hipad_xy expects lane0.x <= lane1.x because x points right"
    elif coord_convention == "pnn_xy":
        left_lat = sample[:, 0, :, 1].median(dim=1).values
        right_lat = sample[:, 1, :, 1].median(dim=1).values
        ok = (left_lat >= right_lat).float().mean().item()
        expected = "pnn_xy expects lane0.y >= lane1.y because y points left"
    else:
        return

    print(f"[CoordConvention] lane left/right order check: ok_ratio={ok:.3f} ({expected})")
    if ok < 0.55:
        print(
            "[WARN][CoordConvention] lane_points[0/1] order looks suspicious. "
            "Lane prior, lane loss, WeightNet lane tokens, and Theseus lane objectives "
            "all assume lane_points[0]=left and lane_points[1]=right. "
            "Regenerate the .pt data with the current converter if this is B2D converted data."
        )


class PairedOldNewDataset(Dataset):
    def __init__(self, old_pt_path: str, new_pt_path: str, supervision_pt_path: Optional[str] = None):
        self.old_data = torch.load(old_pt_path, map_location="cpu")
        self.new_data = torch.load(new_pt_path, map_location="cpu")
        self.supervision_data = (
            torch.load(supervision_pt_path, map_location="cpu")
            if supervision_pt_path
            else None
        )

        if not isinstance(self.old_data, dict):
            raise TypeError(f"old pt must be dict, got {type(self.old_data)}")
        if not isinstance(self.new_data, dict):
            raise TypeError(f"new pt must be dict, got {type(self.new_data)}")
        if "ego_state" not in self.old_data:
            raise KeyError("old pt missing key: ego_state")

        self.num_old = self.old_data["ego_state"].shape[0]

        if "ego_future_gt" in self.new_data:
            self.num_new = self.new_data["ego_future_gt"].shape[0]
        elif "ego_state" in self.new_data:
            self.num_new = self.new_data["ego_state"].shape[0]
        else:
            raise KeyError("new pt must contain either 'ego_future_gt' or 'ego_state'")

        if self.num_old != self.num_new:
            raise ValueError(
                f"old/new sample count mismatch: old={self.num_old}, new={self.num_new}. "
                f"Forced i-to-i pairing is impossible."
            )

        if self.supervision_data is not None:
            if not isinstance(self.supervision_data, dict):
                raise TypeError(
                    f"supervision pt must be dict, got {type(self.supervision_data)}"
                )
            if "sample_key_hash" not in self.supervision_data:
                raise KeyError("supervision pt missing key: sample_key_hash")
            num_supervision = int(self.supervision_data["sample_key_hash"].shape[0])
            if num_supervision != self.num_old:
                raise ValueError(
                    "old/new/supervision sample count mismatch: "
                    f"old={self.num_old}, new={self.num_new}, supervision={num_supervision}"
                )
            print(
                "[PairedOldNewDataset] metric-aligned supervision enabled: "
                f"{supervision_pt_path}"
            )

        print(
            f"[PairedOldNewDataset] force index-aligned pairing enabled: "
            f"old[i] <=> new[i], num_samples={self.num_old}"
        )

        self.coord_convention = validate_dataset_coord_convention(
            self.old_data,
            self.new_data,
            expected=os.environ.get("PNN_COORD_CONVENTION"),
        )
        warn_if_lane_order_suspicious(self.old_data, self.coord_convention)

    def __len__(self):
        return self.num_old

    def _get_new_gt_and_mask(self, idx: int):
        if "ego_future_gt" in self.new_data:
            gt = self.new_data["ego_future_gt"][idx].float()
        else:
            gt = self.new_data["ego_state"][idx, 4:10].float()

        if "ego_future_gt_valid_mask" in self.new_data:
            valid = self.new_data["ego_future_gt_valid_mask"][idx].bool()
        else:
            valid = ~torch.isnan(gt).any()

        gt = torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        return gt, valid

    def __getitem__(self, idx: int):
        item = {}

        for k in [
            "ego_state",
            "ped_states",
            "veh_states",
            "static_states",
            "lane_points",
        ]:
            if k not in self.old_data:
                raise KeyError(f"old pt missing key: {k}")
            item[k] = self.old_data[k][idx]

        if "ped_mask" in self.old_data:
            item["ped_mask"] = self.old_data["ped_mask"][idx]
        if "veh_mask" in self.old_data:
            item["veh_mask"] = self.old_data["veh_mask"][idx]
        if "static_mask" in self.old_data:
            item["static_mask"] = self.old_data["static_mask"][idx]

        gt_new, valid_new = self._get_new_gt_and_mask(idx)
        item["ego_future_gt_new"] = gt_new
        item["ego_future_gt_valid_mask"] = valid_new
        if "gt_reference_line" in self.new_data:
            item["gt_reference_line"] = self.new_data["gt_reference_line"][idx]
            item["gt_reference_line_valid_mask"] = self.new_data.get(
                "gt_reference_line_valid_mask",
                self.new_data["ego_future_gt_valid_mask"],
            )[idx]
        else:
            # Backward-compatible GT path reference for older paired tensors.
            # Regenerated data stores this explicitly with provenance metadata.
            start = self.old_data["ego_state"][idx, 0:2].float().unsqueeze(0)
            item["gt_reference_line"] = torch.cat([start, gt_new], dim=0)
            item["gt_reference_line_valid_mask"] = valid_new
        for key in (
            "gt_actor_boxes_2hz",
            "gt_actor_mask_2hz",
            "gt_actor_truncated",
            "hipad_plan_2hz",
            "official_hipad_obj_box_col",
            "official_fut_valid_mask",
        ):
            if key in self.new_data:
                item[key] = self.new_data[key][idx]
        if self.supervision_data is not None:
            for key in (
                "metric_gt_actor_boxes_2hz",
                "metric_gt_actor_mask_2hz",
                "metric_gt_actor_type_2hz",
                "metric_gt_actor_truncated",
                "gt_solid_lane_points",
                "gt_solid_lane_mask",
                "gt_solid_lane_truncated",
                "gt_obj_collision_mask_2hz",
                "gt_lane_collision_mask_2hz",
                "sample_key_hash",
            ):
                if key in self.supervision_data:
                    item[key] = self.supervision_data[key][idx]
            # The aligned boxes intentionally override the older adjacent-frame
            # reconstruction while preserving old checkpoints and tensor files.
            if "metric_gt_actor_boxes_2hz" in item:
                item["gt_actor_boxes_2hz"] = item["metric_gt_actor_boxes_2hz"]
                item["gt_actor_mask_2hz"] = item["metric_gt_actor_mask_2hz"]
        item["old_index"] = torch.tensor(idx, dtype=torch.long)
        item["new_index"] = torch.tensor(idx, dtype=torch.long)
        return item


# ============================================================
# Dynamics rollout
# ============================================================
def rollout_all_agents(
    dynamics_model,
    ego_state,
    ped_states,
    veh_states,
    u_ego,
    u_peds,
    u_vehs,
    N_step=TRAJ_LEN,
    dt=DT,
):
    B = ego_state.size(0)
    Np = ped_states.size(1)
    Nv = veh_states.size(1)

    ego = ego_state[:, :4]
    ped = ped_states[:, :, :4]
    veh = veh_states[:, :, :4]

    ego_list = []
    ped_list = []
    veh_list = []

    for t in range(N_step):
        dx_ego = dynamics_model(ego, u_ego[:, t])
        ego = ego + dt * dx_ego

        dx_ped = dynamics_model(
            ped.reshape(B * Np, 4),
            u_peds[:, :, t].reshape(B * Np, 2),
        )
        ped = ped + dt * dx_ped.view(B, Np, 4)

        dx_veh = dynamics_model(
            veh.reshape(B * Nv, 4),
            u_vehs[:, :, t].reshape(B * Nv, 2),
        )
        veh = veh + dt * dx_veh.view(B, Nv, 4)

        ego = torch.stack(
            [
                ego[:, 0],
                ego[:, 1],
                torch.atan2(torch.sin(ego[:, 2]), torch.cos(ego[:, 2])),
                torch.clamp(ego[:, 3], min=0.0),
            ],
            dim=1,
        )

        ped = torch.stack(
            [
                ped[:, :, 0],
                ped[:, :, 1],
                torch.atan2(torch.sin(ped[:, :, 2]), torch.cos(ped[:, :, 2])),
                torch.clamp(ped[:, :, 3], min=0.0),
            ],
            dim=2,
        )

        veh = torch.stack(
            [
                veh[:, :, 0],
                veh[:, :, 1],
                torch.atan2(torch.sin(veh[:, :, 2]), torch.cos(veh[:, :, 2])),
                torch.clamp(veh[:, :, 3], min=0.0),
            ],
            dim=2,
        )

        ego_list.append(ego)
        ped_list.append(ped)
        veh_list.append(veh)

    ego_traj = torch.stack(ego_list, dim=1)
    ped_traj = torch.stack(ped_list, dim=2)
    veh_traj = torch.stack(veh_list, dim=2)
    return ego_traj, ped_traj, veh_traj


# ============================================================
# Theseus / DIPP utilities
# ============================================================
def _reshape_control(control_tensor: torch.Tensor, horizon: int) -> torch.Tensor:
    return control_tensor.view(-1, horizon, 2)


def _angle_normalize(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def smooth_abs_excess(x: torch.Tensor, limit: float, sharpness: float = 10.0) -> torch.Tensor:
    return F.softplus(float(sharpness) * (x.abs() - float(limit))) / float(sharpness)


def route_progress_residuals(ego_traj: torch.Tensor, ego_state: torch.Tensor, horizon: int) -> torch.Tensor:
    start_xy = ego_state[:, :2]
    final_ref = ego_state[:, 8:10]
    route_vec = final_ref - start_xy
    route_norm = torch.norm(route_vec, dim=-1, keepdim=True).clamp_min(1e-4)
    route_dir = route_vec / route_norm
    route_perp = torch.stack([-route_dir[:, 1], route_dir[:, 0]], dim=-1)

    refs = [ego_state[:, 4:6], ego_state[:, 6:8], ego_state[:, 8:10]]
    indices = [min(9, horizon - 1), min(19, horizon - 1), min(29, horizon - 1)]
    residuals = []
    for idx, ref_xy in zip(indices, refs):
        err = ego_traj[:, idx, :2] - ref_xy
        lateral = (err * route_perp).sum(dim=-1)
        along = (err * route_dir).sum(dim=-1)
        progress = F.relu(-along) + PROGRESS_OVERSHOOT_WEIGHT * F.relu(along)
        residuals.extend([lateral, progress])
    return torch.stack(residuals, dim=1)


def bicycle_model_compatible(
    control: torch.Tensor,
    ego_state: torch.Tensor,
    dt: float = DT,
    wheelbase: float = WHEELBASE,
    max_a: float = MAX_ACC,
    max_d: float = MAX_STEER,
) -> torch.Tensor:
    if control.device != ego_state.device or control.dtype != ego_state.dtype:
        control = control.to(device=ego_state.device, dtype=ego_state.dtype)

    B, T, _ = control.shape
    state = ego_state[:, :4]

    if not torch.is_tensor(wheelbase):
        wheelbase_tensor = torch.tensor(
            float(wheelbase),
            device=ego_state.device,
            dtype=ego_state.dtype,
        )
    else:
        wheelbase_tensor = wheelbase.to(device=ego_state.device, dtype=ego_state.dtype)

    traj = []

    for t in range(T):
        a = control[:, t, 0].clamp(-max_a, max_a)
        delta = control[:, t, 1].clamp(-max_d, max_d)

        theta = torch.atan2(torch.sin(state[:, 2]), torch.cos(state[:, 2]))
        v = torch.clamp(state[:, 3], min=1e-8)

        dx_pos = v * torch.cos(theta)
        dy_pos = v * torch.sin(theta)
        dtheta = (v / wheelbase_tensor) * torch.tan(delta)
        dv = a

        next_state = torch.stack(
            [
                state[:, 0] + dt * dx_pos,
                state[:, 1] + dt * dy_pos,
                theta + dt * dtheta,
                state[:, 3] + dt * dv,
            ],
            dim=1,
        )

        next_state = torch.stack(
            [
                next_state[:, 0],
                next_state[:, 1],
                torch.atan2(torch.sin(next_state[:, 2]), torch.cos(next_state[:, 2])),
                torch.clamp(next_state[:, 3], min=0.0),
            ],
            dim=1,
        )

        traj.append(next_state)
        state = next_state
    return torch.stack(traj, dim=1)


def apply_forward_offset_to_traj(traj: torch.Tensor, offset: float) -> torch.Tensor:
    """Shift ego trajectory along each state's heading.

    Used for reference-point conversion. For example, if the PNN dynamics state
    is 0.3m ahead of the HiP-AD/GT reference point, use offset=-0.3 before
    computing losses/metrics against HiP-AD/GT targets.
    """
    offset = float(offset)
    if abs(offset) < 1e-8:
        return traj
    theta = traj[..., 2]
    shifted_xy = torch.stack(
        [
            traj[..., 0] + offset * torch.cos(theta),
            traj[..., 1] + offset * torch.sin(theta),
        ],
        dim=-1,
    )
    return torch.cat([shifted_xy, traj[..., 2:]], dim=-1)


def make_acceleration(horizon: int):
    def acceleration(optim_vars, aux_vars):
        control = _reshape_control(optim_vars[0].tensor, horizon)
        return control[:, :, 0]
    return acceleration


def make_jerk(horizon: int):
    def jerk(optim_vars, aux_vars):
        control = _reshape_control(optim_vars[0].tensor, horizon)
        acc = control[:, :, 0]
        return smooth_abs_excess(torch.diff(acc, dim=1) / DT, JERK_COMFORT_LIMIT)
    return jerk


def make_steering(horizon: int):
    def steering(optim_vars, aux_vars):
        control = _reshape_control(optim_vars[0].tensor, horizon)
        return control[:, :, 1]
    return steering


def make_steering_change(horizon: int):
    def steering_change(optim_vars, aux_vars):
        control = _reshape_control(optim_vars[0].tensor, horizon)
        steer = control[:, :, 1]
        return smooth_abs_excess(torch.diff(steer, dim=1) / DT, STEER_RATE_COMFORT_LIMIT)
    return steering_change


def make_control_anchor(horizon: int):
    def control_anchor(optim_vars, aux_vars):
        control = _reshape_control(optim_vars[0].tensor, horizon)
        initial_control = _reshape_control(aux_vars[0].tensor, horizon)
        normalized_delta = torch.stack(
            [
                (control[..., 0] - initial_control[..., 0]) / MAX_ACC,
                (control[..., 1] - initial_control[..., 1]) / MAX_STEER,
            ],
            dim=-1,
        )
        return normalized_delta.reshape(control.shape[0], horizon * 2)
    return control_anchor


def make_lane_xy(horizon: int):
    def lane_xy(optim_vars, aux_vars):
        ref_line = aux_vars[0].tensor
        ego_state = aux_vars[1].tensor
        control = _reshape_control(optim_vars[0].tensor, horizon)
        traj = bicycle_model_compatible(control, ego_state)
        traj = apply_forward_offset_to_traj(traj, -REFERENCE_FORWARD_OFFSET)

        distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
        k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
        ref_points = torch.gather(ref_line, 1, k)

        center_error = torch.norm(traj[:, 1::2, :2] - ref_points[:, 1::2, :2], dim=-1)
        drivable_excess = smooth_abs_excess(
            center_error,
            max(LANE_HALF_WIDTH_FALLBACK - LANE_DAC_MARGIN, 0.1),
            sharpness=8.0,
        )
        lane_error = torch.cat([center_error, 2.0 * drivable_excess], dim=1)
        return lane_error
    return lane_xy


def make_lane_theta(horizon: int):
    def lane_theta(optim_vars, aux_vars):
        ref_line = aux_vars[0].tensor
        ego_state = aux_vars[1].tensor
        control = _reshape_control(optim_vars[0].tensor, horizon)
        traj = bicycle_model_compatible(control, ego_state)
        traj = apply_forward_offset_to_traj(traj, -REFERENCE_FORWARD_OFFSET)

        distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
        k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
        ref_points = torch.gather(ref_line, 1, k)

        theta = traj[:, :, 2]
        lane_error = _angle_normalize(theta[:, 1::2] - ref_points[:, 1::2, 2])
        return lane_error
    return lane_theta


def make_route_target(horizon: int):
    def route_target(optim_vars, aux_vars):
        ego_state = aux_vars[0].tensor
        control = _reshape_control(optim_vars[0].tensor, horizon)
        traj = bicycle_model_compatible(control, ego_state)
        traj = apply_forward_offset_to_traj(traj, -REFERENCE_FORWARD_OFFSET)

        return route_progress_residuals(traj, ego_state, horizon)
    return route_target


def make_safety(horizon: int):
    def safety(optim_vars, aux_vars):
        ego_state = aux_vars[0].tensor
        agents_future = aux_vars[1].tensor
        agents_mask = aux_vars[2].tensor.bool()
        agent_safety_distance = aux_vars[3].tensor
        agent_lateral_safety_distance = aux_vars[4].tensor

        control = _reshape_control(optim_vars[0].tensor, horizon)
        traj = bicycle_model_compatible(control, ego_state)
        traj = apply_forward_offset_to_traj(traj, -REFERENCE_FORWARD_OFFSET)
        ego_xy = traj[:, :, :2]
        route_direction = ego_state[:, 8:10] - ego_state[:, :2]

        B, Nobj, T, _ = agents_future.shape
        residual = compute_collision_ttc_residual(
            ego_xy=ego_xy,
            agents_xy=agents_future,
            agents_mask=agents_mask,
            collision_dist=agent_safety_distance,
            collision_lateral_dist=agent_lateral_safety_distance,
            route_direction=route_direction,
        )
        return residual.reshape(B, Nobj * T)
    return safety


def build_objective(
    objective,
    control_variables,
    ego_state,
    ref_line,
    agents_future,
    agents_mask,
    agent_safety_distance,
    agent_lateral_safety_distance,
    initial_control_reference,
    cost_function_weights,
    control_anchor_weight,
    horizon_tensor,
    num_objects: int,
):
    horizon = int(horizon_tensor.tensor.item())
    lane_theta_dim = horizon // 2
    route_target_dim = 6
    safety_dim = num_objects * horizon

    objective.add(
        th.AutoDiffCostFunction(
            [control_variables], make_acceleration(horizon), horizon, cost_function_weights[0],
            aux_vars=[], autograd_vectorize=False, name="acceleration"
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            [control_variables], make_jerk(horizon), horizon - 1, cost_function_weights[1],
            aux_vars=[], autograd_vectorize=False, name="jerk"
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            [control_variables], make_steering(horizon), horizon, cost_function_weights[2],
            aux_vars=[], autograd_vectorize=False, name="steering"
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            [control_variables], make_steering_change(horizon), horizon - 1, cost_function_weights[3],
            aux_vars=[], autograd_vectorize=False, name="steering_change"
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            [control_variables], make_lane_xy(horizon), horizon, cost_function_weights[4],
            aux_vars=[ref_line, ego_state], autograd_vectorize=False, name="lane_xy"
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            [control_variables], make_lane_theta(horizon), lane_theta_dim, cost_function_weights[5],
            aux_vars=[ref_line, ego_state], autograd_vectorize=False, name="lane_theta"
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            [control_variables], make_route_target(horizon), route_target_dim, cost_function_weights[6],
            aux_vars=[ego_state], autograd_vectorize=False, name="route_target"
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            [control_variables], make_safety(horizon), safety_dim, cost_function_weights[7],
            aux_vars=[
                ego_state,
                agents_future,
                agents_mask,
                agent_safety_distance,
                agent_lateral_safety_distance,
            ],
            autograd_vectorize=False, name="safe"
        )
    )
    objective.add(
        th.AutoDiffCostFunction(
            [control_variables],
            make_control_anchor(horizon),
            horizon * 2,
            control_anchor_weight,
            aux_vars=[initial_control_reference],
            autograd_vectorize=False,
            name="control_anchor",
        )
    )
    return objective


class MotionPlannerCompatible:
    def __init__(
        self,
        trajectory_len: int,
        feature_len: int,
        num_objects: int,
        device: torch.device,
        optimizer_type: str = "levenberg_marquardt",
        max_iterations: int = 10,
        step_size: float = 0.10,
    ):
        self.device = device
        self.trajectory_len = trajectory_len
        self.feature_len = feature_len
        self.num_objects = num_objects
        self.control_dim = 2
        self.control_dof = trajectory_len * self.control_dim

        cost_function_weights = [
            th.ScaleCostWeight(th.Variable(torch.ones(1), name=f"cost_function_weight_{i+1}"))
            for i in range(feature_len)
        ]
        control_anchor_weight = th.ScaleCostWeight(
            th.Variable(torch.ones(1), name="control_anchor_weight")
        )

        control_variables = th.Vector(dof=self.control_dof, name="control_variables")
        ego_state = th.Variable(torch.empty(1, 10), name="ego_state")
        ref_line_info = th.Variable(torch.empty(1, 20, 3), name="ref_line_info")
        agents_future = th.Variable(torch.empty(1, num_objects, trajectory_len, 2), name="agents_future")
        agents_mask = th.Variable(torch.empty(1, num_objects), name="agents_mask")
        agent_safety_distance = th.Variable(
            torch.empty(1, num_objects), name="agent_safety_distance"
        )
        agent_lateral_safety_distance = th.Variable(
            torch.empty(1, num_objects), name="agent_lateral_safety_distance"
        )
        initial_control_reference = th.Variable(
            torch.empty(1, self.control_dof), name="initial_control_reference"
        )
        horizon_tensor = th.Variable(torch.tensor([float(trajectory_len)], dtype=torch.float32), name="horizon")

        objective = th.Objective()
        self.objective = build_objective(
            objective=objective,
            control_variables=control_variables,
            ego_state=ego_state,
            ref_line=ref_line_info,
            agents_future=agents_future,
            agents_mask=agents_mask,
            agent_safety_distance=agent_safety_distance,
            agent_lateral_safety_distance=agent_lateral_safety_distance,
            initial_control_reference=initial_control_reference,
            cost_function_weights=cost_function_weights,
            control_anchor_weight=control_anchor_weight,
            horizon_tensor=horizon_tensor,
            num_objects=num_objects,
        )

        self.optimizer_type = optimizer_type
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.optimizer = self._build_optimizer(objective)

        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(device=device)

    def _build_optimizer(self, objective):
        requested = str(self.optimizer_type).lower()
        if requested in {"lm", "levenberg", "levenberg_marquardt"} and hasattr(th, "LevenbergMarquardt"):
            try:
                return th.LevenbergMarquardt(
                    objective,
                    th.LUDenseSolver,
                    vectorize=False,
                    max_iterations=self.max_iterations,
                    step_size=self.step_size,
                )
            except TypeError:
                pass

        return th.GaussNewton(
            objective,
            th.LUDenseSolver,
            vectorize=False,
            max_iterations=self.max_iterations,
            step_size=self.step_size,
        )


def canonicalize_lane_direction(lane_points: torch.Tensor) -> torch.Tensor:
    """Orient PNN-frame lane polylines from near to far (increasing x)."""
    if lane_points.dim() != 4 or lane_points.size(-1) != 2:
        raise ValueError(f"lane_points must have shape [B,L,T,2], got {tuple(lane_points.shape)}")
    window = max(1, lane_points.size(2) // 4)
    start_x = lane_points[:, :, :window, 0].median(dim=2).values
    end_x = lane_points[:, :, -window:, 0].median(dim=2).values
    reverse = end_x < start_x
    flipped = torch.flip(lane_points, dims=(2,))
    return torch.where(reverse[:, :, None, None], flipped, lane_points)


def build_ref_line_from_xy(points: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if points.dim() != 3 or points.size(-1) != 2 or points.size(1) < 2:
        raise ValueError(f"reference points must have shape [B,T,2], got {tuple(points.shape)}")
    centerline = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    delta = centerline[:, 1:, :] - centerline[:, :-1, :]
    delta_norm = torch.norm(delta, dim=-1)
    theta_raw = torch.atan2(delta[:, :, 1], delta[:, :, 0])
    theta_raw = torch.where(delta_norm > eps, theta_raw, torch.zeros_like(theta_raw))
    theta_list = []
    prev = theta_raw[:, 0]
    theta_list.append(prev)
    for t in range(1, theta_raw.shape[1]):
        cur = torch.where(delta_norm[:, t] > eps, theta_raw[:, t], prev)
        theta_list.append(cur)
        prev = cur
    theta = torch.stack(theta_list, dim=1)
    theta = torch.cat([theta, theta[:, -1:]], dim=1)
    theta = torch.atan2(torch.sin(theta), torch.cos(theta))
    return torch.cat([centerline, theta.unsqueeze(-1)], dim=-1)


def build_ref_line_from_two_boundaries(lane_points_2: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if lane_points_2.dim() != 4 or lane_points_2.size(1) < 2 or lane_points_2.size(-1) != 2:
        raise ValueError(
            f"lane_points_2 should have shape [B,2,T,2] or [B,>=2,T,2], got {tuple(lane_points_2.shape)}"
        )

    lane_points_2 = canonicalize_lane_direction(
        torch.nan_to_num(lane_points_2, nan=0.0, posinf=0.0, neginf=0.0)
    )
    left_line = lane_points_2[:, 0]
    right_line = lane_points_2[:, 1]
    centerline = 0.5 * (left_line + right_line)

    return build_ref_line_from_xy(centerline, eps=eps)


def adapt_lane_points_2_to_10(lane_points_2: torch.Tensor) -> torch.Tensor:
    left_line = lane_points_2[:, 0:1]
    right_line = lane_points_2[:, 1:2]
    alphas = torch.linspace(0.0, 1.0, 10, device=lane_points_2.device, dtype=lane_points_2.dtype)
    alphas = alphas.view(1, 10, 1, 1)
    return (1.0 - alphas) * left_line + alphas * right_line


def split_lane_for_control_and_weight(lane_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lane_points = canonicalize_lane_direction(lane_points)
    if lane_points.shape[1] == 2:
        return lane_points, adapt_lane_points_2_to_10(lane_points)
    if lane_points.shape[1] == 10:
        return lane_points[:, :2], lane_points
    raise ValueError(f"lane_points second dim should be 2 or 10, got {lane_points.shape[1]}")


def build_agents_future_from_states(
    ped_states: torch.Tensor,
    veh_states: torch.Tensor,
    ped_mask: Optional[torch.Tensor],
    veh_mask: Optional[torch.Tensor],
    horizon: int = TRAJ_LEN,
):
    B, Np, _ = ped_states.shape
    Nv = veh_states.shape[1]
    device = ped_states.device
    dtype = ped_states.dtype

    alphas = torch.linspace(0.0, 1.0, horizon, device=device, dtype=dtype).view(1, 1, horizon, 1)

    ped_start = ped_states[:, :, 0:2].unsqueeze(2)
    ped_goal = ped_states[:, :, 4:6].unsqueeze(2)
    ped_future = (1.0 - alphas) * ped_start + alphas * ped_goal

    veh_start = veh_states[:, :, 0:2].unsqueeze(2)
    veh_goal = veh_states[:, :, 4:6].unsqueeze(2)
    veh_future = (1.0 - alphas) * veh_start + alphas * veh_goal

    agents_future = torch.cat([ped_future, veh_future], dim=1)

    if ped_mask is None:
        ped_mask = torch.ones((B, Np), device=device, dtype=torch.bool)
    if veh_mask is None:
        veh_mask = torch.ones((B, Nv), device=device, dtype=torch.bool)
    agents_mask = torch.cat([ped_mask, veh_mask], dim=1)
    return agents_future, agents_mask


def build_fixed_agent_safety_trajectories(
    states: torch.Tensor,
    horizon: int = TRAJ_LEN,
    dt: float = DT,
) -> torch.Tensor:
    """Build detached exogenous actor trajectories from state start/end points."""
    B, N, _ = states.shape
    if N == 0:
        return states.new_zeros((B, 0, horizon, 4))

    start = states[:, :, 0:2]
    goal = states[:, :, 4:6]
    direction = goal - start
    moving = torch.norm(direction, dim=-1) > 1e-3
    goal_heading = torch.atan2(direction[..., 1], direction[..., 0])
    heading = torch.where(moving, goal_heading, states[:, :, 2])
    speed = torch.norm(direction, dim=-1) / max(float(horizon) * float(dt), 1e-6)

    # rollout_all_agents stores the first post-update state, so use (t + 1) / T.
    alphas = torch.arange(
        1,
        horizon + 1,
        device=states.device,
        dtype=states.dtype,
    ).view(1, 1, horizon, 1) / float(horizon)
    xy = start.unsqueeze(2) + alphas * direction.unsqueeze(2)
    heading = heading.unsqueeze(2).expand(-1, -1, horizon)
    speed = speed.unsqueeze(2).expand(-1, -1, horizon)
    return torch.cat([xy, heading.unsqueeze(-1), speed.unsqueeze(-1)], dim=-1).detach()


def compute_rollout_collision_risk(
    ego_traj: torch.Tensor,
    ped_states: torch.Tensor,
    veh_states: torch.Tensor,
    ped_mask: Optional[torch.Tensor],
    veh_mask: Optional[torch.Tensor],
    safe_dist: float = 10.0,
    collision_dist: float = SAFE_MARGIN,
    sharpness: float = 1.5,
) -> torch.Tensor:
    agents_future, agents_mask = build_agents_future_from_states(
        ped_states,
        veh_states,
        ped_mask,
        veh_mask,
        horizon=ego_traj.shape[1],
    )
    ego_xy = ego_traj[:, :, :2]
    d = torch.norm(ego_xy.unsqueeze(1) - agents_future, dim=-1)
    d = torch.where(agents_mask.unsqueeze(-1).bool(), d, torch.full_like(d, safe_dist * 5.0))
    min_future_dist = d.amin(dim=(1, 2))
    return torch.sigmoid((float(collision_dist) - min_future_dist) / float(sharpness))


def compute_collision_ttc_residual(
    ego_xy: torch.Tensor,
    agents_xy: torch.Tensor,
    agents_mask: Optional[torch.Tensor],
    collision_dist: float = SAFE_MARGIN,
    collision_lateral_dist=None,
    ttc_horizon: float = TTC_HORIZON,
    distance_weight: float = 1.0,
    ttc_weight: float = TTC_WEIGHT,
    route_direction: Optional[torch.Tensor] = None,
    no_pass_weight: float = 2.0,
) -> torch.Tensor:
    B, Nobj, T, _ = agents_xy.shape
    ego_xy_expand = ego_xy.unsqueeze(1).expand(-1, Nobj, -1, -1)
    rel_pos = agents_xy - ego_xy_expand
    euclidean_dist = torch.norm(rel_pos, dim=-1).clamp_min(TTC_EPS)

    def expand_margin(value, name):
        value = torch.as_tensor(value, device=euclidean_dist.device, dtype=euclidean_dist.dtype)
        if value.ndim == 0:
            return value
        if value.ndim == 1:
            if value.numel() == Nobj:
                return value.view(1, Nobj, 1)
            if value.numel() == B:
                return value.view(B, 1, 1)
            raise ValueError(f"{name} length must be B={B} or Nobj={Nobj}")
        if value.ndim == 2 and value.shape == (B, Nobj):
            return value.unsqueeze(-1)
        raise ValueError(
            f"{name} must be scalar, [Nobj], [B], or [B,Nobj], got {tuple(value.shape)}"
        )

    margin = expand_margin(collision_dist, "collision_dist")
    lateral_margin = None
    if collision_lateral_dist is not None:
        lateral_margin = expand_margin(collision_lateral_dist, "collision_lateral_dist")

    route_dir = None
    long_gap = None
    lateral_gap = None
    dist = euclidean_dist
    if route_direction is not None:
        route_dir = F.normalize(route_direction.to(euclidean_dist), dim=-1, eps=1e-4)
        route_perp = torch.stack([-route_dir[:, 1], route_dir[:, 0]], dim=-1)
        long_gap = (rel_pos * route_dir[:, None, None, :]).sum(dim=-1)
        lateral_gap = (rel_pos * route_perp[:, None, None, :]).sum(dim=-1).abs()
        if lateral_margin is not None:
            lateral_scale = margin / lateral_margin.clamp_min(0.1)
            dist = torch.sqrt(
                long_gap.square()
                + (lateral_gap * lateral_scale).square()
                + TTC_EPS ** 2
            )

    distance_residual = F.softplus(
        COLLISION_BARRIER_SHARPNESS * (margin - dist)
    ) / COLLISION_BARRIER_SHARPNESS

    no_pass_residual = torch.zeros_like(distance_residual)
    if route_dir is not None:
        corridor_half_width = lateral_margin if lateral_margin is not None else 0.6 * margin
        corridor_gate = torch.sigmoid((corridor_half_width - lateral_gap) / 0.30)
        initially_ahead = (long_gap[:, :, :1] > 0.5).to(dist.dtype)
        no_pass_residual = initially_ahead * corridor_gate * (
            F.softplus(
                COLLISION_BARRIER_SHARPNESS * (margin - long_gap)
            ) / COLLISION_BARRIER_SHARPNESS
        )

    if T > 1:
        dist_prev = dist[:, :, :-1]
        closing_speed = (dist_prev - dist[:, :, 1:]) / DT
        time_to_margin = (dist_prev - margin) / closing_speed.clamp_min(TTC_EPS)
        ttc_residual = F.relu((float(ttc_horizon) - time_to_margin) / float(ttc_horizon))
        ttc_residual = torch.where(
            closing_speed > TTC_EPS,
            ttc_residual,
            torch.zeros_like(ttc_residual),
        )
        ttc_residual = F.pad(ttc_residual, (0, 1))
    else:
        ttc_residual = torch.zeros_like(distance_residual)

    residual = (
        float(distance_weight) * distance_residual
        + float(ttc_weight) * ttc_residual
        + float(no_pass_weight) * no_pass_residual
    )
    if agents_mask is not None:
        residual = residual * agents_mask.unsqueeze(-1).to(residual.dtype)
    return residual


def compute_gt_actor_box_violation(
    ego_traj: torch.Tensor,
    gt_actor_boxes_2hz: torch.Tensor,
    gt_actor_mask_2hz: torch.Tensor,
    rect_distance: nn.Module,
    sample_valid_mask: Optional[torch.Tensor] = None,
    frame_valid_mask: Optional[torch.Tensor] = None,
    margin: float = 0.0,
    topk: int = 1,
    smooth_temperature: float = 0.0,
    time_weights: Optional[Tuple[float, ...]] = None,
    return_per_frame: bool = False,
) -> torch.Tensor:
    """Differentiable counterpart of the official 2 Hz occupancy collision.

    Actor boxes are (x,y,yaw,length,width) in the PNN frame. The official STP3
    ego footprint is centered 0.5 m ahead of each trajectory point.
    """
    B, T, _ = ego_traj.shape
    if T <= gt_actor_boxes_2hz.shape[1]:
        # Full HiPAD plans are already sampled at the same 2 Hz as GT boxes.
        frame_indices = list(range(T))
    else:
        frame_indices = [min(i, T - 1) for i in (4, 9, 14, 19, 24, 29)]
    num_frames = min(len(frame_indices), gt_actor_boxes_2hz.shape[1])
    per_frame = []
    for frame_slot, traj_idx in enumerate(frame_indices[:num_frames]):
        actor = gt_actor_boxes_2hz[:, frame_slot]
        actor_mask = gt_actor_mask_2hz[:, frame_slot].bool()
        A = actor.shape[1]
        if A == 0:
            per_frame.append(ego_traj.new_zeros(B))
            continue

        ego = ego_traj[:, traj_idx]
        ego_center = ego.new_zeros(B, 4)
        ego_center[:, :2] = ego[:, :2]
        # STP3 translates an axis-aligned current-ego footprint; it does not
        # rotate the footprint with the future trajectory tangent.
        ego_center[:, 0] = ego_center[:, 0] + OFFICIAL_EGO_CENTER_FORWARD_OFFSET
        ego_center[:, 2] = 0.0
        ego_expand = ego_center.unsqueeze(1).expand(-1, A, -1)
        actor_state = torch.cat(
            [actor[..., :3], torch.zeros_like(actor[..., :1])], dim=-1
        )
        separation, penetration = rect_distance(
            ego_expand.reshape(B * A, 4),
            (OFFICIAL_EGO_LENGTH, OFFICIAL_EGO_WIDTH),
            actor_state.reshape(B * A, 4),
            (actor[..., 3].reshape(-1), actor[..., 4].reshape(-1)),
        )
        signed_separation = (separation - penetration).reshape(B, A)
        residual = float(margin) - signed_separation
        if float(smooth_temperature) > 0.0:
            temperature = float(smooth_temperature)
            violation = temperature * F.softplus(residual / temperature)
        else:
            violation = F.relu(residual)
        violation = violation * actor_mask.to(violation.dtype)
        k = min(max(int(topk), 1), A)
        top = torch.topk(violation, k=k, dim=1).values
        frame_violation = 0.7 * top[:, 0] + 0.3 * top.mean(dim=1)
        per_frame.append(frame_violation)

    if not per_frame:
        values = ego_traj.new_zeros((B, 0))
        result = ego_traj.new_zeros(B)
    else:
        values = torch.stack(per_frame, dim=1)
        if frame_valid_mask is not None:
            valid = frame_valid_mask[:, :values.shape[1]].to(
                device=values.device, dtype=values.dtype
            )
        else:
            valid = torch.ones_like(values)
        if time_weights is None:
            weights = torch.ones(values.shape[1], device=values.device, dtype=values.dtype)
        else:
            weights = torch.as_tensor(time_weights, device=values.device, dtype=values.dtype).flatten()
            if weights.numel() != values.shape[1]:
                weights = F.interpolate(
                    weights.view(1, 1, -1),
                    size=values.shape[1],
                    mode="linear",
                    align_corners=True,
                ).view(-1)
        weighted_valid = valid * weights.view(1, -1)
        result = (values * weighted_valid).sum(dim=1) / weighted_valid.sum(dim=1).clamp_min(1.0)
    if sample_valid_mask is not None:
        result = result * sample_valid_mask.reshape(B).to(result.dtype)
        values = values * sample_valid_mask.reshape(B, 1).to(values.dtype)
    if frame_valid_mask is not None and values.numel():
        values = values * frame_valid_mask[:, :values.shape[1]].to(values.dtype)
    return values if return_per_frame else result


def compute_static_detection_box_violation(
    ego_traj: torch.Tensor,
    static_states: torch.Tensor,
    static_mask: torch.Tensor,
    rect_distance: nn.Module,
    margin: float = 0.60,
    smooth_temperature: float = 0.20,
    max_distance: float = 55.0,
    ego_z_min: float = -1.90,
    ego_z_max: float = 0.80,
    time_weights: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    return_per_frame: bool = False,
) -> torch.Tensor:
    """Penalize the rollout against collision-relevant HiP-AD static boxes.

    This uses predictions only. Vertical overlap removes overhead traffic-light
    boxes, while the XY oriented rectangle test matches the existing actor-box
    safety geometry.
    """
    B, S, _ = static_states.shape
    if S == 0:
        return ego_traj.new_zeros(B)
    z = static_states[..., 2]
    height = static_states[..., 6].clamp_min(0.05)
    bottom = z - 0.5 * height
    top = z + 0.5 * height
    vertical_overlap = (top >= float(ego_z_min)) & (bottom <= float(ego_z_max))
    nearby = torch.linalg.norm(static_states[..., :2], dim=-1) <= float(max_distance)
    valid = static_mask.bool() & vertical_overlap & nearby

    box = torch.stack(
        [
            static_states[..., 0],
            static_states[..., 1],
            static_states[..., 3],
            static_states[..., 4].clamp_min(0.05),
            static_states[..., 5].clamp_min(0.05),
        ],
        dim=-1,
    )
    boxes_2hz = box.unsqueeze(1).expand(B, 6, S, 5)
    mask_2hz = valid.unsqueeze(1).expand(B, 6, S)
    return compute_gt_actor_box_violation(
        ego_traj=ego_traj,
        gt_actor_boxes_2hz=boxes_2hz,
        gt_actor_mask_2hz=mask_2hz,
        rect_distance=rect_distance,
        margin=margin,
        topk=1,
        smooth_temperature=smooth_temperature,
        time_weights=time_weights,
        return_per_frame=return_per_frame,
    )


def _sample_trajectory_at_official_2hz(ego_traj: torch.Tensor) -> torch.Tensor:
    """Select the same six 2 Hz positions used by the official planner metric."""
    if ego_traj.shape[1] <= 6:
        return ego_traj[:, :6, :2]
    frame_indices = [min(i, ego_traj.shape[1] - 1) for i in (4, 9, 14, 19, 24, 29)]
    return ego_traj[:, frame_indices, :2]


def compute_official_actor_raster_proxy_gate(
    ego_traj: torch.Tensor,
    gt_actor_boxes_2hz: torch.Tensor,
    gt_actor_mask_2hz: torch.Tensor,
    sample_valid_mask: Optional[torch.Tensor] = None,
    frame_valid_mask: Optional[torch.Tensor] = None,
    actor_raster_padding: float = 0.12,
    actor_raster_dilation_pixels: float = 0.0,
) -> torch.Tensor:
    """Approximate the official raster collision test at each 2 Hz frame.

    The official metric rasterizes actors at 0.5 m and translates a fixed set
    of 32 ego-footprint pixels. Here those exact footprint offsets and 0.5 m
    center quantization are queried against the corrected future actor boxes.
    The detached gate chooses hard frames; continuous box distance supplies the
    optimization gradient separately.
    """
    traj_2hz = _sample_trajectory_at_official_2hz(ego_traj)
    num_frames = min(traj_2hz.shape[1], gt_actor_boxes_2hz.shape[1])
    traj_2hz = traj_2hz[:, :num_frames]
    actors = gt_actor_boxes_2hz[:, :num_frames]
    actor_mask = gt_actor_mask_2hz[:, :num_frames].bool()
    if num_frames == 0 or actors.shape[2] == 0:
        return ego_traj.new_zeros((ego_traj.shape[0], num_frames))

    # Offsets are the 32 pixels generated by skimage.draw.polygon in
    # PlanningMetric.evaluate_single_coll for the 4.084 m x 1.85 m ego box.
    row_offsets = torch.arange(-3, 5, device=ego_traj.device, dtype=ego_traj.dtype)
    col_offsets = torch.arange(-2, 2, device=ego_traj.device, dtype=ego_traj.dtype)
    row_grid, col_grid = torch.meshgrid(row_offsets, col_offsets, indexing="ij")
    row_offsets = row_grid.reshape(-1)
    col_offsets = col_grid.reshape(-1)
    query_row = torch.trunc(
        200.0 - (traj_2hz[..., 0, None] / 0.5 + 100.0 + row_offsets)
    )
    query_col = torch.trunc(
        -traj_2hz[..., 1, None] / 0.5 + 100.0 + col_offsets
    )
    query_pixels = torch.stack(
        [query_row.clamp(0.0, 199.0), query_col.clamp(0.0, 199.0)], dim=-1
    )

    half_length = 0.5 * actors[..., 3] + float(actor_raster_padding)
    half_width = 0.5 * actors[..., 4] + float(actor_raster_padding)
    local_signs = actors.new_tensor(
        ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
    )
    local_corners = torch.stack([half_length, half_width], dim=-1).unsqueeze(-2)
    local_corners = local_corners * local_signs
    yaw = actors[..., 2].unsqueeze(-1)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    corner_x = (
        local_corners[..., 0] * cos_yaw
        - local_corners[..., 1] * sin_yaw
        + actors[..., 0].unsqueeze(-1)
    )
    corner_y = (
        local_corners[..., 0] * sin_yaw
        + local_corners[..., 1] * cos_yaw
        + actors[..., 1].unsqueeze(-1)
    )
    actor_pixels = torch.stack(
        [
            torch.round(100.0 - corner_x / 0.5),
            torch.round(100.0 - corner_y / 0.5),
        ],
        dim=-1,
    )
    edge = torch.roll(actor_pixels, shifts=-1, dims=-2) - actor_pixels
    point_delta = (
        query_pixels[:, :, :, None, None, :]
        - actor_pixels[:, :, None, :, :, :]
    )
    cross = (
        edge[:, :, None, :, :, 0] * point_delta[..., 1]
        - edge[:, :, None, :, :, 1] * point_delta[..., 0]
    )
    # cv2.fillPoly includes raster boundary pixels that a strict continuous
    # point-in-polygon test can omit. A small edge-normalized tolerance makes
    # the detached training gate reproduce that behavior without changing the
    # continuous box-distance term that supplies gradients.
    edge_tolerance = (
        edge.square().sum(dim=-1).sqrt()
        * float(actor_raster_dilation_pixels)
    )
    inside_polygon = (
        cross >= -edge_tolerance[:, :, None, :, :]
    ).all(dim=-1) | (
        cross <= edge_tolerance[:, :, None, :, :]
    ).all(dim=-1)
    occupied = inside_polygon & actor_mask[:, :, None, :]
    gate = occupied.any(dim=-1).any(dim=-1).to(ego_traj.dtype)
    if frame_valid_mask is not None:
        gate = gate * frame_valid_mask[:, :num_frames].to(gate.dtype)
    if sample_valid_mask is not None:
        gate = gate * sample_valid_mask.reshape(-1, 1).to(gate.dtype)
    return gate


def positive_normalized_frame_loss_per_sample(
    frame_loss: torch.Tensor,
    positive_mask: torch.Tensor,
    time_weights: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    """Return per-sample values whose DDP mean equals a global positive mean."""
    mask = positive_mask.to(frame_loss.dtype)
    if time_weights is not None:
        weights = torch.as_tensor(
            time_weights, device=frame_loss.device, dtype=frame_loss.dtype
        )[:frame_loss.shape[1]]
        mask = mask * weights.view(1, -1)
    positive_count = mask.detach().sum()
    if dist.is_initialized():
        dist.all_reduce(positive_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1
    if positive_count.item() <= 0:
        return frame_loss.new_zeros(frame_loss.shape[0])
    scale = frame_loss.shape[0] * world_size / positive_count.clamp_min(1.0)
    return (frame_loss * mask).sum(dim=1) * scale


def sanitize_planner_tensor(x: torch.Tensor, clamp_abs: Optional[float] = None) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_abs is not None:
        x = x.clamp(min=-float(clamp_abs), max=float(clamp_abs))
    return x


def _expand_cost_vector(
    value,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        x = value.to(device=device, dtype=dtype)
    else:
        x = torch.tensor(value, device=device, dtype=dtype)

    if x.dim() == 0:
        return x.view(1, 1).expand(batch_size, NUM_COSTS)
    if x.dim() == 1:
        if x.numel() == 1:
            return x.view(1, 1).expand(batch_size, NUM_COSTS)
        if x.numel() != NUM_COSTS:
            raise ValueError(f"{name} must be scalar or length {NUM_COSTS}, got shape={tuple(x.shape)}")
        return x.view(1, NUM_COSTS).expand(batch_size, -1)
    if x.dim() == 2:
        if x.shape == (batch_size, NUM_COSTS):
            return x
        if x.shape == (1, NUM_COSTS):
            return x.expand(batch_size, -1)
    raise ValueError(f"{name} must be scalar, [{NUM_COSTS}], or [{batch_size},{NUM_COSTS}], got {tuple(x.shape)}")


def sanitize_planner_cost_weights(
    cost_weights: torch.Tensor,
    min_weight=1e-3,
    max_weight=20.0,
    fallback_weights: Optional[torch.Tensor] = None,
    renormalize_to_fallback_sum: bool = True,
) -> torch.Tensor:
    B = cost_weights.shape[0]
    device = cost_weights.device
    dtype = cost_weights.dtype

    if fallback_weights is None:
        fallback_weights = torch.tensor(
            [1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 3.0, 2.0],
            dtype=dtype,
            device=device,
        ).unsqueeze(0).expand(B, -1)
    else:
        fallback_weights = _expand_cost_vector(fallback_weights, B, device, dtype, "fallback_weights")

    min_w = _expand_cost_vector(min_weight, B, device, dtype, "min_weight")
    max_w = _expand_cost_vector(max_weight, B, device, dtype, "max_weight")
    if torch.any(max_w <= min_w):
        raise ValueError("max_weight must be greater than min_weight for every cost component")

    finite_mask = torch.isfinite(cost_weights).all(dim=-1, keepdim=True)
    x = torch.where(torch.isfinite(cost_weights), cost_weights, fallback_weights)
    x = torch.where(finite_mask, x, fallback_weights)
    x = torch.maximum(torch.minimum(x, max_w), min_w)

    if renormalize_to_fallback_sum:
        min_sum = min_w.sum(dim=-1, keepdim=True)
        target_sum = fallback_weights.sum(dim=-1, keepdim=True).clamp_min(min_sum)
        x = x / x.sum(dim=-1, keepdim=True).clamp_min(min_sum) * target_sum
        x = torch.maximum(torch.minimum(x, max_w), min_w)
    return x


def is_theseus_singular_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return (
        "singular" in msg
        or "linear optimizer" in msg
        or "torch.linalg.solve" in msg
        or "input matrix" in msg
        or "linalgerror" in msg
        or "cuda driver error" in msg
        or "invalid argument" in msg
        or "cudacachingallocator" in msg
        or "internal assert failed" in msg
        or "!handles_.at" in msg
        or "please report a bug to pytorch" in msg
        or "functorch" in msg
        or "jacrev" in msg
        or "vmap" in msg
    )


def run_upper_dipp(
    planner: MotionPlannerCompatible,
    ego_state: torch.Tensor,
    lane_points_control: torch.Tensor,
    init_control: torch.Tensor,
    cost_weights: torch.Tensor,
    ped_states: torch.Tensor,
    veh_states: torch.Tensor,
    ped_mask: Optional[torch.Tensor],
    veh_mask: Optional[torch.Tensor],
    planner_weight_min: float = 1e-3,
    planner_weight_max=20.0,
    planner_weight_renormalize_to_default_sum: bool = True,
    ped_safety_distance: float = 2.5,
    veh_safety_distance: float = 4.0,
    ped_lateral_safety_distance: float = 1.2,
    veh_lateral_safety_distance: float = 1.8,
    control_anchor_weight: float = 500.0,
    control_anchor_risk_floor: float = 0.05,
):
    ego_state = sanitize_planner_tensor(ego_state, clamp_abs=1e4)
    lane_points_control = sanitize_planner_tensor(lane_points_control, clamp_abs=1e4)
    init_control = sanitize_planner_tensor(init_control, clamp_abs=50.0)
    ped_states = sanitize_planner_tensor(ped_states, clamp_abs=1e4)
    veh_states = sanitize_planner_tensor(veh_states, clamp_abs=1e4)

    init_control = torch.stack(
        [
            init_control[..., 0].clamp(-MAX_ACC, MAX_ACC),
            init_control[..., 1].clamp(-MAX_STEER, MAX_STEER),
        ],
        dim=-1,
    )
    cost_weights = sanitize_planner_cost_weights(
        cost_weights,
        min_weight=planner_weight_min,
        max_weight=planner_weight_max,
        renormalize_to_fallback_sum=planner_weight_renormalize_to_default_sum,
    )

    ref_line_info = build_ref_line_from_two_boundaries(lane_points_control)
    agents_future, agents_mask = build_agents_future_from_states(
        ped_states, veh_states, ped_mask, veh_mask, horizon=TRAJ_LEN
    )
    agents_future = sanitize_planner_tensor(agents_future, clamp_abs=1e4)
    agent_safety_distance = torch.cat(
        [
            torch.full(
                (ego_state.shape[0], ped_states.shape[1]),
                float(ped_safety_distance),
                device=ego_state.device,
                dtype=ego_state.dtype,
            ),
            torch.full(
                (ego_state.shape[0], veh_states.shape[1]),
                float(veh_safety_distance),
                device=ego_state.device,
                dtype=ego_state.dtype,
            ),
        ],
        dim=1,
    )
    agent_lateral_safety_distance = torch.cat(
        [
            torch.full(
                (ego_state.shape[0], ped_states.shape[1]),
                float(ped_lateral_safety_distance),
                device=ego_state.device,
                dtype=ego_state.dtype,
            ),
            torch.full(
                (ego_state.shape[0], veh_states.shape[1]),
                float(veh_lateral_safety_distance),
                device=ego_state.device,
                dtype=ego_state.dtype,
            ),
        ],
        dim=1,
    )

    with torch.no_grad():
        initial_traj = bicycle_model_compatible(init_control, ego_state)
        initial_safety_residual = compute_collision_ttc_residual(
            ego_xy=initial_traj[:, :, :2],
            agents_xy=agents_future,
            agents_mask=agents_mask,
            collision_dist=agent_safety_distance,
            collision_lateral_dist=agent_lateral_safety_distance,
            route_direction=ego_state[:, 8:10] - ego_state[:, :2],
        )
        initial_risk = 1.0 - torch.exp(
            -initial_safety_residual.amax(dim=(1, 2)).clamp_min(0.0)
        )
        anchor_scale = 1.0 - (
            1.0 - float(control_anchor_risk_floor)
        ) * initial_risk
        adaptive_anchor_weight = float(control_anchor_weight) * anchor_scale

    B = ego_state.shape[0]
    planner_inputs = {
        "control_variables": init_control.contiguous().view(B, TRAJ_LEN * 2),
        "ego_state": ego_state,
        "ref_line_info": ref_line_info,
        "agents_future": agents_future,
        "agents_mask": agents_mask.to(ego_state.dtype),
        "agent_safety_distance": agent_safety_distance,
        "agent_lateral_safety_distance": agent_lateral_safety_distance,
        "initial_control_reference": init_control.contiguous().view(B, TRAJ_LEN * 2),
        "control_anchor_weight": torch.full(
            (B, 1), 1.0, device=ego_state.device, dtype=ego_state.dtype
        ) * adaptive_anchor_weight.view(B, 1),
    }
    for i in range(NUM_COSTS):
        planner_inputs[f"cost_function_weight_{i+1}"] = cost_weights[:, i:i + 1]

    updated_inputs, info = planner.layer.forward(
        planner_inputs,
        optimizer_kwargs={
            "track_best_solution": False,
            "backward_mode": th.BackwardMode.UNROLL,
        },
    )

    best_control = updated_inputs["control_variables"].to(
        device=ego_state.device,
        dtype=ego_state.dtype,
    ).view(B, TRAJ_LEN, 2)
    best_control = sanitize_planner_tensor(best_control, clamp_abs=50.0)
    best_control = torch.stack(
        [
            best_control[..., 0].clamp(-MAX_ACC, MAX_ACC),
            best_control[..., 1].clamp(-MAX_STEER, MAX_STEER),
        ],
        dim=-1,
    )

    ego_dipp_traj = bicycle_model_compatible(best_control, ego_state)
    ego_dipp_traj = sanitize_planner_tensor(ego_dipp_traj, clamp_abs=1e4)
    return best_control, ego_dipp_traj, ref_line_info


# ============================================================
# Weight model helpers
# ============================================================
def init_weight_decoder_with_default_prior(
    model: nn.Module,
    prior_weights=(1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 3.0, 2.0),
    residual_output: bool = True,
):
    last_linear = model.weight_decoder[-1]
    with torch.no_grad():
        nn.init.zeros_(last_linear.weight)
        if residual_output:
            nn.init.zeros_(last_linear.bias)
        else:
            prior = torch.tensor(prior_weights, dtype=last_linear.bias.dtype, device=last_linear.bias.device)
            prior = prior / prior.sum()
            prior_logits = torch.log(prior.clamp_min(1e-8))
            last_linear.bias.copy_(prior_logits)


@torch.no_grad()
def ema_update_weight_encoder_from_control(
    weight_model: nn.Module,
    control_core: nn.Module,
    beta: float = 0.99,
    names: Optional[List[str]] = None,
):
    src = unwrap_module(control_core)
    dst = unwrap_module(weight_model)
    if names is None:
        names = ["ego_encoder", "ped_encoder", "veh_encoder", "map_encoder"]

    for name in names:
        if not (hasattr(src, name) and hasattr(dst, name)):
            continue
        src_m = getattr(src, name)
        dst_m = getattr(dst, name)

        src_params = dict(src_m.named_parameters())
        dst_params = dict(dst_m.named_parameters())
        for p_name, p_dst in dst_params.items():
            if p_name not in src_params:
                continue
            p_src = src_params[p_name]
            p_dst.data.mul_(beta).add_(p_src.data, alpha=(1.0 - beta))

        src_buffers = dict(src_m.named_buffers())
        dst_buffers = dict(dst_m.named_buffers())
        for b_name, b_dst in dst_buffers.items():
            if b_name not in src_buffers:
                continue
            b_dst.data.copy_(src_buffers[b_name].data)


def build_default_cost_weights(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    weights=None,
) -> torch.Tensor:
    if weights is None:
        weights = [1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 3.0, 2.0]
    if len(weights) != NUM_COSTS:
        raise ValueError(f"default cost weights must contain {NUM_COSTS} values, got {weights}")
    return torch.tensor(weights, dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1)


def compute_min_agent_distance(
    ego_state: torch.Tensor,
    ped_states: Optional[torch.Tensor],
    veh_states: Optional[torch.Tensor],
    ped_mask: Optional[torch.Tensor],
    veh_mask: Optional[torch.Tensor],
    far_value: float = 1e6,
) -> torch.Tensor:
    device = ego_state.device
    dtype = ego_state.dtype
    B = ego_state.shape[0]

    ego_xy = ego_state[:, :2]
    min_dist = torch.full((B,), far_value, device=device, dtype=dtype)

    if ped_states is not None and ped_states.numel() > 0:
        ped_xy = ped_states[:, :, :2]
        d_ped = torch.norm(ped_xy - ego_xy.unsqueeze(1), dim=-1)
        if ped_mask is not None:
            d_ped = torch.where(ped_mask, d_ped, torch.full_like(d_ped, far_value))
        min_dist = torch.minimum(min_dist, d_ped.min(dim=1).values)

    if veh_states is not None and veh_states.numel() > 0:
        veh_xy = veh_states[:, :, :2]
        d_veh = torch.norm(veh_xy - ego_xy.unsqueeze(1), dim=-1)
        if veh_mask is not None:
            d_veh = torch.where(veh_mask, d_veh, torch.full_like(d_veh, far_value))
        min_dist = torch.minimum(min_dist, d_veh.min(dim=1).values)

    return min_dist


def distance_to_risk_score(
    min_dist: torch.Tensor,
    safe_dist: float = 10.0,
    sharpness: float = 1.5,
) -> torch.Tensor:
    risk = 1.0 / (1.0 + torch.exp((min_dist - safe_dist) / sharpness))
    return risk.unsqueeze(1)


def _extract_weight_tensor(weight_output):
    if isinstance(weight_output, torch.Tensor):
        return weight_output
    if isinstance(weight_output, (tuple, list)) and len(weight_output) > 0:
        return _extract_weight_tensor(weight_output[0])
    if isinstance(weight_output, dict):
        for key in ["cost_weights", "weights", "weight", "pred_weights", "prob", "probs"]:
            if key in weight_output:
                return weight_output[key]
    raise TypeError(f"Unsupported weight output type: {type(weight_output)}")


def normalize_prob(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=eps, posinf=1.0, neginf=eps)
    x = x.clamp_min(eps)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)


def build_scene_feature_vector_from_batch(
    ego_state: torch.Tensor,
    ped_states: Optional[torch.Tensor],
    veh_states: Optional[torch.Tensor],
    lane_points: Optional[torch.Tensor],
    ped_mask: Optional[torch.Tensor] = None,
    veh_mask: Optional[torch.Tensor] = None,
    safe_dist: float = 10.0,
    sharpness: float = 1.5,
) -> torch.Tensor:
    device = ego_state.device
    dtype = ego_state.dtype
    B = ego_state.size(0)

    ego_xy = ego_state[:, :2]
    ego_speed = ego_state[:, 3:4] if ego_state.size(1) > 3 else torch.zeros(B, 1, device=device, dtype=dtype)

    far_dist = safe_dist * 5.0

    def masked_min_distance(agent_states: Optional[torch.Tensor], agent_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if agent_states is None or agent_states.numel() == 0:
            return torch.full((B, 1), far_dist, device=device, dtype=dtype)
        agent_xy = agent_states[:, :, :2]
        d = torch.norm(agent_xy - ego_xy.unsqueeze(1), dim=-1)
        if agent_mask is not None:
            d = torch.where(agent_mask.bool(), d, torch.full_like(d, far_dist))
        return d.min(dim=1, keepdim=True).values.clamp(0.0, far_dist)

    min_ped_dist = masked_min_distance(ped_states, ped_mask)
    min_veh_dist = masked_min_distance(veh_states, veh_mask)
    min_agent_dist = torch.minimum(min_ped_dist, min_veh_dist)
    risk_score = 1.0 / (1.0 + torch.exp((min_agent_dist - safe_dist) / sharpness))

    if ped_mask is not None:
        num_peds = ped_mask.float().sum(dim=1, keepdim=True)
    elif ped_states is not None:
        num_peds = torch.full((B, 1), ped_states.size(1), device=device, dtype=dtype)
    else:
        num_peds = torch.zeros((B, 1), device=device, dtype=dtype)

    if veh_mask is not None:
        num_vehs = veh_mask.float().sum(dim=1, keepdim=True)
    elif veh_states is not None:
        num_vehs = torch.full((B, 1), veh_states.size(1), device=device, dtype=dtype)
    else:
        num_vehs = torch.zeros((B, 1), device=device, dtype=dtype)

    if lane_points is not None and lane_points.numel() > 0 and lane_points.dim() == 4 and lane_points.size(1) >= 2:
        left = lane_points[:, 0]
        right = lane_points[:, 1]
        lane_width = torch.norm(left - right, dim=-1).mean(dim=1, keepdim=True)

        center = 0.5 * (left + right)
        if center.size(1) >= 3:
            delta = center[:, 1:] - center[:, :-1]
            theta = torch.atan2(delta[..., 1], delta[..., 0])
            if theta.size(1) >= 2:
                dtheta = torch.atan2(
                    torch.sin(theta[:, 1:] - theta[:, :-1]),
                    torch.cos(theta[:, 1:] - theta[:, :-1]),
                )
                curvature = torch.mean(torch.abs(dtheta), dim=1, keepdim=True)
            else:
                curvature = torch.zeros((B, 1), device=device, dtype=dtype)
        else:
            curvature = torch.zeros((B, 1), device=device, dtype=dtype)
    else:
        lane_width = torch.zeros((B, 1), device=device, dtype=dtype)
        curvature = torch.zeros((B, 1), device=device, dtype=dtype)

    if ego_state.size(1) >= 10:
        route_feat = ego_state[:, 4:10]
    else:
        route_feat = torch.zeros(B, 6, device=device, dtype=dtype)
    route_dist = torch.stack(
        [
            torch.norm(route_feat[:, 0:2], dim=-1),
            torch.norm(route_feat[:, 2:4], dim=-1),
            torch.norm(route_feat[:, 4:6], dim=-1),
        ],
        dim=-1,
    )

    scene_vec = torch.cat(
        [
            ego_speed.clamp(0.0, 40.0) / 20.0,
            min_ped_dist / far_dist,
            min_veh_dist / far_dist,
            min_agent_dist / far_dist,
            risk_score,
            num_peds.clamp(0.0, 32.0) / 32.0,
            num_vehs.clamp(0.0, 32.0) / 32.0,
            lane_width.clamp(0.0, 8.0) / 4.0,
            curvature.clamp(0.0, math.pi) / math.pi,
            route_dist.clamp(0.0, 100.0) / 50.0,
        ],
        dim=-1,
    )
    return scene_vec


def build_scene_adaptive_cost_prior(
    scene_vec: torch.Tensor,
    base_weights: torch.Tensor,
    collision_risk: Optional[torch.Tensor] = None,
    dense_gain: float = 1.0,
    turn_gain: float = 0.8,
    high_speed_gain: float = 0.7,
    high_speed_threshold: float = 12.0,
    high_speed_sharpness: float = 3.0,
) -> Dict[str, torch.Tensor]:
    B = scene_vec.shape[0]
    device = scene_vec.device
    dtype = scene_vec.dtype
    prior = _expand_cost_vector(base_weights, B, device, dtype, "base_weights").clone()

    ego_speed = scene_vec[:, 0:1].clamp(0.0, 4.0) * 20.0
    risk_score = scene_vec[:, 4:5].clamp(0.0, 1.0)
    density_score = (0.5 * scene_vec[:, 5:6] + 0.5 * scene_vec[:, 6:7]).clamp(0.0, 1.0)
    dense_score = (0.65 * risk_score + 0.35 * density_score).clamp(0.0, 1.0)
    if collision_risk is None:
        collision_score = torch.zeros_like(risk_score)
    else:
        collision_score = collision_risk.to(device=device, dtype=dtype).view(B, 1).clamp(0.0, 1.0)
    safe_score = (0.50 * risk_score + 0.30 * collision_score + 0.20 * density_score).clamp(0.0, 1.0)
    turn_score = scene_vec[:, 8:9].clamp(0.0, 1.0)
    high_speed_score = torch.sigmoid((ego_speed - float(high_speed_threshold)) / float(high_speed_sharpness))
    lane_width = scene_vec[:, 7:8].clamp(0.0, 2.0) * 4.0
    narrow_lane_score = ((3.8 - lane_width) / 1.5).clamp(0.0, 1.0)
    progress_score = ((1.0 - risk_score) * (1.0 - density_score)).clamp(0.0, 1.0)

    log_scale = torch.zeros_like(prior)
    idx = {name: i for i, name in enumerate(COST_NAMES)}

    def add(name: str, term: torch.Tensor, gain: float):
        log_scale[:, idx[name]] = log_scale[:, idx[name]] + float(gain) * term.squeeze(-1)

    add("safe", safe_score, +1.00 * dense_gain)
    add("jerk", dense_score, +0.35 * dense_gain)
    add("steering_change", dense_score, +0.35 * dense_gain)
    add("lane_xy", safe_score, +0.25 * dense_gain)
    add("route_target", dense_score, -0.45 * dense_gain)
    add("route_target", progress_score, +0.35 * dense_gain)

    add("acceleration", turn_score, +0.25 * turn_gain)
    add("jerk", turn_score, +0.50 * turn_gain)
    add("steering", turn_score, +0.25 * turn_gain)
    add("steering_change", turn_score, +0.65 * turn_gain)
    add("lane_theta", turn_score, +0.20 * turn_gain)
    add("route_target", turn_score, +0.20 * turn_gain)
    add("lane_xy", narrow_lane_score, +0.55 * turn_gain)

    add("safe", high_speed_score, +0.60 * high_speed_gain)
    add("lane_xy", high_speed_score, +0.35 * high_speed_gain)
    add("lane_theta", high_speed_score, +0.35 * high_speed_gain)
    add("steering", high_speed_score, +0.20 * high_speed_gain)
    add("steering_change", high_speed_score, +0.50 * high_speed_gain)
    add("route_target", high_speed_score, -0.20 * high_speed_gain)

    prior = prior * torch.exp(log_scale)
    prior = torch.nan_to_num(prior, nan=1.0, posinf=20.0, neginf=1e-3)
    return {
        "scene_prior_weights": prior,
        "dense_score": dense_score.squeeze(-1),
        "safe_score": safe_score.squeeze(-1),
        "collision_score": collision_score.squeeze(-1),
        "turn_score": turn_score.squeeze(-1),
        "high_speed_score": high_speed_score.squeeze(-1),
        "narrow_lane_score": narrow_lane_score.squeeze(-1),
        "progress_score": progress_score.squeeze(-1),
    }


def compute_logspace_residual_cost_weights(
    residual_raw: torch.Tensor,
    scene_prior_weights: torch.Tensor,
    min_weight=1e-3,
    max_weight=20.0,
    delta_max=0.7,
    renormalize_to_fallback_sum: bool = False,
) -> Dict[str, torch.Tensor]:
    B = scene_prior_weights.shape[0]
    device = scene_prior_weights.device
    dtype = scene_prior_weights.dtype
    residual_raw = residual_raw.to(device=device, dtype=dtype)
    if residual_raw.shape[-1] != NUM_COSTS:
        raise ValueError(f"residual_raw last dim must be {NUM_COSTS}, got {tuple(residual_raw.shape)}")

    delta_max_vec = _expand_cost_vector(delta_max, B, device, dtype, "delta_max")
    delta = delta_max_vec * torch.tanh(residual_raw)
    raw_weights = scene_prior_weights * torch.exp(delta)
    cost_weights = sanitize_planner_cost_weights(
        raw_weights,
        min_weight=min_weight,
        max_weight=max_weight,
        fallback_weights=scene_prior_weights,
        renormalize_to_fallback_sum=renormalize_to_fallback_sum,
    )
    return {
        "cost_weights": cost_weights,
        "pred_weights_prob": normalize_prob(cost_weights),
        "weight_delta": delta,
        "raw_weights": raw_weights,
    }


def compute_direct_weightnet_prob(
    weight_model: nn.Module,
    ego_state: torch.Tensor,
    ped_states: torch.Tensor,
    veh_states: torch.Tensor,
    lane_for_weight: torch.Tensor,
    ped_mask: Optional[torch.Tensor],
    veh_mask: Optional[torch.Tensor],
    safe_dist: float,
    scene_prior_weights: Optional[torch.Tensor] = None,
    scene_vec: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    prior_log_weights = None
    if scene_prior_weights is not None:
        prior_log_weights = scene_prior_weights.clamp_min(1e-8).log()

    try:
        weight_out = weight_model(
            ego_state=ego_state,
            ped_states=ped_states[..., :6],
            veh_states=veh_states[..., :6],
            lane_points=lane_for_weight,
            ped_mask=ped_mask,
            veh_mask=veh_mask,
            prior_log_weights=prior_log_weights,
            return_logits=True,
        )
        if isinstance(weight_out, (tuple, list)) and len(weight_out) >= 2:
            direct_prob, weight_delta_raw = weight_out[0], weight_out[1]
            refine_gate = weight_out[2] if len(weight_out) >= 3 else None
        elif isinstance(weight_out, dict) and "logits" in weight_out:
            weight_delta_raw = weight_out["logits"]
            direct_prob = weight_out.get("weights", torch.softmax(weight_delta_raw, dim=-1))
            refine_gate = weight_out.get("refine_gate")
        else:
            weight_delta_raw = _extract_weight_tensor(weight_out)
            direct_prob = torch.softmax(weight_delta_raw, dim=-1)
            refine_gate = None
    except TypeError:
        weight_out = weight_model(
            ego_state=ego_state,
            ped_states=ped_states[..., :6],
            veh_states=veh_states[..., :6],
            lane_points=lane_for_weight,
            ped_mask=ped_mask,
            veh_mask=veh_mask,
            prior_log_weights=prior_log_weights,
            return_logits=False,
        )
        direct_prob = normalize_prob(_extract_weight_tensor(weight_out))
        weight_delta_raw = torch.log(direct_prob.clamp_min(1e-8))
        refine_gate = None

    if refine_gate is None:
        refine_gate = torch.ones(
            ego_state.shape[0], 1, device=ego_state.device, dtype=ego_state.dtype
        )
    refine_gate = torch.nan_to_num(refine_gate, nan=0.0).clamp(0.0, 1.0)

    if scene_vec is None:
        scene_vec = build_scene_feature_vector_from_batch(
            ego_state=ego_state,
            ped_states=ped_states,
            veh_states=veh_states,
            lane_points=lane_for_weight,
            ped_mask=ped_mask,
            veh_mask=veh_mask,
            safe_dist=safe_dist,
        )
    return {
        "weight_delta_raw": weight_delta_raw,
        "pred_weights_prob_direct": normalize_prob(direct_prob),
        "refine_gate": refine_gate,
        "scene_vec": scene_vec,
    }


def compute_weight_loss_differentiable(
    ego_dipp_traj: torch.Tensor,
    ego_future_gt_new: torch.Tensor,
    ego_future_gt_valid_mask: torch.Tensor,
):
    B, T, _ = ego_dipp_traj.shape
    idx_1 = min(9, T - 1)
    idx_2 = min(19, T - 1)
    idx_3 = min(29, T - 1)

    pred_points = torch.stack(
        [
            ego_dipp_traj[:, idx_1, :2],
            ego_dipp_traj[:, idx_2, :2],
            ego_dipp_traj[:, idx_3, :2],
        ],
        dim=1,
    )
    gt_points = ego_future_gt_new.view(B, 3, 2)
    per_sample_loss = ((pred_points - gt_points) ** 2).mean(dim=(1, 2))

    valid = ego_future_gt_valid_mask
    if valid.any():
        return per_sample_loss[valid].mean()
    return torch.zeros((), device=ego_dipp_traj.device, dtype=ego_dipp_traj.dtype)


def compute_weight_regularizers(pred_weights_prob: torch.Tensor, prior_weights: torch.Tensor):
    eps = 1e-8
    p = pred_weights_prob.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    q = prior_weights.to(device=p.device, dtype=p.dtype)
    if q.dim() == 1:
        q = q.unsqueeze(0).expand_as(p)
    elif q.dim() == 2 and q.shape[0] == 1:
        q = q.expand_as(p)
    elif q.shape != p.shape:
        raise ValueError(f"prior_weights shape must be [{NUM_COSTS}] or {tuple(p.shape)}, got {tuple(q.shape)}")
    q = q.clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)

    entropy = -(p * torch.log(p)).sum(dim=-1).mean()
    diversity = p.std(dim=0, unbiased=False).mean()
    kl_to_prior = (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean()
    return entropy, diversity, kl_to_prior


def compute_weight_extreme_penalty(
    pred_weights_prob: torch.Tensor,
    max_allowed_prob: float = 0.50,
    min_allowed_prob: float = 0.0,
) -> torch.Tensor:
    eps = 1e-8
    p = pred_weights_prob.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    max_prob = p.max(dim=-1).values
    high_penalty = F.relu(max_prob - float(max_allowed_prob)).pow(2).mean()

    if float(min_allowed_prob) > 0.0:
        low_penalty = F.relu(float(min_allowed_prob) - p).pow(2).mean()
    else:
        low_penalty = p.new_zeros(())

    return high_penalty + low_penalty


def compute_weight_separation_loss(
    pred_weights_prob: torch.Tensor,
    risk_score: torch.Tensor,
    base_margin: float = 0.03,
    scalar_scale: float = 0.20,
    min_risk_gap: float = 0.10,
) -> torch.Tensor:
    if pred_weights_prob.shape[0] < 2:
        return pred_weights_prob.new_zeros(())

    risk = risk_score.detach().view(-1, 1).float()
    weights = pred_weights_prob.float()

    dist_w = torch.cdist(weights, weights, p=2)
    dist_r = torch.cdist(risk, risk, p=1)

    target_margin = base_margin + scalar_scale * dist_r
    pair_mask = torch.triu(torch.ones_like(dist_w, dtype=torch.bool), diagonal=1)
    pair_mask = pair_mask & (dist_r > min_risk_gap)
    if pair_mask.sum() == 0:
        return pred_weights_prob.new_zeros(())

    return F.relu(target_margin - dist_w)[pair_mask].mean().to(pred_weights_prob.dtype)


# ============================================================
# Semantic and feedback supervision for adaptive weights
# ============================================================
WEAK_FREE_WEIGHTS = (2.0, 3.5, 1.2, 3.5, 0.6, 0.6, 5.5, 0.8)
WEAK_RISKY_WEIGHTS = (0.5, 3.0, 0.5, 3.0, 3.5, 3.5, 2.0, 6.0)


def build_semantic_rule_target_weights(
    scene_vec: torch.Tensor,
    risk_score: torch.Tensor,
    cfg_runtime: Dict,
) -> torch.Tensor:
    device = scene_vec.device
    dtype = scene_vec.dtype
    B = scene_vec.shape[0]

    w_free = torch.tensor(
        cfg_runtime.get("weak_free_weights", WEAK_FREE_WEIGHTS),
        device=device,
        dtype=dtype,
    ).view(1, NUM_COSTS)
    w_risky = torch.tensor(
        cfg_runtime.get("weak_risky_weights", WEAK_RISKY_WEIGHTS),
        device=device,
        dtype=dtype,
    ).view(1, NUM_COSTS)

    risk = risk_score.view(B, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
    num_peds = (scene_vec[:, 5:6] * 32.0).clamp_min(0.0)
    num_vehs = (scene_vec[:, 6:7] * 32.0).clamp_min(0.0)
    density = ((num_peds + num_vehs) / cfg_runtime.get("density_norm_agents", 8.0)).clamp(0.0, 1.0)

    free_score = (1.0 - (0.75 * risk + 0.25 * density)).clamp(0.0, 1.0)
    interact_score = 1.0 - free_score
    target = free_score * w_free + interact_score * w_risky

    ego_speed_norm = scene_vec[:, 0:1].clamp(0.0, 2.0)
    lane_width = scene_vec[:, 7:8] * 4.0
    curvature_norm = scene_vec[:, 8:9].clamp(0.0, 1.0)
    narrow_lane = ((cfg_runtime.get("nominal_lane_width", 3.8) - lane_width) / 1.5).clamp(0.0, 1.0)

    target[:, 1:2] = target[:, 1:2] + 0.40 * ego_speed_norm
    target[:, 3:4] = target[:, 3:4] + 0.40 * ego_speed_norm
    target[:, 4:5] = target[:, 4:5] + 0.90 * curvature_norm + 0.70 * narrow_lane + 0.35 * interact_score
    target[:, 5:6] = target[:, 5:6] + 1.00 * curvature_norm
    target[:, 6:7] = target[:, 6:7] + 0.75 * free_score - 0.25 * interact_score
    target[:, 7:8] = target[:, 7:8] + 1.35 * interact_score + 0.45 * narrow_lane
    return normalize_prob(target)


def compute_rule_kl_loss(pred_weights_prob: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    pred_log = normalize_prob(pred_weights_prob).clamp_min(1e-8).log()
    return F.kl_div(pred_log, normalize_prob(target_prob).detach(), reduction="batchmean")


def build_hipad_collision_target_weights(
    scene_prior_weights: torch.Tensor,
    hipad_collision_gate: torch.Tensor,
    cfg_runtime: Dict,
) -> torch.Tensor:
    """Raise safety weight only when the full HiPAD plan collides with GT actors."""
    target = scene_prior_weights.detach().clone()
    gate = hipad_collision_gate.detach().to(device=target.device, dtype=target.dtype).view(-1)
    idx = {name: i for i, name in enumerate(COST_NAMES)}
    safe_gain = float(cfg_runtime.get("weight_hipad_safe_log_gain", 0.8))
    route_decay = float(cfg_runtime.get("weight_hipad_route_log_decay", 0.25))
    target[:, idx["safe"]] = target[:, idx["safe"]] * torch.exp(safe_gain * gate)
    target[:, idx["route_target"]] = target[:, idx["route_target"]] * torch.exp(-route_decay * gate)
    return normalize_prob(target)


def build_pnn_collision_target_weights(
    scene_prior_weights: torch.Tensor,
    obj_collision_gate: torch.Tensor,
    lane_collision_gate: torch.Tensor,
    cfg_runtime: Dict,
) -> torch.Tensor:
    """Build a task-specific target distribution from current PNN failures.

    Object and lane failures must not share one ambiguous risk scalar: ACR
    should raise the obstacle-safety cost, while CCR should raise lane position
    and heading costs. Both may relax comfort/progress costs so the optimizer
    has enough authority to brake or steer away from the failure.
    """
    target = scene_prior_weights.detach().clone()
    dtype, device = target.dtype, target.device
    obj = obj_collision_gate.detach().to(device=device, dtype=dtype).view(-1).clamp(0.0, 1.0)
    lane = lane_collision_gate.detach().to(device=device, dtype=dtype).view(-1).clamp(0.0, 1.0)
    any_collision = torch.maximum(obj, lane)
    idx = {name: i for i, name in enumerate(COST_NAMES)}

    target[:, idx["safe"]] *= torch.exp(
        float(cfg_runtime.get("weight_pnn_obj_safe_log_gain", 1.0)) * obj
    )
    target[:, idx["lane_xy"]] *= torch.exp(
        float(cfg_runtime.get("weight_pnn_lane_xy_log_gain", 0.9)) * lane
    )
    target[:, idx["lane_theta"]] *= torch.exp(
        float(cfg_runtime.get("weight_pnn_lane_theta_log_gain", 0.7)) * lane
    )

    comfort_decay = float(cfg_runtime.get("weight_pnn_comfort_log_decay", 0.45))
    for name in ("acceleration", "jerk", "steering", "steering_change"):
        target[:, idx[name]] *= torch.exp(-comfort_decay * any_collision)
    target[:, idx["route_target"]] *= torch.exp(
        -float(cfg_runtime.get("weight_pnn_route_log_decay", 0.55)) * any_collision
    )
    return normalize_prob(target)


def compute_sample_weighted_kl_loss(
    pred_weights_prob: torch.Tensor,
    target_prob: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    pred_log = normalize_prob(pred_weights_prob).clamp_min(1e-8).log()
    target = normalize_prob(target_prob).detach()
    per_sample = (target * (target.clamp_min(1e-8).log() - pred_log)).sum(dim=-1)
    weight = sample_weight.detach().to(device=per_sample.device, dtype=per_sample.dtype).view(-1)
    return (weight * per_sample).sum() / weight.sum().clamp_min(1e-8)


def compute_risk_conditioned_ranking_loss(
    pred_weights_prob: torch.Tensor,
    risk_score: torch.Tensor,
    cfg_runtime: Dict,
) -> torch.Tensor:
    p = normalize_prob(pred_weights_prob)
    risk = risk_score.detach().view(-1)
    high_mask = risk >= cfg_runtime.get("rank_high_risk_th", 0.55)
    low_mask = risk <= cfg_runtime.get("rank_low_risk_th", 0.25)
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return p.new_zeros(())

    high = p[high_mask]
    low = p[low_mask]
    high_r = risk[high_mask]
    low_r = risk[low_mask]
    n = min(high.shape[0], low.shape[0])
    high = high[torch.argsort(high_r, descending=True)[:n]]
    low = low[torch.argsort(low_r, descending=False)[:n]]

    safe_margin = cfg_runtime.get("rank_margin_safe", 0.010)
    route_margin = cfg_runtime.get("rank_margin_route", 0.010)
    comfort_margin = cfg_runtime.get("rank_margin_comfort", 0.008)

    comfort_high = high[:, [0, 1, 2, 3]].mean(dim=1)
    comfort_low = low[:, [0, 1, 2, 3]].mean(dim=1)

    loss_safe = F.relu(safe_margin - (high[:, 7] - low[:, 7])).mean()
    loss_route = F.relu(route_margin - (low[:, 6] - high[:, 6])).mean()
    loss_comfort = F.relu(comfort_margin - (comfort_low - comfort_high)).mean()
    return loss_safe + 0.5 * loss_route + 0.5 * loss_comfort


def compute_entropy_band_loss(
    pred_weights_prob: torch.Tensor,
    entropy_low: float = 1.35,
    entropy_high: float = 2.00,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = normalize_prob(pred_weights_prob)
    entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)
    loss = (F.relu(float(entropy_low) - entropy) + F.relu(entropy - float(entropy_high))).mean()
    return loss, entropy.mean()


def compute_pairwise_diversity_floor_loss(
    pred_weights_prob: torch.Tensor,
    target_pairwise_l2: float = 0.010,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = normalize_prob(pred_weights_prob)
    if p.shape[0] < 2:
        zero = p.new_zeros(())
        return zero, zero
    pairwise = torch.pdist(p, p=2)
    if pairwise.numel() == 0:
        zero = p.new_zeros(())
        return zero, zero
    mean_pairwise = pairwise.mean()
    loss = F.relu(float(target_pairwise_l2) - mean_pairwise).pow(2)
    return loss, mean_pairwise


def compute_feedback_target_weights(
    weighted_ego_comps: Dict[str, torch.Tensor],
    g_lane: torch.Tensor,
    g_safety: torch.Tensor,
    risk_score: torch.Tensor,
    scene_prior_weights: torch.Tensor,
    cfg_runtime: Dict,
) -> torch.Tensor:
    target = scene_prior_weights.detach().clone()
    device = target.device
    dtype = target.dtype
    idx = {name: i for i, name in enumerate(COST_NAMES)}

    comp = torch.stack(
        [weighted_ego_comps[name].detach().to(device=device, dtype=dtype) for name in COST_NAMES],
        dim=-1,
    )
    comp = torch.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    comp_log = torch.log1p(comp)
    comp_center = comp_log.median(dim=0, keepdim=True).values
    comp_scale = (comp_log - comp_center).clamp_min(0.0)

    gain = cfg_runtime.get("feedback_component_gain", 0.30)
    target = target * torch.exp(gain * comp_scale.clamp(max=2.0))

    lane_score = torch.log1p(g_lane.detach().to(device=device, dtype=dtype).clamp_min(0.0)).view(-1, 1).clamp(max=2.0)
    safety_score = torch.log1p(g_safety.detach().to(device=device, dtype=dtype).clamp_min(0.0)).view(-1, 1).clamp(max=2.0)
    risk = risk_score.detach().to(device=device, dtype=dtype).view(-1, 1).clamp(0.0, 1.0)

    target[:, idx["lane_xy"]] = target[:, idx["lane_xy"]] * torch.exp(0.70 * lane_score.squeeze(-1))
    target[:, idx["lane_theta"]] = target[:, idx["lane_theta"]] * torch.exp(0.55 * lane_score.squeeze(-1))
    target[:, idx["safe"]] = target[:, idx["safe"]] * torch.exp(0.85 * safety_score.squeeze(-1) + 0.60 * risk.squeeze(-1))
    target[:, idx["route_target"]] = target[:, idx["route_target"]] * torch.exp(
        -0.35 * risk.squeeze(-1)
        - 0.20 * safety_score.squeeze(-1)
        + 0.15 * (1.0 - lane_score.squeeze(-1).clamp(0.0, 1.0))
    )

    return normalize_prob(target)


# ============================================================
# Lower branch losses
# ============================================================
def nearest_ref_points_and_theta(ego_traj: torch.Tensor, ref_line: torch.Tensor):
    dist = torch.cdist(ego_traj[:, :, :2], ref_line[:, :, :2])
    idx = torch.argmin(dist, dim=-1)
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, 3)
    ref_pts = torch.gather(ref_line, 1, gather_idx)
    return ref_pts


def compute_safe_component_from_rollout(
    ego_traj: torch.Tensor,
    ped_traj: torch.Tensor,
    veh_traj: torch.Tensor,
    ped_mask: Optional[torch.Tensor],
    veh_mask: Optional[torch.Tensor],
    safe_margin: float = SAFE_MARGIN,
):
    B = ego_traj.shape[0]
    ego_xy = ego_traj[:, :, :2]
    agent_parts = []
    mask_parts = []

    if ped_traj is not None and ped_traj.numel() > 0:
        agent_parts.append(ped_traj[:, :, :, :2])
        if ped_mask is None:
            ped_mask = torch.ones((B, ped_traj.shape[1]), device=ego_traj.device, dtype=torch.bool)
        mask_parts.append(ped_mask.bool())

    if veh_traj is not None and veh_traj.numel() > 0:
        agent_parts.append(veh_traj[:, :, :, :2])
        if veh_mask is None:
            veh_mask = torch.ones((B, veh_traj.shape[1]), device=ego_traj.device, dtype=torch.bool)
        mask_parts.append(veh_mask.bool())

    if len(agent_parts) == 0:
        return torch.zeros(B, device=ego_traj.device, dtype=ego_traj.dtype)

    agents_xy = torch.cat(agent_parts, dim=1)
    agents_mask = torch.cat(mask_parts, dim=1)
    residual = compute_collision_ttc_residual(
        ego_xy=ego_xy,
        agents_xy=agents_xy,
        agents_mask=agents_mask,
        collision_dist=safe_margin,
    )
    denom = agents_mask.to(residual.dtype).sum(dim=1).clamp_min(1.0) * residual.shape[-1]
    return residual.sum(dim=(1, 2)) / denom


def compute_weighted_ego_cost_components(
    u_ego: torch.Tensor,
    ego_traj: torch.Tensor,
    ego_state: torch.Tensor,
    lane_points_control: torch.Tensor,
    ped_traj: torch.Tensor,
    veh_traj: torch.Tensor,
    ped_mask: Optional[torch.Tensor],
    veh_mask: Optional[torch.Tensor],
    gt_reference_line: Optional[torch.Tensor] = None,
    gt_reference_line_valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    B, T, _ = u_ego.shape
    predicted_ref_line = build_ref_line_from_two_boundaries(lane_points_control)
    ref_line = predicted_ref_line
    if gt_reference_line is not None:
        # GT is stored at t={0,1,2,3}s. Densify it before nearest-point
        # matching; using only four isolated points creates a staircase loss.
        gt_reference_dense = F.interpolate(
            gt_reference_line.transpose(1, 2),
            size=predicted_ref_line.size(1),
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        gt_ref = build_ref_line_from_xy(gt_reference_dense)
        if gt_reference_line_valid_mask is None:
            valid_ref = torch.ones(B, dtype=torch.bool, device=ego_traj.device)
        else:
            valid_ref = gt_reference_line_valid_mask.reshape(B).bool()
        # Keep predicted map geometry as the fallback for incomplete GT.
        ref_line = torch.where(valid_ref[:, None, None], gt_ref, predicted_ref_line)
    ref_pts = nearest_ref_points_and_theta(ego_traj, ref_line)

    comp_acc = (u_ego[:, :, 0] ** 2).mean(dim=1)

    if T > 1:
        jerk = torch.diff(u_ego[:, :, 0], dim=1) / DT
        comp_jerk = smooth_abs_excess(jerk, JERK_COMFORT_LIMIT).pow(2).mean(dim=1)
    else:
        comp_jerk = torch.zeros(B, device=u_ego.device, dtype=u_ego.dtype)

    comp_steer = (u_ego[:, :, 1] ** 2).mean(dim=1)

    if T > 1:
        steer_change = torch.diff(u_ego[:, :, 1], dim=1) / DT
        comp_steer_change = smooth_abs_excess(
            steer_change,
            STEER_RATE_COMFORT_LIMIT,
        ).pow(2).mean(dim=1)
    else:
        comp_steer_change = torch.zeros(B, device=u_ego.device, dtype=u_ego.dtype)

    lane_center_err = torch.norm(ego_traj[:, 1::2, 0:2] - ref_pts[:, 1::2, 0:2], dim=-1)
    lane_dac_proxy = smooth_abs_excess(
        lane_center_err,
        max(LANE_HALF_WIDTH_FALLBACK - LANE_DAC_MARGIN, 0.1),
        sharpness=8.0,
    )
    comp_lane_xy = (lane_center_err.pow(2) + 2.0 * lane_dac_proxy.pow(2)).mean(dim=1)

    theta_err = torch.atan2(
        torch.sin(ego_traj[:, 1::2, 2] - ref_pts[:, 1::2, 2]),
        torch.cos(ego_traj[:, 1::2, 2] - ref_pts[:, 1::2, 2]),
    )
    comp_lane_theta = (theta_err ** 2).mean(dim=1)

    route_err = route_progress_residuals(ego_traj, ego_state, T)
    comp_route = route_err.pow(2).mean(dim=1)

    comp_safe = compute_safe_component_from_rollout(
        ego_traj=ego_traj,
        ped_traj=ped_traj,
        veh_traj=veh_traj,
        ped_mask=ped_mask,
        veh_mask=veh_mask,
        safe_margin=SAFE_MARGIN,
    )

    return {
        "acceleration": comp_acc,
        "jerk": comp_jerk,
        "steering": comp_steer,
        "steering_change": comp_steer_change,
        "lane_xy": comp_lane_xy,
        "lane_theta": comp_lane_theta,
        "route_target": comp_route,
        "safe": comp_safe,
    }


@torch.no_grad()
def compute_training_planning_proxy_metrics(
    ego_traj: torch.Tensor,
    ego_state: torch.Tensor,
    ego_future_gt_new: torch.Tensor,
    ego_future_gt_valid_mask: torch.Tensor,
    ped_states: torch.Tensor,
    veh_states: torch.Tensor,
    ped_mask: Optional[torch.Tensor],
    veh_mask: Optional[torch.Tensor],
    control: torch.Tensor,
    g_lane: torch.Tensor,
    g_safety: torch.Tensor,
    safety_loss_module: SafetyConstraintLoss,
    comfort_max_lon_accel: float = 2.40,
    comfort_min_lon_accel: float = -4.05,
    comfort_max_lat_accel: float = 4.89,
    comfort_jerk_threshold: float = 4.13,
    comfort_yaw_rate_threshold: float = 0.95,
    comfort_yaw_accel_threshold: float = 1.93,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Cheap train-batch planning indicators.

    These are not full Bench2Drive metrics. They run directly on the training
    batch and are meant for trend monitoring/checkpoint selection:
      - l2_gt_* compares rollout to true future GT when valid.
      - l2_route_* compares rollout to the 1/2/3s route targets in ego_state.
      - obj_col_proxy_* uses rectangular overlap at the six 2Hz plan points.
      - lane/safety violations reuse the training constraint violations.
      - comfort_score_3s follows the same spirit as the open-loop comfort check.
    """
    metrics: Dict[str, torch.Tensor] = {}
    device = ego_traj.device
    dtype = ego_traj.dtype
    B, T, _ = ego_traj.shape
    idxs = [min(9, T - 1), min(19, T - 1), min(29, T - 1)]

    pred_points = torch.stack([ego_traj[:, idx, :2] for idx in idxs], dim=1)
    route_points = torch.stack(
        [ego_state[:, 4:6], ego_state[:, 6:8], ego_state[:, 8:10]],
        dim=1,
    )
    valid = ego_future_gt_valid_mask.bool().view(-1)
    valid_count = valid.to(dtype).sum()

    route_l2 = torch.norm(pred_points - route_points, dim=-1)
    for i in range(3):
        metrics[f"l2_route_{i + 1}s"] = route_l2[:, i].mean()

    if valid_count > 0:
        gt_l2 = torch.norm(pred_points[valid] - ego_future_gt_new[valid, :3, :2], dim=-1)
        for i in range(3):
            metrics[f"l2_gt_{i + 1}s"] = gt_l2[:, i].mean()
    else:
        nan = torch.tensor(float("nan"), device=device, dtype=dtype)
        for i in range(3):
            metrics[f"l2_gt_{i + 1}s"] = nan

    sample_idxs = [min(i, T - 1) for i in (4, 9, 14, 19, 24, 29)]

    def group_box_collision(
        states: torch.Tensor,
        mask: Optional[torch.Tensor],
        shape: Tuple[float, float],
    ) -> torch.Tensor:
        num_agents = states.shape[1]
        if num_agents == 0:
            return torch.zeros((B, len(sample_idxs)), device=device, dtype=torch.bool)
        if mask is None:
            mask = torch.ones((B, num_agents), device=device, dtype=torch.bool)
        else:
            mask = mask.bool()

        start = states[:, :, 0:2]
        goal = states[:, :, 4:6]
        direction = goal - start
        moving = torch.norm(direction, dim=-1) > 1e-3
        goal_heading = torch.atan2(direction[..., 1], direction[..., 0])
        heading = torch.where(moving, goal_heading, states[:, :, 2])
        collisions = []
        for idx in sample_idxs:
            alpha = float(idx + 1) / float(T)
            xy = start + alpha * direction
            group_state = torch.cat(
                [xy, heading.unsqueeze(-1), torch.zeros_like(heading.unsqueeze(-1))],
                dim=-1,
            )
            ego_expand = ego_traj[:, idx, :4].unsqueeze(1).expand(-1, num_agents, -1)
            _, penetration = safety_loss_module.rect_dist(
                ego_expand.reshape(B * num_agents, 4),
                safety_loss_module.ego_shape,
                group_state.reshape(B * num_agents, 4),
                shape,
            )
            frame_collision = (
                (penetration.reshape(B, num_agents) > 0) & mask
            ).any(dim=1)
            collisions.append(frame_collision)
        return torch.stack(collisions, dim=1)

    ped_collision = group_box_collision(ped_states, ped_mask, safety_loss_module.ped_shape)
    veh_collision = group_box_collision(veh_states, veh_mask, safety_loss_module.veh_shape)
    box_collision = ped_collision | veh_collision
    for i, sample_count in enumerate((2, 4, 6)):
        metrics[f"obj_col_proxy_{i + 1}s"] = (
            box_collision[:, :sample_count].any(dim=1).to(dtype).mean()
        )

    metrics["lane_violation_mean"] = g_lane.detach().to(dtype).mean()
    metrics["lane_violation_rate"] = (g_lane.detach() > 0).to(dtype).mean()
    metrics["safety_violation_mean"] = g_safety.detach().to(dtype).mean()
    metrics["safety_violation_rate"] = (g_safety.detach() > 0).to(dtype).mean()

    v = ego_traj[:, :, 3].clamp_min(0.0)
    theta = ego_traj[:, :, 2]
    if T > 1:
        lon_acc = torch.diff(v, dim=1) / DT
        yaw_rate = torch.atan2(
            torch.sin(torch.diff(theta, dim=1)),
            torch.cos(torch.diff(theta, dim=1)),
        ) / DT
    else:
        lon_acc = torch.zeros((B, 1), device=device, dtype=dtype)
        yaw_rate = torch.zeros((B, 1), device=device, dtype=dtype)
    if T > 2:
        lon_jerk = torch.diff(lon_acc, dim=1) / DT
        yaw_accel = torch.diff(yaw_rate, dim=1) / DT
    else:
        lon_jerk = torch.zeros((B, 1), device=device, dtype=dtype)
        yaw_accel = torch.zeros((B, 1), device=device, dtype=dtype)

    lat_acc = v[:, 1:] * yaw_rate if T > 1 else torch.zeros((B, 1), device=device, dtype=dtype)
    max_abs_lon_accel = lon_acc.abs().amax(dim=1)
    max_abs_lat_accel = lat_acc.abs().amax(dim=1)
    max_abs_jerk = lon_jerk.abs().amax(dim=1)
    max_abs_yaw_rate = yaw_rate.abs().amax(dim=1)
    max_abs_yaw_accel = yaw_accel.abs().amax(dim=1)
    lon_acc_ok = (
        (lon_acc >= float(comfort_min_lon_accel))
        & (lon_acc <= float(comfort_max_lon_accel))
    ).all(dim=1)
    comfort_ok = (
        lon_acc_ok
        & (max_abs_lat_accel <= float(comfort_max_lat_accel))
        & (max_abs_jerk <= float(comfort_jerk_threshold))
        & (max_abs_yaw_rate <= float(comfort_yaw_rate_threshold))
        & (max_abs_yaw_accel <= float(comfort_yaw_accel_threshold))
    ).to(dtype)
    metrics["comfort_score_3s"] = comfort_ok.mean()
    metrics["max_abs_lon_accel_3s"] = max_abs_lon_accel.mean()
    metrics["max_abs_lat_accel_3s"] = max_abs_lat_accel.mean()
    metrics["max_abs_jerk_3s"] = max_abs_jerk.mean()
    metrics["max_abs_yaw_rate_3s"] = max_abs_yaw_rate.mean()
    metrics["max_abs_yaw_accel_3s"] = max_abs_yaw_accel.mean()
    return metrics, valid_count


def compute_residual_control_loss(
    u_ego,
    u_peds,
    u_vehs,
    ego_traj,
    ped_traj,
    veh_traj,
    ped_states,
    veh_states,
    lane_points,
    ped_mask,
    veh_mask,
    device,
    lane_loss_module,
    safety_loss_module,
    soft_lambda_module,
):
    B, Np, T = u_peds.shape[:3]
    Nv = u_vehs.shape[1]

    if ped_mask is None:
        ped_mask = torch.ones((B, Np), dtype=torch.bool, device=device)
    if veh_mask is None:
        veh_mask = torch.ones((B, Nv), dtype=torch.bool, device=device)

    soft_lambdas = soft_lambda_module()

    ctrl_p = (u_peds[..., 0] ** 2 + u_peds[..., 1] ** 2)
    ctrl_v = (u_vehs[..., 0] ** 2 + u_vehs[..., 1] ** 2)
    loss_control_p = masked_mean_per_sample(ctrl_p, ped_mask) * soft_lambdas["ctrl_p"]
    loss_control_v = masked_mean_per_sample(ctrl_v, veh_mask) * soft_lambdas["ctrl_v"]

    ped_goal_err = (ped_traj[:, :, -1, 0] - ped_states[:, :, 4]) ** 2 + (
        ped_traj[:, :, -1, 1] - ped_states[:, :, 5]
    ) ** 2
    veh_goal_err = (veh_traj[:, :, -1, 0] - veh_states[:, :, 4]) ** 2 + (
        veh_traj[:, :, -1, 1] - veh_states[:, :, 5]
    ) ** 2
    loss_track_p = masked_mean_per_sample(ped_goal_err, ped_mask) * 0.5
    loss_track_v = masked_mean_per_sample(veh_goal_err, veh_mask) * 2.0

    g_safety = safety_loss_module.compute_constraint_violation(ego_traj, ped_traj, veh_traj, ped_mask, veh_mask)
    g_lane = lane_loss_module.compute_constraint_violation(ego_traj[:, :, :2], lane_points)
    loss_safety = safety_loss_module.lambda_val.detach() * F.softplus(10.0 * g_safety) / 10.0
    loss_lane_hard = lane_loss_module.lambda_val.detach() * F.softplus(10.0 * g_lane) / 10.0

    ped_vel = ped_traj[..., 3]
    veh_vel = veh_traj[..., 3]
    loss_vel_p = masked_mean_per_sample(torch.clamp(ped_vel - 2.0, min=0.0), ped_mask) * soft_lambdas["vel_p"]
    loss_vel_v = masked_mean_per_sample(torch.clamp(veh_vel - 15.0, min=0.0), veh_mask) * soft_lambdas["vel_v"]

    if u_ego.shape[1] > 1:
        delta_a = u_ego[:, 1:, 0] - u_ego[:, :-1, 0]
        delta_delta = u_ego[:, 1:, 1] - u_ego[:, :-1, 1]
        exceed = torch.clamp(delta_delta.abs() - 0.04, min=0.0)
        loss_control_rate_ego = (
            delta_a.pow(2).mean(dim=1) + 5.0 * F.softplus(10.0 * exceed).mean(dim=1)
        ) * soft_lambdas["ctrl_ego"]
    else:
        loss_control_rate_ego = torch.zeros(B, device=device, dtype=u_ego.dtype)

    if u_vehs.shape[2] > 1:
        delta_a_v = u_vehs[:, :, 1:, 0] - u_vehs[:, :, :-1, 0]
        delta_d_v = u_vehs[:, :, 1:, 1] - u_vehs[:, :, :-1, 1]
        exceed_v = torch.clamp(delta_d_v.abs() - 0.04, min=0.0)
        veh_rate = delta_a_v.pow(2).mean(dim=-1) + 5.0 * F.softplus(10.0 * exceed_v).mean(dim=-1)
        loss_control_rate_veh = masked_mean_per_sample(veh_rate, veh_mask) * soft_lambdas["ctrl_v"]
    else:
        loss_control_rate_veh = torch.zeros(B, device=device, dtype=u_ego.dtype)

    l1_ego = u_ego.abs().mean(dim=(1, 2))
    l1_p = masked_mean_per_sample(u_peds.abs().sum(dim=-1), ped_mask)
    l1_v = masked_mean_per_sample(u_vehs.abs().sum(dim=-1), veh_mask)
    l1_penalty = soft_lambdas["l1"] * (l1_ego + l1_p + l1_v)

    residual = (
        loss_control_p + loss_control_v +
        loss_track_p + loss_track_v +
        loss_safety + loss_lane_hard +
        loss_vel_p + loss_vel_v +
        loss_control_rate_ego + loss_control_rate_veh +
        l1_penalty
    )
    aux = {
        "g_lane": g_lane,
        "g_safety": g_safety,
    }
    return residual, aux


def compute_ego_track_loss_with_lane_check_per_sample(
    ego_traj: torch.Tensor,
    ego_state: torch.Tensor,
    lane_lines: torch.Tensor,
    margin: float = 0.3,
    time_weights: Tuple[float, float, float, float, float, float] = (1.8, 2.5, 1.8, 1.2, 0.8, 0.5),
) -> torch.Tensor:
    B, T, _ = ego_traj.shape
    device = ego_traj.device
    dtype = ego_traj.dtype
    # PNN receives only 1s/2s/3s route targets. For open-loop metrics, however,
    # Bench2Drive reports cumulative ADE over 2Hz points. Use linearly
    # interpolated 0.5/1.5/2.5s pseudo-targets so the ControlNet is discouraged
    # from over-shooting in the first 1-2 seconds and merely returning to the
    # 3s endpoint later.
    ref_indices = [min(i, T - 1) for i in (4, 9, 14, 19, 24, 29)]
    start_xy = ego_state[:, 0:2]
    p1 = ego_state[:, 4:6]
    p2 = ego_state[:, 6:8]
    p3 = ego_state[:, 8:10]
    gt_points = torch.stack(
        [
            start_xy + 0.5 * (p1 - start_xy),
            p1,
            0.5 * (p1 + p2),
            p2,
            0.5 * (p2 + p3),
            p3,
        ],
        dim=1,
    )
    weights = torch.tensor(time_weights, device=device, dtype=dtype).clamp_min(0.0)

    loss_sum = torch.zeros(B, device=device, dtype=ego_traj.dtype)
    valid_weight = torch.zeros(B, device=device, dtype=ego_traj.dtype)

    for i, idx in enumerate(ref_indices):
        pred_xy = ego_traj[:, idx, :2]
        sq_err = ((pred_xy - gt_points[:, i]) ** 2).sum(dim=-1)
        w = weights[i]
        # Keep route tracking active near lane boundaries. The old boolean mask
        # removed this gradient exactly when the rollout started drifting; lane
        # and safety terms should oppose unsafe route targets continuously.
        loss_sum = loss_sum + w * sq_err
        valid_weight = valid_weight + w

    return loss_sum / valid_weight.clamp_min(1e-6)


def compute_lane_clearance_loss_per_sample(
    ego_xy: torch.Tensor,
    lane_lines: torch.Tensor,
    margin: float = 0.8,
) -> torch.Tensor:
    """Continuous lane-boundary clearance penalty.

    The existing hard lane constraint mainly reacts after a violation. This
    term gives a smooth gradient when the rollout approaches either boundary,
    which is important for the current B2D failure mode: PNN often overshoots
    during the first 1-2 seconds and drifts close to a solid boundary.
    """
    if lane_lines is None or lane_lines.dim() != 4 or lane_lines.shape[0] != ego_xy.shape[0]:
        return torch.zeros(ego_xy.shape[0], device=ego_xy.device, dtype=ego_xy.dtype)

    B, T, _ = ego_xy.shape
    device = ego_xy.device
    dtype = ego_xy.dtype
    if T <= 1:
        return torch.zeros(B, device=device, dtype=dtype)

    heading_vec = torch.zeros_like(ego_xy)
    heading_vec[:, :-1] = ego_xy[:, 1:] - ego_xy[:, :-1]
    heading_vec[:, -1] = ego_xy[:, -1] - ego_xy[:, -2]
    heading_norm = torch.norm(heading_vec, dim=-1, keepdim=True)
    heading_unit = heading_vec / heading_norm.clamp_min(1e-6)
    nonzero_heading = (heading_norm.squeeze(-1) >= 1e-6).to(dtype)

    # [B, T, L, N, 2]
    vec = lane_lines[:, None, :, :, :] - ego_xy[:, :, None, None, :]
    abs_dist = torch.norm(vec, dim=-1)
    heading_expand = heading_unit[:, :, None, None, :]
    cross_z = heading_expand[..., 0] * vec[..., 1] - heading_expand[..., 1] * vec[..., 0]
    signed_dist = torch.sign(cross_z) * abs_dist

    inf = torch.full_like(abs_dist, float("inf"))
    left_dist = torch.where(signed_dist > 0, abs_dist, inf).amin(dim=(2, 3))
    right_dist = torch.where(signed_dist < 0, abs_dist, inf).amin(dim=(2, 3))

    left_loss = F.relu(float(margin) - left_dist).pow(2)
    right_loss = F.relu(float(margin) - right_dist).pow(2)
    clearance_loss = torch.nan_to_num(left_loss + right_loss, nan=0.0, posinf=0.0, neginf=0.0)
    return (clearance_loss * nonzero_heading).mean(dim=1)


def compute_metric_solid_lane_violation(
    ego_traj: torch.Tensor,
    solid_lane_points: Optional[torch.Tensor],
    solid_lane_mask: Optional[torch.Tensor],
    frame_valid_mask: Optional[torch.Tensor] = None,
    gt_reference_line: Optional[torch.Tensor] = None,
    use_gt_safe_side: bool = False,
    time_weights: Tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    hard_max_weight: float = 0.0,
    margin: float = 0.05,
    return_per_frame: bool = False,
) -> torch.Tensor:
    """Metric-aligned full-footprint violation against GT solid boundaries.

    When a GT reference is available, the signed constraint keeps the ego on
    the same side of each nearby solid line as the collision-free GT path.
    This avoids the ambiguous gradient of an unsigned distance after the
    predicted footprint has already crossed a boundary.
    """
    B, T, _ = ego_traj.shape
    device, dtype = ego_traj.device, ego_traj.dtype
    if (
        solid_lane_points is None
        or solid_lane_mask is None
        or solid_lane_points.ndim != 4
        or solid_lane_points.shape[0] != B
        or solid_lane_points.shape[1] == 0
        or solid_lane_points.shape[2] < 2
    ):
        num_frames = T if T <= 6 else 6
        values = ego_traj.new_zeros((B, num_frames))
        return values if return_per_frame else ego_traj.new_zeros(B)

    frame_indices = list(range(T)) if T <= 6 else [min(i, T - 1) for i in (4, 9, 14, 19, 24, 29)]
    states = ego_traj[:, frame_indices]
    xy = states[..., :2]
    # Match PlanningMetric.evaluate_lane_edge_coll exactly: infer footprint yaw
    # from the six 2 Hz points and retain the previous heading below 0.1 m.
    # The PNN frame points forward along +x, so the stationary initial yaw is 0.
    yaw_steps = []
    previous_yaw = torch.zeros(B, device=device, dtype=dtype)
    for frame in range(xy.shape[1]):
        if xy.shape[1] == 1:
            direction = torch.zeros_like(xy[:, frame])
            direction[:, 0] = 1.0
        elif frame == 0:
            direction = xy[:, 1] - xy[:, 0]
        else:
            direction = xy[:, frame] - xy[:, frame - 1]
        moving = direction.norm(dim=-1) > 0.1
        # atan2(0, 0) has undefined backward even when torch.where later
        # selects the previous yaw. Replace stationary directions before atan2
        # so stopped samples cannot inject NaN gradients into ControlNet.
        fallback_direction = torch.stack(
            [torch.cos(previous_yaw), torch.sin(previous_yaw)], dim=-1
        )
        safe_direction = torch.where(moving[:, None], direction, fallback_direction)
        candidate_yaw = torch.atan2(safe_direction[:, 1], safe_direction[:, 0])
        previous_yaw = torch.where(
            moving,
            candidate_yaw,
            previous_yaw,
        )
        yaw_steps.append(previous_yaw)
    yaw = torch.stack(yaw_steps, dim=1)

    heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    left = torch.stack([-heading[..., 1], heading[..., 0]], dim=-1)
    center = xy + float(OFFICIAL_EGO_CENTER_FORWARD_OFFSET) * heading

    points = solid_lane_points.to(device=ego_traj.device, dtype=ego_traj.dtype)
    line_start, line_end = points[:, :, :-1], points[:, :, 1:]
    segment = line_end - line_start
    segment_sq = segment.pow(2).sum(dim=-1).clamp_min(1e-8)
    rel = center[:, :, None, None] - line_start[:, None]
    projection = ((rel * segment[:, None]).sum(dim=-1) / segment_sq[:, None]).clamp(0.0, 1.0)
    nearest = line_start[:, None] + projection[..., None] * segment[:, None]
    offset = center[:, :, None, None] - nearest
    distance = offset.norm(dim=-1)

    segment_unit = segment / segment_sq.sqrt().unsqueeze(-1)
    segment_normal = torch.stack([-segment_unit[..., 1], segment_unit[..., 0]], dim=-1)[:, None]
    use_signed_side = (
        bool(use_gt_safe_side)
        and gt_reference_line is not None
        and gt_reference_line.ndim == 3
        and gt_reference_line.shape[0] == B
        and gt_reference_line.shape[1] >= 2
    )
    if use_signed_side:
        reference = gt_reference_line.to(device=device, dtype=dtype)
        # Determine a stable safe side for every solid-line segment from the
        # closest point on the complete GT reference, not only the GT point at
        # the same horizon. A deviating PNN trajectory may hit another segment
        # of a curved polyline that the old same-time filter silently removed.
        ref_rel = reference[:, :, None, None, :] - line_start[:, None]
        ref_projection = (
            (ref_rel * segment[:, None]).sum(dim=-1) / segment_sq[:, None]
        ).clamp(0.0, 1.0)
        ref_nearest = line_start[:, None] + ref_projection[..., None] * segment[:, None]
        ref_offset = reference[:, :, None, None, :] - ref_nearest
        ref_distance = ref_offset.norm(dim=-1)
        nearest_ref_index = ref_distance.argmin(dim=1, keepdim=True)
        ref_signed_all = (ref_offset * segment_normal).sum(dim=-1)
        ref_signed = torch.gather(ref_signed_all, 1, nearest_ref_index).squeeze(1)
        safe_side = torch.where(ref_signed >= 0, 1.0, -1.0).detach()[:, None]
        signed_distance = safe_side * (offset * segment_normal).sum(dim=-1)
        normal = segment_normal
        # The official metric only collides with finite segments near the ego
        # footprint. Applying a signed half-plane constraint to remote lines
        # creates huge false penalties whenever the map contains another road.
        # Keep every segment close enough to touch the footprint plus 0.5 m of
        # look-ahead clearance; this also preserves multiple nearby boundaries.
        relevance_radius = (
            math.hypot(OFFICIAL_EGO_LENGTH / 2.0, OFFICIAL_EGO_WIDTH / 2.0)
            + float(margin)
            + 0.5
        )
        nearest_ref_segment = distance <= relevance_radius
    else:
        nearest_ref_segment = torch.ones_like(distance, dtype=torch.bool)
        signed_distance = distance
        offset_unit = offset / distance.clamp_min(1e-6).unsqueeze(-1)
        normal = torch.where((distance > 1e-6).unsqueeze(-1), offset_unit, segment_normal)
    longitudinal_support = (
        normal * heading[:, :, None, None]
    ).sum(dim=-1).abs() * (OFFICIAL_EGO_LENGTH / 2.0)
    lateral_support = (
        normal * left[:, :, None, None]
    ).sum(dim=-1).abs() * (OFFICIAL_EGO_WIDTH / 2.0)
    clearance = signed_distance - longitudinal_support - lateral_support
    violation = F.relu(float(margin) - clearance)

    line_valid = solid_lane_mask[:, None, :, None].to(device=ego_traj.device).bool()
    segment_valid = (segment_sq[:, None] > 1e-8)
    violation = torch.where(
        line_valid & segment_valid & nearest_ref_segment,
        violation,
        torch.zeros_like(violation),
    )
    frame_violation = violation.amax(dim=(2, 3))
    if frame_valid_mask is None:
        valid = torch.ones_like(frame_violation)
    else:
        valid = frame_valid_mask[:, :frame_violation.shape[1]].to(
            device=frame_violation.device, dtype=frame_violation.dtype
        )
    if return_per_frame:
        return frame_violation * valid
    weights = torch.as_tensor(time_weights, device=device, dtype=dtype).flatten()
    if weights.numel() != frame_violation.shape[1]:
        weights = F.interpolate(
            weights.view(1, 1, -1),
            size=frame_violation.shape[1],
            mode="linear",
            align_corners=True,
        ).view(-1)
    weighted_valid = valid * weights.view(1, -1)
    weighted_mean = (
        (frame_violation * weighted_valid).sum(dim=1)
        / weighted_valid.sum(dim=1).clamp_min(1.0)
    )
    hard_weight = min(max(float(hard_max_weight), 0.0), 1.0)
    if hard_weight <= 0.0:
        return weighted_mean
    valid_violation = torch.where(
        valid.bool(), frame_violation, torch.full_like(frame_violation, float("-inf"))
    )
    hard_max = valid_violation.amax(dim=1)
    hard_max = torch.where(torch.isfinite(hard_max), hard_max, torch.zeros_like(hard_max))
    return (1.0 - hard_weight) * weighted_mean + hard_weight * hard_max


def compute_obstacle_clearance_loss_per_sample(
    ego_traj: torch.Tensor,
    ped_traj: torch.Tensor,
    veh_traj: torch.Tensor,
    ped_mask: Optional[torch.Tensor],
    veh_mask: Optional[torch.Tensor],
    veh_margin: float = 2.5,
    ped_margin: float = 1.8,
    time_weights: Tuple[float, float, float] = (1.5, 1.2, 1.0),
    topk: int = 3,
) -> torch.Tensor:
    """Dense ego-agent clearance penalty aligned with raw ACR.

    The previous implementation averaged clearance over every valid actor,
    which diluted the exact failure mode we care about: one close pedestrian or
    vehicle.  This version uses a max/top-k aggregation per timestep and puts
    larger weights on the first 1-2 seconds where current raw ACR is worst.
    """
    B, T, _ = ego_traj.shape
    device = ego_traj.device
    dtype = ego_traj.dtype
    per_agent_losses = []

    def _agent_loss(agent_traj: torch.Tensor, agent_mask: Optional[torch.Tensor], margin: float) -> Optional[torch.Tensor]:
        if agent_traj is None or agent_traj.numel() == 0 or agent_traj.shape[1] == 0:
            return None
        agent_xy = agent_traj[:, :, :T, :2]
        d = torch.norm(ego_traj[:, None, :, :2] - agent_xy, dim=-1)
        close_loss = F.relu(float(margin) - d).pow(2)
        if agent_mask is not None:
            valid = agent_mask[:, :, None].bool()
        else:
            valid = torch.ones(d.shape[:2] + (1,), device=device, dtype=torch.bool)
        return torch.where(valid, close_loss, torch.zeros_like(close_loss))

    veh_loss = _agent_loss(veh_traj, veh_mask, veh_margin)
    ped_loss = _agent_loss(ped_traj, ped_mask, ped_margin)
    if veh_loss is not None:
        per_agent_losses.append(veh_loss)
    if ped_loss is not None:
        per_agent_losses.append(ped_loss)
    if not per_agent_losses:
        return torch.zeros(B, device=device, dtype=dtype)

    # [B, Nall, T]
    losses = torch.cat(per_agent_losses, dim=1)
    k = min(max(int(topk), 1), losses.shape[1])
    top = torch.topk(losses, k=k, dim=1).values
    # Max dominates raw-ACR-like behaviour; top-k mean keeps gradients stable.
    per_timestep = 0.7 * top[:, 0, :] + 0.3 * top.mean(dim=1)

    idxs = [min(9, T - 1), min(19, T - 1), min(29, T - 1)]
    endpoint_weights = torch.tensor(time_weights, device=device, dtype=dtype).clamp_min(0.0)
    dense_weights = torch.zeros(T, device=device, dtype=dtype)
    prev = 0
    for w, idx in zip(endpoint_weights, idxs):
        dense_weights[prev : idx + 1] = w
        prev = idx + 1
    if prev < T:
        dense_weights[prev:] = endpoint_weights[-1]
    return (per_timestep * dense_weights[None, :]).sum(dim=1) / dense_weights.sum().clamp_min(1e-6)

def compute_route_speed_threshold_loss_per_sample(
    ego_traj: torch.Tensor,
    ego_state: torch.Tensor,
    u_ego: torch.Tensor,
    speed_margin: float = 1.0,
    endpoint_weights: Tuple[float, float, float] = (2.0, 1.2, 0.5),
    brake_trigger_margin: float = 1.0,
    positive_accel_threshold: float = 0.3,
    brake_accel_weight: float = 0.2,
) -> torch.Tensor:
    """Penalize obvious overspeeding relative to route-derived targets.

    This is intentionally thresholded: normal acceleration/braking is not
    punished. A sample is treated as a braking/stop-like case only when the
    current speed is clearly larger than the average speed required to reach
    the 3s route target. In those cases we additionally discourage positive
    acceleration in the first second.
    """
    B, T, _ = ego_traj.shape
    device = ego_traj.device
    dtype = ego_traj.dtype
    indices = [min(9, T - 1), min(19, T - 1), min(29, T - 1)]
    times = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
    weights = torch.tensor(endpoint_weights, device=device, dtype=dtype)

    start_xy = ego_state[:, :2]
    route_points = torch.stack(
        [ego_state[:, 4:6], ego_state[:, 6:8], ego_state[:, 8:10]],
        dim=1,
    )
    route_dist = torch.norm(route_points - start_xy[:, None, :], dim=-1)
    route_speed_upper = route_dist / times.view(1, 3)

    pred_speed = torch.stack([ego_traj[:, idx, 3] for idx in indices], dim=1).clamp_min(0.0)
    speed_excess = F.relu(pred_speed - (route_speed_upper + float(speed_margin)))
    speed_loss = (speed_excess.pow(2) * weights.view(1, 3)).sum(dim=1) / weights.sum().clamp_min(1e-6)

    v0 = ego_state[:, 3].clamp_min(0.0)
    need_brake = (v0 > route_speed_upper[:, 2] + float(brake_trigger_margin)).to(dtype)
    if u_ego.shape[1] > 0 and float(brake_accel_weight) > 0.0:
        first_horizon = min(10, u_ego.shape[1])
        positive_accel_excess = F.relu(u_ego[:, :first_horizon, 0] - float(positive_accel_threshold))
        brake_loss = positive_accel_excess.pow(2).mean(dim=1) * need_brake
    else:
        brake_loss = torch.zeros(B, device=device, dtype=dtype)

    return speed_loss + float(brake_accel_weight) * brake_loss


def compute_dense_route_speed_loss_per_sample(
    ego_traj: torch.Tensor,
    ego_state: torch.Tensor,
    speed_margin: float = 0.5,
) -> torch.Tensor:
    """Penalize overspeeding at every rollout step using route segment speeds.

    The previous route-speed loss only checked 1/2/3s endpoints. Current B2D
    failures show high speed and jerk between endpoints, especially before lane
    edge violations. This dense version uses route target distances in
    [0,1], [1,2], [2,3] seconds and gives gradients at all 0.1s rollout steps.
    """
    B, T, _ = ego_traj.shape
    device = ego_traj.device
    dtype = ego_traj.dtype
    if T <= 0:
        return torch.zeros(B, device=device, dtype=dtype)

    start_xy = ego_state[:, :2]
    route_points = torch.stack(
        [ego_state[:, 4:6], ego_state[:, 6:8], ego_state[:, 8:10]],
        dim=1,
    )
    segment_start = torch.cat([start_xy[:, None, :], route_points[:, :2]], dim=1)
    segment_end = route_points
    segment_speed = torch.norm(segment_end - segment_start, dim=-1).clamp_min(0.0)

    step_idx = torch.arange(T, device=device)
    seg_idx = torch.div(step_idx, 10, rounding_mode="floor").clamp(max=2)
    speed_upper = segment_speed[:, seg_idx] + float(speed_margin)
    pred_speed = ego_traj[:, :, 3].clamp_min(0.0)
    speed_excess = F.relu(pred_speed - speed_upper)

    # Early overspeeding is more likely to create lane-edge and pedestrian
    # conflicts before the planner can recover.
    time_weights = torch.ones(T, device=device, dtype=dtype)
    time_weights[: min(T, 10)] = 1.5
    time_weights[min(T, 20):] = 0.75
    return (speed_excess.pow(2) * time_weights.view(1, T)).sum(dim=1) / time_weights.sum().clamp_min(1e-6)


def compute_forward_overshoot_loss_per_sample(
    ego_traj: torch.Tensor,
    ego_state: torch.Tensor,
    margin: float = 0.3,
    endpoint_weights: Tuple[float, float, float] = (2.0, 1.5, 0.8),
) -> torch.Tensor:
    """Penalize only forward progress beyond the 1/2/3s route targets.

    This is intentionally one-sided. It is different from the route L2 loss:
    lateral detours and being slightly conservative are not punished here. The
    term targets the B2D failure mode where the rollout keeps a plausible speed
    but advances past a near route target, then hits lane markings/objects.
    """
    B, T, _ = ego_traj.shape
    device = ego_traj.device
    dtype = ego_traj.dtype
    if T <= 0:
        return torch.zeros(B, device=device, dtype=dtype)

    indices = [min(9, T - 1), min(19, T - 1), min(29, T - 1)]
    start_xy = ego_state[:, :2]
    route_points = torch.stack(
        [ego_state[:, 4:6], ego_state[:, 6:8], ego_state[:, 8:10]],
        dim=1,
    )
    route_vec = route_points - start_xy[:, None, :]
    target_progress = torch.norm(route_vec, dim=-1)
    route_dir = route_vec / target_progress.clamp_min(1e-4).unsqueeze(-1)
    pred_points = torch.stack([ego_traj[:, idx, :2] for idx in indices], dim=1)
    pred_progress = ((pred_points - start_xy[:, None, :]) * route_dir).sum(dim=-1)
    overshoot = F.relu(pred_progress - target_progress - float(margin))
    weights = torch.tensor(endpoint_weights, device=device, dtype=dtype)
    valid = (target_progress > 1e-3).to(dtype)
    weighted = overshoot.pow(2) * weights.view(1, 3) * valid
    denom = (weights.view(1, 3) * valid).sum(dim=1).clamp_min(1e-6)
    return weighted.sum(dim=1) / denom


def compute_rollout_comfort_threshold_loss_per_sample(
    ego_traj: torch.Tensor,
    acc_threshold: float = 2.40,
    min_lon_accel: float = -4.05,
    lat_accel_threshold: float = 4.89,
    jerk_threshold: float = 4.13,
    yaw_rate_threshold: float = 0.95,
    yaw_accel_threshold: float = 1.93,
    jerk_weight: float = 1.0,
    acc_weight: float = 0.35,
    yaw_rate_weight: float = 0.25,
    yaw_accel_weight: float = 0.35,
) -> torch.Tensor:
    """Thresholded comfort loss on rollout states.

    This is deliberately soft-thresholded instead of blindly minimizing all
    acceleration. We only punish values above Bench2Drive-style comfort bounds,
    matching the user's goal: reduce jerk/discontinuity without making the car
    timid in normal maneuvers.
    """
    B, T, _ = ego_traj.shape
    device = ego_traj.device
    dtype = ego_traj.dtype
    if T <= 1:
        return torch.zeros(B, device=device, dtype=dtype)

    v = ego_traj[:, :, 3].clamp_min(0.0)
    theta = ego_traj[:, :, 2]
    lon_acc = torch.diff(v, dim=1) / DT
    yaw_rate = torch.atan2(
        torch.sin(torch.diff(theta, dim=1)),
        torch.cos(torch.diff(theta, dim=1)),
    ) / DT

    lon_acc_loss = (
        F.relu(lon_acc - float(acc_threshold)).pow(2)
        + F.relu(float(min_lon_accel) - lon_acc).pow(2)
    ).mean(dim=1)
    lat_acc = v[:, 1:] * yaw_rate
    lat_acc_loss = F.relu(
        lat_acc.abs() - float(lat_accel_threshold)
    ).pow(2).mean(dim=1)
    acc_loss = lon_acc_loss + lat_acc_loss
    yaw_rate_loss = F.relu(yaw_rate.abs() - float(yaw_rate_threshold)).pow(2).mean(dim=1)

    if T > 2:
        lon_jerk = torch.diff(lon_acc, dim=1) / DT
        yaw_accel = torch.diff(yaw_rate, dim=1) / DT
        jerk_loss = F.relu(lon_jerk.abs() - float(jerk_threshold)).pow(2).mean(dim=1)
        yaw_accel_loss = F.relu(yaw_accel.abs() - float(yaw_accel_threshold)).pow(2).mean(dim=1)
    else:
        jerk_loss = torch.zeros(B, device=device, dtype=dtype)
        yaw_accel_loss = torch.zeros(B, device=device, dtype=dtype)

    return (
        float(acc_weight) * acc_loss
        + float(jerk_weight) * jerk_loss
        + float(yaw_rate_weight) * yaw_rate_loss
        + float(yaw_accel_weight) * yaw_accel_loss
    )


def compute_original_control_loss_with_adaptive_weights(
    u_ego,
    u_peds,
    u_vehs,
    ego_traj,
    ped_traj,
    veh_traj,
    ego_state,
    ped_states,
    veh_states,
    lane_points,
    ped_mask,
    veh_mask,
    device,
    lane_loss_module,
    safety_loss_module,
    soft_lambda_module,
    cost_weights: torch.Tensor,
    default_weights: torch.Tensor,
    ped_traj_safety: Optional[torch.Tensor] = None,
    veh_traj_safety: Optional[torch.Tensor] = None,
    safety_loss_weight: Any = 1.0,
    gt_actor_boxes_2hz: Optional[torch.Tensor] = None,
    gt_actor_mask_2hz: Optional[torch.Tensor] = None,
    gt_obj_frame_valid_mask: Optional[torch.Tensor] = None,
    official_fut_valid_mask: Optional[torch.Tensor] = None,
    metric_safety_margin: float = 0.0,
    metric_safety_topk: int = 1,
    metric_safety_smooth_temperature: float = 0.0,
    metric_safety_time_weights: Optional[Tuple[float, ...]] = None,
    route_speed_loss_weight: float = 0.0,
    route_speed_margin: float = 1.0,
    route_speed_brake_weight: float = 0.2,
    route_speed_brake_trigger_margin: float = 1.0,
    route_speed_positive_accel_threshold: float = 0.3,
    ego_track_time_weights: Tuple[float, float, float, float, float, float] = (1.8, 2.5, 1.8, 1.2, 0.8, 0.5),
    lane_clearance_loss_weight: float = 0.0,
    lane_clearance_margin: float = 0.8,
    gt_solid_lane_points: Optional[torch.Tensor] = None,
    gt_solid_lane_mask: Optional[torch.Tensor] = None,
    gt_lane_frame_valid_mask: Optional[torch.Tensor] = None,
    gt_reference_line: Optional[torch.Tensor] = None,
    metric_lane_loss_weight: float = 0.0,
    metric_lane_margin: float = 0.05,
    metric_lane_use_gt_safe_side: bool = False,
    metric_lane_time_weights: Tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    metric_lane_hard_max_weight: float = 0.0,
    disable_predicted_lane_losses: bool = False,
    dense_speed_loss_weight: float = 0.0,
    dense_speed_margin: float = 0.5,
    forward_overshoot_loss_weight: float = 0.0,
    forward_overshoot_margin: float = 0.3,
    forward_overshoot_time_weights: Tuple[float, float, float] = (2.0, 1.5, 0.8),
    obstacle_clearance_loss_weight: float = 0.0,
    obstacle_clearance_margin_veh: float = 2.5,
    obstacle_clearance_margin_ped: float = 1.8,
    obstacle_clearance_time_weights: Tuple[float, float, float] = (1.5, 1.2, 1.0),
    obstacle_clearance_topk: int = 3,
    comfort_loss_weight: float = 0.0,
    comfort_acc_threshold: float = 2.40,
    comfort_min_lon_accel: float = -4.05,
    comfort_lat_accel_threshold: float = 4.89,
    comfort_jerk_threshold: float = 4.13,
    comfort_yaw_rate_threshold: float = 0.95,
    comfort_yaw_accel_threshold: float = 1.93,
):
    B, Np, T = u_peds.shape[:3]
    Nv = u_vehs.shape[1]

    if ped_mask is None:
        ped_mask = torch.ones((B, Np), dtype=torch.bool, device=device)
    if veh_mask is None:
        veh_mask = torch.ones((B, Nv), dtype=torch.bool, device=device)

    if ped_traj_safety is None:
        ped_traj_safety = ped_traj.detach()
    if veh_traj_safety is None:
        veh_traj_safety = veh_traj.detach()

    soft_lambdas = soft_lambda_module()
    scale = torch.nan_to_num(
        cost_weights / default_weights.clamp_min(1e-3),
        nan=1.0,
        posinf=10.0,
        neginf=0.1,
    ).clamp(0.05, 10.0)
    idx = {name: i for i, name in enumerate(COST_NAMES)}

    loss_control_ego = soft_lambdas["ctrl_ego"] * (
        scale[:, idx["acceleration"]] * (u_ego[..., 0] ** 2).mean(dim=1)
        + scale[:, idx["steering"]] * (u_ego[..., 1] ** 2).mean(dim=1)
    )

    ctrl_p = (u_peds[..., 0] ** 2 + u_peds[..., 1] ** 2)
    ctrl_v = (u_vehs[..., 0] ** 2 + u_vehs[..., 1] ** 2)
    loss_control_p = masked_mean_per_sample(ctrl_p, ped_mask) * soft_lambdas["ctrl_p"]
    loss_control_v = masked_mean_per_sample(ctrl_v, veh_mask) * soft_lambdas["ctrl_v"]

    loss_track_ego = (
        3.0
        * scale[:, idx["route_target"]]
        * compute_ego_track_loss_with_lane_check_per_sample(
            ego_traj,
            ego_state,
            lane_points,
            time_weights=ego_track_time_weights,
        )
    )
    loss_route_speed = float(route_speed_loss_weight) * compute_route_speed_threshold_loss_per_sample(
        ego_traj=ego_traj,
        ego_state=ego_state,
        u_ego=u_ego,
        speed_margin=route_speed_margin,
        brake_trigger_margin=route_speed_brake_trigger_margin,
        positive_accel_threshold=route_speed_positive_accel_threshold,
        brake_accel_weight=route_speed_brake_weight,
    )
    loss_dense_speed = float(dense_speed_loss_weight) * compute_dense_route_speed_loss_per_sample(
        ego_traj=ego_traj,
        ego_state=ego_state,
        speed_margin=dense_speed_margin,
    )
    loss_forward_overshoot = float(forward_overshoot_loss_weight) * compute_forward_overshoot_loss_per_sample(
        ego_traj=ego_traj,
        ego_state=ego_state,
        margin=forward_overshoot_margin,
        endpoint_weights=forward_overshoot_time_weights,
    )
    loss_obstacle_clearance = float(obstacle_clearance_loss_weight) * compute_obstacle_clearance_loss_per_sample(
        ego_traj=ego_traj,
        ped_traj=ped_traj_safety,
        veh_traj=veh_traj_safety,
        ped_mask=ped_mask,
        veh_mask=veh_mask,
        veh_margin=obstacle_clearance_margin_veh,
        ped_margin=obstacle_clearance_margin_ped,
        time_weights=obstacle_clearance_time_weights,
        topk=obstacle_clearance_topk,
    )
    loss_comfort_threshold = float(comfort_loss_weight) * compute_rollout_comfort_threshold_loss_per_sample(
        ego_traj=ego_traj,
        acc_threshold=comfort_acc_threshold,
        min_lon_accel=comfort_min_lon_accel,
        lat_accel_threshold=comfort_lat_accel_threshold,
        jerk_threshold=comfort_jerk_threshold,
        yaw_rate_threshold=comfort_yaw_rate_threshold,
        yaw_accel_threshold=comfort_yaw_accel_threshold,
    )

    ped_goal_err = (ped_traj[:, :, -1, 0] - ped_states[:, :, 4]) ** 2 + (
        ped_traj[:, :, -1, 1] - ped_states[:, :, 5]
    ) ** 2
    veh_goal_err = (veh_traj[:, :, -1, 0] - veh_states[:, :, 4]) ** 2 + (
        veh_traj[:, :, -1, 1] - veh_states[:, :, 5]
    ) ** 2
    loss_track_p = masked_mean_per_sample(ped_goal_err, ped_mask) * 0.5
    loss_track_v = masked_mean_per_sample(veh_goal_err, veh_mask) * 2.0

    if gt_actor_boxes_2hz is not None and gt_actor_mask_2hz is not None:
        g_safety = compute_gt_actor_box_violation(
            ego_traj=ego_traj,
            gt_actor_boxes_2hz=gt_actor_boxes_2hz,
            gt_actor_mask_2hz=gt_actor_mask_2hz,
            rect_distance=safety_loss_module.rect_dist,
            sample_valid_mask=official_fut_valid_mask,
            frame_valid_mask=gt_obj_frame_valid_mask,
            margin=metric_safety_margin,
            topk=metric_safety_topk,
            smooth_temperature=metric_safety_smooth_temperature,
            time_weights=metric_safety_time_weights,
        )
    else:
        g_safety = safety_loss_module.compute_constraint_violation(
            ego_traj,
            ped_traj_safety,
            veh_traj_safety,
            ped_mask,
            veh_mask,
        )
    g_lane = lane_loss_module.compute_constraint_violation(ego_traj[:, :, :2], lane_points)
    lane_scale = 0.5 * (scale[:, idx["lane_xy"]] + scale[:, idx["lane_theta"]])
    safe_scale = scale[:, idx["safe"]]
    safety_sample_weight = torch.as_tensor(
        safety_loss_weight,
        device=device,
        dtype=g_safety.dtype,
    )
    if safety_sample_weight.ndim == 0:
        safety_sample_weight = safety_sample_weight.expand(B)
    else:
        safety_sample_weight = safety_sample_weight.reshape(B)
    loss_safety = (
        safety_sample_weight
        * safe_scale
        * safety_loss_module.lambda_val.detach()
        * (F.softplus(10.0 * g_safety) - math.log(2.0))
        / 10.0
    )
    if disable_predicted_lane_losses:
        loss_lane_hard = torch.zeros_like(g_lane)
        loss_lane_clearance = torch.zeros_like(g_lane)
    else:
        loss_lane_hard = (
            lane_scale
            * lane_loss_module.lambda_val.detach()
            * (F.softplus(10.0 * g_lane) - math.log(2.0))
            / 10.0
        )
        loss_lane_clearance = (
            float(lane_clearance_loss_weight)
            * lane_scale
            * compute_lane_clearance_loss_per_sample(
                ego_traj[:, :, :2],
                lane_points,
                margin=lane_clearance_margin,
            )
        )
    g_metric_lane = compute_metric_solid_lane_violation(
        ego_traj=ego_traj,
        solid_lane_points=gt_solid_lane_points,
        solid_lane_mask=gt_solid_lane_mask,
        frame_valid_mask=gt_lane_frame_valid_mask,
        gt_reference_line=gt_reference_line,
        use_gt_safe_side=metric_lane_use_gt_safe_side,
        time_weights=metric_lane_time_weights,
        hard_max_weight=metric_lane_hard_max_weight,
        margin=metric_lane_margin,
    )
    loss_metric_lane = float(metric_lane_loss_weight) * lane_scale * g_metric_lane

    ped_vel = ped_traj[..., 3]
    veh_vel = veh_traj[..., 3]
    loss_vel_p = masked_mean_per_sample(torch.clamp(ped_vel - 2.0, min=0.0), ped_mask) * soft_lambdas["vel_p"]
    loss_vel_v = masked_mean_per_sample(torch.clamp(veh_vel - 15.0, min=0.0), veh_mask) * soft_lambdas["vel_v"]

    if u_ego.shape[1] > 1:
        delta_a = u_ego[:, 1:, 0] - u_ego[:, :-1, 0]
        delta_delta = u_ego[:, 1:, 1] - u_ego[:, :-1, 1]
        exceed = torch.clamp(delta_delta.abs() - 0.04, min=0.0)
        loss_control_rate_ego = soft_lambdas["ctrl_ego"] * (
            scale[:, idx["jerk"]] * delta_a.pow(2).mean(dim=1)
            + 5.0
            * scale[:, idx["steering_change"]]
            * F.softplus(10.0 * exceed).mean(dim=1)
        )
    else:
        loss_control_rate_ego = torch.zeros(B, device=device, dtype=u_ego.dtype)

    if u_vehs.shape[2] > 1:
        delta_a_v = u_vehs[:, :, 1:, 0] - u_vehs[:, :, :-1, 0]
        delta_d_v = u_vehs[:, :, 1:, 1] - u_vehs[:, :, :-1, 1]
        exceed_v = torch.clamp(delta_d_v.abs() - 0.04, min=0.0)
        veh_rate = delta_a_v.pow(2).mean(dim=-1) + 5.0 * F.softplus(10.0 * exceed_v).mean(dim=-1)
        loss_control_rate_veh = masked_mean_per_sample(veh_rate, veh_mask) * soft_lambdas["ctrl_v"]
    else:
        loss_control_rate_veh = torch.zeros(B, device=device, dtype=u_ego.dtype)

    l1_ego = u_ego.abs().mean(dim=(1, 2))
    l1_p = masked_mean_per_sample(u_peds.abs().sum(dim=-1), ped_mask)
    l1_v = masked_mean_per_sample(u_vehs.abs().sum(dim=-1), veh_mask)
    l1_penalty = soft_lambdas["l1"] * (l1_ego + l1_p + l1_v)

    residual = (
        loss_control_ego + loss_control_p + loss_control_v +
        loss_track_ego + loss_route_speed + loss_dense_speed + loss_forward_overshoot + loss_track_p + loss_track_v +
        loss_safety + loss_obstacle_clearance + loss_lane_hard + loss_lane_clearance + loss_metric_lane +
        loss_vel_p + loss_vel_v + loss_comfort_threshold +
        loss_control_rate_ego + loss_control_rate_veh +
        l1_penalty
    )
    aux = {
        "g_lane": g_lane,
        "g_metric_lane": g_metric_lane,
        "g_safety": g_safety,
        "loss_track_ego": loss_track_ego,
        "loss_safety": loss_safety,
        "loss_lane_hard": loss_lane_hard,
        "loss_lane_clearance": loss_lane_clearance,
        "loss_metric_lane": loss_metric_lane,
        "loss_comfort_threshold": loss_comfort_threshold,
        "loss_control_rate_ego": loss_control_rate_ego,
        "loss_teacher_trust": torch.zeros_like(loss_track_ego),
    }
    return residual, aux


# ============================================================
# Main train
# ============================================================
def train(rank, world_size, cfg_runtime):
    global PROGRESS_OVERSHOOT_WEIGHT, REFERENCE_FORWARD_OFFSET
    PROGRESS_OVERSHOOT_WEIGHT = float(
        cfg_runtime.get("progress_overshoot_weight", PROGRESS_OVERSHOOT_WEIGHT)
    )
    REFERENCE_FORWARD_OFFSET = float(
        cfg_runtime.get("reference_forward_offset", REFERENCE_FORWARD_OFFSET)
    )

    setup(rank, world_size)
    device = torch.device("cuda", rank)
    torch.backends.cudnn.benchmark = True

    save_dir = cfg_runtime["save_dir"]
    mkdir(save_dir)
    mkdir(os.path.join(save_dir, "checkpoints"))
    mkdir(os.path.join(save_dir, "plots"))
    mkdir(os.path.join(save_dir, "tb"))

    writer = SummaryWriter(os.path.join(save_dir, "tb"), flush_secs=30) if rank == 0 else None

    def maybe_unfreeze_weightnet_highlevel(weight_model_ddp: nn.Module):
        wm = unwrap_module(weight_model_ddp)
        trainable_keywords = ["attn_fusion", "res_fc1", "res_fc2", "weight_decoder"]
        for name, p in wm.named_parameters():
            if any(k in name for k in trainable_keywords):
                p.requires_grad = True

    _original_load = torch.load

    def _safe_cpu_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return _original_load(*args, **kwargs)

    old_data_path = cfg_runtime["old_train_data_path"]
    new_data_path = cfg_runtime["new_train_data_path"]
    if cfg_runtime.get("coord_convention"):
        os.environ["PNN_COORD_CONVENTION"] = str(cfg_runtime["coord_convention"])

    try:
        torch.load = _safe_cpu_load
        dataset = PairedOldNewDataset(
            old_data_path,
            new_data_path,
            supervision_pt_path=cfg_runtime.get("supervision_data_path"),
        )
    finally:
        torch.load = _original_load
    if cfg_runtime.get("require_gt_actor_boxes", False):
        required_actor_fields = {
            "gt_actor_boxes_2hz",
            "gt_actor_mask_2hz",
            "official_fut_valid_mask",
        }
        missing_actor_fields = required_actor_fields.difference(dataset.new_data)
        if missing_actor_fields:
            raise RuntimeError(
                "PNN_REQUIRE_GT_ACTOR_BOXES=1 but dataset is missing: "
                f"{sorted(missing_actor_fields)}"
            )
    if cfg_runtime.get("require_hipad_plan_2hz", False) and "hipad_plan_2hz" not in dataset.new_data:
        raise RuntimeError("PNN_REQUIRE_HIPAD_PLAN_2HZ=1 but dataset is missing hipad_plan_2hz")
    if cfg_runtime.get("require_metric_supervision", False):
        required_supervision_fields = {
            "metric_gt_actor_boxes_2hz",
            "metric_gt_actor_mask_2hz",
            "gt_solid_lane_points",
            "gt_solid_lane_mask",
            "gt_obj_collision_mask_2hz",
            "gt_lane_collision_mask_2hz",
        }
        available = dataset.supervision_data or {}
        missing = required_supervision_fields.difference(available)
        if missing:
            raise RuntimeError(
                "PNN_REQUIRE_METRIC_SUPERVISION=1 but sidecar is missing: "
                f"{sorted(missing)}"
            )
    if cfg_runtime.get("require_solid_lane_supervision", False):
        available = dataset.supervision_data or {}
        required_lane_fields = {
            "gt_solid_lane_points",
            "gt_solid_lane_mask",
        }
        missing = required_lane_fields.difference(available)
        if missing:
            raise RuntimeError(
                "PNN_REQUIRE_SOLID_LANE_SUPERVISION=1 but sidecar is missing: "
                f"{sorted(missing)}"
            )
        lane_points_check = available["gt_solid_lane_points"]
        lane_mask_check = available["gt_solid_lane_mask"].bool()
        if not torch.isfinite(lane_points_check).all():
            raise RuntimeError("GT solid-lane sidecar contains NaN/Inf")
        valid_lane_count = int(lane_mask_check.sum().item())
        if valid_lane_count <= 0:
            raise RuntimeError("GT solid-lane sidecar contains no valid lines")
        if rank == 0:
            print(
                "[v10] solid-lane supervision verified: "
                f"valid_lines={valid_lane_count} "
                f"samples_with_lines={int(lane_mask_check.any(dim=1).sum().item())}"
            )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg_runtime["batch_size"],
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg_runtime["num_workers"],
        pin_memory=False,
    )

    # Reuse the tensors already loaded by PairedOldNewDataset. Loading the same
    # .pt again in every DDP rank slows down restarts and increases CPU memory
    # pressure without changing the statistics.
    data_stats = dataset.old_data
    stats_q_low = float(cfg_runtime.get("stats_quantile_low", 0.0))
    stats_q_high = float(cfg_runtime.get("stats_quantile_high", 1.0))
    min_ego, max_ego = tensor_feature_minmax(
        data_stats["ego_state"],
        q_low=stats_q_low,
        q_high=stats_q_high,
    )
    min_ego = min_ego.to(device)
    max_ego = max_ego.to(device)
    min_ped, max_ped = masked_agent_stats_minmax(
        data_stats["ped_states"],
        data_stats.get("ped_mask"),
        q_low=stats_q_low,
        q_high=stats_q_high,
    )
    min_veh, max_veh = masked_agent_stats_minmax(
        data_stats["veh_states"],
        data_stats.get("veh_mask"),
        q_low=stats_q_low,
        q_high=stats_q_high,
    )
    min_ped = min_ped.to(device)
    max_ped = max_ped.to(device)
    min_veh = min_veh.to(device)
    max_veh = max_veh.to(device)
    min_static, max_static = masked_agent_stats_minmax(
        data_stats["static_states"],
        data_stats.get("static_mask"),
        feature_dim=StaticAwareControlNet.STATIC_FEATURE_DIM,
        q_low=stats_q_low,
        q_high=stats_q_high,
    )
    min_static = min_static.to(device)
    max_static = max_static.to(device)
    min_lane, max_lane = tensor_feature_minmax(
        data_stats["lane_points"][:, 0:2].reshape(-1, 2),
        q_low=stats_q_low,
        q_high=stats_q_high,
    )
    min_lane = min_lane.to(device)
    max_lane = max_lane.to(device)
    clamp_normalized_inputs = bool(cfg_runtime.get("clamp_normalized_inputs", False))
    if rank == 0:
        print(
            f"[v10] normalization stats: q_low={stats_q_low} "
            f"q_high={stats_q_high} clamp={clamp_normalized_inputs}"
        )

    policy_core = StaticAwareControlNet(
        embed_dim=cfg_runtime["embed_dim"],
        num_heads=cfg_runtime["num_heads"],
        future_steps=TRAJ_LEN,
    ).to(device)

    control_ckpt_path = cfg_runtime.get("control_ckpt_path")
    control_ckpt = {}
    if control_ckpt_path:
        control_ckpt = torch.load(control_ckpt_path, map_location=device)
        if "neural_net" not in control_ckpt:
            raise KeyError("control checkpoint is missing 'neural_net'")
        policy_core.load_legacy_control_state_dict(control_ckpt["neural_net"])
        if rank == 0:
            print(f"[v10] loaded control checkpoint: {control_ckpt_path}")
    elif rank == 0:
        if cfg_runtime.get("resume_ckpt_path"):
            print("[v10] control_ckpt_path is empty; policy will be loaded from resume_ckpt_path.")
        else:
            print("[v10] no control/resume checkpoint; training ControlNet/WeightNet from scratch.")

    if cfg_runtime.get("freeze_legacy_control", False):
        static_train_mode = cfg_runtime.get("static_train_mode", "all")
        if static_train_mode not in {"all", "gate_only"}:
            raise ValueError(
                "static_train_mode must be 'all' or 'gate_only', got "
                f"{static_train_mode!r}"
            )
        for name, parameter in policy_core.named_parameters():
            if static_train_mode == "gate_only":
                trainable = (
                    name.startswith("static_encoder.")
                    or name.startswith("static_risk_gate.")
                )
            else:
                trainable = name.startswith("static_")
            parameter.requires_grad_(trainable)
        if rank == 0:
            trainable = sum(
                parameter.numel()
                for parameter in policy_core.parameters()
                if parameter.requires_grad
            )
            print(
                "[static-v1] froze legacy ControlNet; "
                f"mode={static_train_mode} "
                f"trainable static parameters={trainable}"
            )

    teacher_policy = None
    teacher_ckpt_path = cfg_runtime.get("teacher_ckpt_path")
    if teacher_ckpt_path:
        teacher_ckpt = torch.load(teacher_ckpt_path, map_location=device)
        if "neural_net" not in teacher_ckpt:
            raise KeyError("teacher checkpoint is missing 'neural_net'")
        teacher_policy = copy.deepcopy(policy_core)
        teacher_policy.load_legacy_control_state_dict(teacher_ckpt["neural_net"])
        teacher_policy.requires_grad_(False)
        teacher_policy.eval()
        if rank == 0:
            print(f"[v10] loaded frozen trajectory teacher: {teacher_ckpt_path}")
    policy_core = DDP(policy_core, device_ids=[rank], find_unused_parameters=False)
    freeze_control_policy = bool(cfg_runtime.get("freeze_control_policy", False))
    if freeze_control_policy:
        policy_core.module.requires_grad_(False)
        policy_core.eval()
        if rank == 0:
            print("[v10] ControlNet is frozen; Stage 2 updates WeightNet only.")

    linear_dynamics = BicycleModel(WHEELBASE, dt=DT).to(device)
    nnc_dyn = StaticNNCDynamics(policy_core).to(device)

    weight_model = WeightNet(
        embed_dim=cfg_runtime["embed_dim"],
        num_heads=cfg_runtime["num_heads"],
        num_tasks=NUM_COSTS,
        temperature=cfg_runtime.get("weight_temperature", 1.5),
        use_prior_context=cfg_runtime.get("weightnet_use_prior_context", True),
        prior_context_mode=cfg_runtime.get("weightnet_prior_context_mode", "log"),
        initial_refine_gate=cfg_runtime.get("weight_initial_refine_gate", 0.01),
    ).to(device)

    if control_ckpt_path:
        load_control_encoder_to_weightnet(
            weight_net=weight_model,
            ckpt_path=control_ckpt_path,
            map_location=device,
            verbose=(rank == 0),
        )
    elif rank == 0:
        print("[v10] skip ControlNet -> WeightNet partial load for scratch training.")

    init_weight_decoder_with_default_prior(
        weight_model,
        prior_weights=cfg_runtime.get("weight_prior", (1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 3.0, 2.0)),
        residual_output=cfg_runtime.get("weightnet_outputs_residual", True),
    )

    freeze_pretrained_part(weight_model)
    weight_model = DDP(weight_model, device_ids=[rank], find_unused_parameters=False)
    maybe_unfreeze_weightnet_highlevel(weight_model)

    sample_item = dataset[0]
    num_objects = int(sample_item["ped_states"].shape[0] + sample_item["veh_states"].shape[0])

    upper_dipp = MotionPlannerCompatible(
        trajectory_len=TRAJ_LEN,
        feature_len=NUM_COSTS,
        num_objects=num_objects,
        device=device,
        optimizer_type=cfg_runtime.get("planner_optimizer", "levenberg_marquardt"),
        max_iterations=cfg_runtime.get("planner_max_iterations", 10),
        step_size=cfg_runtime.get("planner_step_size", 0.10),
    )

    lane_loss_module = LaneBoundaryLagrangianLoss(init_lambda=10.0).to(device)
    # Official object collision is ego-vs-object. Actor-vs-actor violations are
    # exogenous to ego control and would otherwise inflate an uncontrollable dual.
    safety_loss_module = SafetyConstraintLoss(
        init_lambda=10.0,
        min_dist=cfg_runtime.get("metric_safety_margin", 1.5),
        topk=cfg_runtime.get("metric_safety_topk", 3),
        veh_veh_weight=0.0,
    ).to(device)
    soft_lambda_module = SoftConstraintLambdas().to(device)
    train_soft_constraint_lambdas = bool(
        cfg_runtime.get("train_soft_constraint_lambdas", False)
    )
    soft_lambda_module.requires_grad_(train_soft_constraint_lambdas)

    if "lane_loss_module" in control_ckpt:
        lane_loss_module.load_state_dict(control_ckpt["lane_loss_module"])
    if "safety_loss_module" in control_ckpt:
        safety_loss_module.load_state_dict(control_ckpt["safety_loss_module"])
    if "soft_lambda_module" in control_ckpt:
        soft_lambda_module.load_state_dict(control_ckpt["soft_lambda_module"])

    resume_ckpt = None
    resume_ckpt_path = cfg_runtime.get("resume_ckpt_path")
    if resume_ckpt_path:
        resume_ckpt = torch.load(resume_ckpt_path, map_location=device)
        if "neural_net" in resume_ckpt:
            resume_state = resume_ckpt["neural_net"]
            if any(key.startswith("static_encoder.") for key in resume_state):
                policy_core.module.load_compatible_static_state_dict(
                    resume_state
                )
            else:
                policy_core.module.load_legacy_control_state_dict(resume_state)
            if rank == 0:
                print(f"[v10] loaded policy from resume_ckpt_path: {resume_ckpt_path}")
        if "weight_model" in resume_ckpt:
            missing, unexpected = unwrap_module(weight_model).load_state_dict(resume_ckpt["weight_model"], strict=False)
            if rank == 0:
                print(
                    f"[v10] loaded weight_model from resume_ckpt_path: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
        if "lane_loss_module" in resume_ckpt:
            lane_loss_module.load_state_dict(resume_ckpt["lane_loss_module"])
        if "safety_loss_module" in resume_ckpt:
            safety_loss_module.load_state_dict(resume_ckpt["safety_loss_module"])
        if "soft_lambda_module" in resume_ckpt:
            soft_lambda_module.load_state_dict(resume_ckpt["soft_lambda_module"])

    a_ego = torch.tensor([-10.0, -1.066], device=device).view(1, 1, 2)
    b_ego = torch.tensor([10.0, 1.066], device=device).view(1, 1, 2)
    a_peds = torch.tensor([-1.0, -math.pi / 4], device=device).view(1, 1, 1, 2)
    b_peds = torch.tensor([1.0, math.pi / 4], device=device).view(1, 1, 1, 2)
    a_vehs = torch.tensor([-10.0, -1.066], device=device).view(1, 1, 1, 2)
    b_vehs = torch.tensor([10.0, 1.066], device=device).view(1, 1, 1, 2)

    control_optimized_params = list(nnc_dyn.parameters())
    if train_soft_constraint_lambdas:
        control_optimized_params += list(soft_lambda_module.parameters())
    optimizer_control = Adam(control_optimized_params, lr=cfg_runtime["lr_control"])

    wm = unwrap_module(weight_model)
    weight_trainable_params = [p for p in wm.parameters() if p.requires_grad]
    weight_optimized_params = weight_trainable_params
    optimizer_weight = AdamW(
        weight_optimized_params,
        lr=cfg_runtime["lr_weight"],
        weight_decay=cfg_runtime.get("weight_decay_weightnet", 1e-4),
    )

    scheduler = ReduceLROnPlateau(optimizer_control, mode="min", factor=0.5, patience=20)
    if resume_ckpt is not None and cfg_runtime.get("resume_optimizer_state", True):
        if "optimizer_control" in resume_ckpt:
            optimizer_control.load_state_dict(resume_ckpt["optimizer_control"])
        if "optimizer_weight" in resume_ckpt:
            optimizer_weight.load_state_dict(resume_ckpt["optimizer_weight"])
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        if cfg_runtime.get("override_resume_lr", False):
            for param_group in optimizer_control.param_groups:
                param_group["lr"] = float(cfg_runtime["lr_control"])
            for param_group in optimizer_weight.param_groups:
                param_group["lr"] = float(cfg_runtime["lr_weight"])
        if rank == 0:
            print(
                "[v10] loaded optimizer/scheduler state from resume checkpoint "
                f"with control_lr={optimizer_control.param_groups[0]['lr']:.3e} "
                f"weight_lr={optimizer_weight.param_groups[0]['lr']:.3e}"
            )
    elif resume_ckpt is not None and rank == 0:
        print(
            "[v10] skipped optimizer/scheduler state from resume checkpoint; "
            f"using fresh control_lr={optimizer_control.param_groups[0]['lr']:.3e} "
            f"weight_lr={optimizer_weight.param_groups[0]['lr']:.3e}"
        )

    g_ema_lane = 0.0
    g_ema_safety = 0.0
    ema_momentum = 0.9
    rho_aug = 3.0
    lambda_cap = 100000.0
    decay_dual = 1e-4
    eta_dual_lane = 1e-3
    eta_dual_safety = 1e-3
    eps_dead_lane = 1e-3
    eps_dead_safety = 1e-4

    def resume_float(key: str, default: float = float("inf")) -> float:
        if resume_ckpt is None:
            return default
        value = resume_ckpt.get(key, default)
        return default if value is None else float(value)

    history_rows = list(resume_ckpt.get("history_rows", [])) if resume_ckpt is not None else []
    best_control_total = resume_float("best_control_total")
    best_l2_with_variation = resume_float("best_l2_with_variation")
    start_epoch = int(resume_ckpt.get("epoch", -1)) + 1 if resume_ckpt is not None else 0
    if rank == 0 and start_epoch > 0:
        print(f"[v10] resuming training at epoch={start_epoch}")

    prior_for_reg = torch.tensor(
        cfg_runtime.get("weight_prior", (1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 3.0, 2.0)),
        dtype=torch.float32,
        device=device,
    )

    base_lambda_entropy = cfg_runtime.get("lambda_entropy", 1e-3)
    base_lambda_diversity = cfg_runtime.get("lambda_diversity", 5e-3)
    base_lambda_kl = cfg_runtime.get("lambda_kl", 1e-3)
    weight_update_interval = max(1, int(cfg_runtime.get("weight_update_interval", 1)))
    weight_dipp_update_interval = max(
        1, int(cfg_runtime.get("weight_dipp_update_interval", 1))
    )
    weight_dipp_start_epoch = int(cfg_runtime.get("weight_dipp_start_epoch", 0))
    ema_update_interval = cfg_runtime.get("ema_update_interval", 100)
    dipp_traj_disabled = False

    for epoch in tqdm(range(start_epoch, cfg_runtime["epochs"]), desc=f"Rank{rank} Training", leave=True):
        sampler.set_epoch(epoch)

        update_control_epoch = (
            not freeze_control_policy
            and epoch >= int(cfg_runtime.get("control_update_start_epoch", 0))
        )

        if not update_control_epoch:
            nnc_dyn.eval()
        else:
            nnc_dyn.train()
        unwrap_module(weight_model).train()
        lane_loss_module.train()
        safety_loss_module.train()
        if teacher_policy is not None:
            teacher_policy.eval()
        soft_lambda_module.train()

        epoch_control_total = 0.0
        epoch_control_core = 0.0
        epoch_gt_reference_lane = 0.0
        epoch_control_components = {
            name: 0.0 for name in MONITORED_CONTROL_LOSS_COMPONENT_NAMES
        }
        epoch_weight_loss = 0.0
        epoch_weight_traj_loss = 0.0
        epoch_weight_rule_loss = 0.0
        epoch_weight_feedback_loss = 0.0
        epoch_weight_rank_loss = 0.0
        epoch_weight_sep_loss = 0.0
        epoch_weight_entropy_band_loss = 0.0
        epoch_weight_diversity_floor_loss = 0.0
        epoch_weight_extreme_loss = 0.0
        epoch_weight_pnn_collision_loss = 0.0
        epoch_weight_dipp_safety_loss = 0.0
        epoch_weight_dipp_lane_loss = 0.0
        epoch_weight_dipp_trust_loss = 0.0
        epoch_aug = 0.0
        epoch_entropy = 0.0
        epoch_diversity = 0.0
        epoch_kl = 0.0
        epoch_costw_sum = torch.zeros(NUM_COSTS, device=device)
        epoch_costw_sq_sum = torch.zeros(NUM_COSTS, device=device)
        epoch_pairwise_l2_sum = 0.0
        epoch_proxy_sums: Dict[str, torch.Tensor] = {}
        epoch_proxy_counts: Dict[str, torch.Tensor] = {}
        epoch_steps = 0

        for batch_idx, batch in enumerate(data_loader):
            max_train_batches = int(cfg_runtime.get("max_train_batches", 0))
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            optimizer_control.zero_grad()
            optimizer_weight.zero_grad()

            global_step = epoch * len(data_loader) + batch_idx
            update_weight = (
                global_step % weight_update_interval == 0
                and epoch >= cfg_runtime.get("weight_update_start_epoch", 0)
            )
            update_weight_dipp = (
                update_weight
                and epoch >= weight_dipp_start_epoch
                and global_step % weight_dipp_update_interval == 0
            )

            ego_state = batch["ego_state"].to(device).float()
            ped_states = batch["ped_states"].to(device).float()
            veh_states = batch["veh_states"].to(device).float()
            static_states = batch["static_states"].to(device).float()
            lane_points = canonicalize_lane_direction(
                batch["lane_points"].to(device).float()
            )

            ped_mask = batch.get("ped_mask", None)
            veh_mask = batch.get("veh_mask", None)
            static_mask = batch.get("static_mask", None)
            ped_mask = ped_mask.to(device).bool() if ped_mask is not None else None
            veh_mask = veh_mask.to(device).bool() if veh_mask is not None else None
            static_mask = (
                static_mask.to(device).bool() if static_mask is not None else None
            )
            ego_future_gt_new = batch["ego_future_gt_new"].to(device).float()
            ego_future_gt_valid_mask = batch["ego_future_gt_valid_mask"].to(device).bool()
            gt_reference_line = batch["gt_reference_line"].to(device).float()
            gt_reference_line_valid_mask = batch["gt_reference_line_valid_mask"].to(device).bool()
            gt_actor_boxes_2hz = batch.get("gt_actor_boxes_2hz")
            gt_actor_mask_2hz = batch.get("gt_actor_mask_2hz")
            gt_obj_collision_mask_2hz = batch.get("gt_obj_collision_mask_2hz")
            gt_solid_lane_points = batch.get("gt_solid_lane_points")
            gt_solid_lane_mask = batch.get("gt_solid_lane_mask")
            gt_lane_collision_mask_2hz = batch.get("gt_lane_collision_mask_2hz")
            official_fut_valid_mask = batch.get("official_fut_valid_mask")
            hipad_plan_2hz = batch.get("hipad_plan_2hz")
            gt_actor_boxes_2hz = (
                gt_actor_boxes_2hz.to(device).float() if gt_actor_boxes_2hz is not None else None
            )
            gt_actor_mask_2hz = (
                gt_actor_mask_2hz.to(device).bool() if gt_actor_mask_2hz is not None else None
            )
            train_raw_collisions = bool(cfg_runtime.get("train_raw_collisions", False))
            gt_obj_frame_valid_mask = (
                None
                if train_raw_collisions
                else (
                    ~gt_obj_collision_mask_2hz.to(device).bool()
                    if gt_obj_collision_mask_2hz is not None
                    else None
                )
            )
            gt_solid_lane_points = (
                gt_solid_lane_points.to(device).float()
                if gt_solid_lane_points is not None
                else None
            )
            gt_solid_lane_mask = (
                gt_solid_lane_mask.to(device).bool()
                if gt_solid_lane_mask is not None
                else None
            )
            gt_lane_frame_valid_mask = (
                None
                if train_raw_collisions
                else (
                    ~gt_lane_collision_mask_2hz.to(device).bool()
                    if gt_lane_collision_mask_2hz is not None
                    else None
                )
            )
            official_fut_valid_mask = (
                official_fut_valid_mask.to(device).bool()
                if official_fut_valid_mask is not None
                else ego_future_gt_valid_mask
            )
            hipad_plan_2hz = (
                hipad_plan_2hz.to(device).float() if hipad_plan_2hz is not None else None
            )

            B = ego_state.shape[0]
            lane_for_control, lane_for_weight = split_lane_for_control_and_weight(lane_points)

            ego_state_n = normalize(ego_state, min_ego, max_ego)
            ped_states_n = normalize(ped_states, min_ped, max_ped)
            veh_states_n = normalize(veh_states, min_veh, max_veh)
            static_states_n = normalize(
                static_states, min_static, max_static
            )
            lane_control_n = normalize(
                lane_for_control.reshape(B, -1, 2), min_lane, max_lane
            ).reshape(B, lane_for_control.shape[1], lane_for_control.shape[2], 2)
            lane_weight_n = normalize(
                lane_for_weight.reshape(B, -1, 2), min_lane, max_lane
            ).reshape(B, lane_for_weight.shape[1], lane_for_weight.shape[2], 2)
            if clamp_normalized_inputs:
                ego_state_n = ego_state_n.clamp(-1.0, 1.0)
                ped_states_n = ped_states_n.clamp(-1.0, 1.0)
                veh_states_n = veh_states_n.clamp(-1.0, 1.0)
                static_states_n = static_states_n.clamp(-1.0, 1.0)
                lane_control_n = lane_control_n.clamp(-1.0, 1.0)
                lane_weight_n = lane_weight_n.clamp(-1.0, 1.0)

            u_ego_n, u_peds_n, u_vehs_n = nnc_dyn(
                ego_state_n,
                ped_states_n,
                veh_states_n,
                lane_control_n,
                build_model_padding_mask(ped_mask),
                build_model_padding_mask(veh_mask),
                static_states_n,
                static_mask,
            )

            init_control = inverse_normalize(u_ego_n, a_ego, b_ego)
            u_peds = inverse_normalize(u_peds_n, a_peds, b_peds)
            u_vehs = inverse_normalize(u_vehs_n, a_vehs, b_vehs)

            init_control = torch.stack(
                [
                    init_control[..., 0].clamp(-MAX_ACC, MAX_ACC),
                    init_control[..., 1].clamp(-MAX_STEER, MAX_STEER),
                ],
                dim=-1,
            )
            u_peds = torch.stack(
                [
                    u_peds[..., 0].clamp(-1.0, 1.0),
                    u_peds[..., 1].clamp(-math.pi / 4, math.pi / 4),
                ],
                dim=-1,
            )
            u_vehs = torch.stack(
                [
                    u_vehs[..., 0].clamp(-MAX_ACC, MAX_ACC),
                    u_vehs[..., 1].clamp(-MAX_STEER, MAX_STEER),
                ],
                dim=-1,
            )

            default_weights = build_default_cost_weights(
                B,
                device,
                torch.float32,
                weights=cfg_runtime.get("default_cost_weights"),
            )
            ego_rollout_traj, ped_traj, veh_traj = rollout_all_agents(
                linear_dynamics,
                ego_state,
                ped_states,
                veh_states,
                init_control,
                u_peds,
                u_vehs,
            )
            with torch.no_grad():
                ped_traj_safety = build_fixed_agent_safety_trajectories(
                    ped_states,
                    horizon=ego_rollout_traj.shape[1],
                )
                veh_traj_safety = build_fixed_agent_safety_trajectories(
                    veh_states,
                    horizon=ego_rollout_traj.shape[1],
                )
            reference_forward_offset = float(cfg_runtime.get("reference_forward_offset", 0.0))
            ego_metric_traj = apply_forward_offset_to_traj(
                ego_rollout_traj,
                -reference_forward_offset,
            )

            hipad_collision_gate = torch.zeros(B, device=device, dtype=ego_metric_traj.dtype)
            hipad_collision_violation = torch.zeros_like(hipad_collision_gate)
            pnn_collision_gate = torch.zeros_like(hipad_collision_gate)
            pnn_collision_violation = torch.zeros_like(hipad_collision_gate)
            pnn_collision_frame_gate = torch.zeros(
                (B, 6), device=device, dtype=ego_metric_traj.dtype
            )
            hipad_collision_frame_gate = torch.zeros_like(pnn_collision_frame_gate)
            use_official_frame_acr = bool(
                cfg_runtime.get("official_frame_acr_enabled", False)
            )
            if gt_actor_boxes_2hz is not None and gt_actor_mask_2hz is not None:
                with torch.no_grad():
                    if use_official_frame_acr:
                        pnn_collision_frame_gate = compute_official_actor_raster_proxy_gate(
                            ego_traj=ego_metric_traj,
                            gt_actor_boxes_2hz=gt_actor_boxes_2hz,
                            gt_actor_mask_2hz=gt_actor_mask_2hz,
                            sample_valid_mask=official_fut_valid_mask,
                            frame_valid_mask=gt_obj_frame_valid_mask,
                            actor_raster_padding=cfg_runtime.get(
                                "official_raster_actor_padding", 0.12
                            ),
                            actor_raster_dilation_pixels=cfg_runtime.get(
                                "official_raster_actor_dilation_pixels", 0.0
                            ),
                        )
                        pnn_collision_gate = pnn_collision_frame_gate.amax(dim=1)
                        pnn_collision_violation = pnn_collision_frame_gate.mean(dim=1)
                    else:
                        pnn_collision_violation = compute_gt_actor_box_violation(
                            ego_traj=ego_metric_traj,
                            gt_actor_boxes_2hz=gt_actor_boxes_2hz,
                            gt_actor_mask_2hz=gt_actor_mask_2hz,
                            rect_distance=safety_loss_module.rect_dist,
                            sample_valid_mask=official_fut_valid_mask,
                            frame_valid_mask=gt_obj_frame_valid_mask,
                            margin=cfg_runtime.get("risk_gate_margin", 0.0),
                            topk=cfg_runtime.get("metric_safety_topk", 1),
                        )
                        pnn_collision_gate = (pnn_collision_violation > 0.0).to(ego_metric_traj.dtype)
            if (
                hipad_plan_2hz is not None
                and gt_actor_boxes_2hz is not None
                and gt_actor_mask_2hz is not None
            ):
                with torch.no_grad():
                    if use_official_frame_acr:
                        hipad_collision_frame_gate = compute_official_actor_raster_proxy_gate(
                            ego_traj=hipad_plan_2hz,
                            gt_actor_boxes_2hz=gt_actor_boxes_2hz,
                            gt_actor_mask_2hz=gt_actor_mask_2hz,
                            sample_valid_mask=official_fut_valid_mask,
                            frame_valid_mask=gt_obj_frame_valid_mask,
                            actor_raster_padding=cfg_runtime.get(
                                "official_raster_actor_padding", 0.12
                            ),
                            actor_raster_dilation_pixels=cfg_runtime.get(
                                "official_raster_actor_dilation_pixels", 0.0
                            ),
                        )
                        hipad_collision_gate = hipad_collision_frame_gate.amax(dim=1)
                        hipad_collision_violation = hipad_collision_frame_gate.mean(dim=1)
                    else:
                        hipad_collision_violation = compute_gt_actor_box_violation(
                            ego_traj=hipad_plan_2hz,
                            gt_actor_boxes_2hz=gt_actor_boxes_2hz,
                            gt_actor_mask_2hz=gt_actor_mask_2hz,
                            rect_distance=safety_loss_module.rect_dist,
                            sample_valid_mask=official_fut_valid_mask,
                            frame_valid_mask=gt_obj_frame_valid_mask,
                            margin=cfg_runtime.get("weight_hipad_risk_margin", 0.0),
                            topk=cfg_runtime.get("metric_safety_topk", 1),
                        )
                        hipad_collision_gate = (hipad_collision_violation > 0.0).to(ego_metric_traj.dtype)

            teacher_metric_traj = None
            teacher_risk_gate = torch.zeros(B, device=device, dtype=ego_metric_traj.dtype)
            teacher_static_risk_gate = torch.zeros_like(teacher_risk_gate)
            teacher_static_frame_gate = torch.zeros_like(
                pnn_collision_frame_gate
            )
            teacher_collision_frame_gate = torch.zeros_like(pnn_collision_frame_gate)
            teacher_lane_collision_frame_gate = torch.zeros_like(
                pnn_collision_frame_gate
            )
            if teacher_policy is not None:
                with torch.no_grad():
                    teacher_u_ego_n, _, _ = teacher_policy(
                        ego_state_n,
                        ped_states_n,
                        veh_states_n,
                        lane_control_n,
                        build_model_padding_mask(ped_mask),
                        build_model_padding_mask(veh_mask),
                        static_states_n,
                        static_mask,
                    )
                    teacher_u_ego = inverse_normalize(teacher_u_ego_n, a_ego, b_ego)
                    teacher_u_ego = torch.stack(
                        [
                            teacher_u_ego[..., 0].clamp(-MAX_ACC, MAX_ACC),
                            teacher_u_ego[..., 1].clamp(-MAX_STEER, MAX_STEER),
                        ],
                        dim=-1,
                    )
                    teacher_rollout = bicycle_model_compatible(teacher_u_ego, ego_state)
                    teacher_metric_traj = apply_forward_offset_to_traj(
                        teacher_rollout,
                        -reference_forward_offset,
                    )
                    if gt_actor_boxes_2hz is not None and gt_actor_mask_2hz is not None:
                        if use_official_frame_acr:
                            teacher_collision_frame_gate = compute_official_actor_raster_proxy_gate(
                                ego_traj=teacher_metric_traj,
                                gt_actor_boxes_2hz=gt_actor_boxes_2hz,
                                gt_actor_mask_2hz=gt_actor_mask_2hz,
                                sample_valid_mask=official_fut_valid_mask,
                                frame_valid_mask=gt_obj_frame_valid_mask,
                                actor_raster_padding=cfg_runtime.get(
                                    "official_raster_actor_padding", 0.12
                                ),
                                actor_raster_dilation_pixels=cfg_runtime.get(
                                    "official_raster_actor_dilation_pixels", 0.0
                                ),
                            )
                            teacher_g_safety = teacher_collision_frame_gate.mean(dim=1)
                        else:
                            teacher_g_safety = compute_gt_actor_box_violation(
                                ego_traj=teacher_metric_traj,
                                gt_actor_boxes_2hz=gt_actor_boxes_2hz,
                                gt_actor_mask_2hz=gt_actor_mask_2hz,
                                rect_distance=safety_loss_module.rect_dist,
                                sample_valid_mask=official_fut_valid_mask,
                                frame_valid_mask=gt_obj_frame_valid_mask,
                                margin=cfg_runtime.get("metric_safety_margin", 0.0),
                                topk=cfg_runtime.get("metric_safety_topk", 1),
                            )
                    else:
                        teacher_g_safety = safety_loss_module.compute_constraint_violation(
                            teacher_metric_traj,
                            ped_traj_safety,
                            veh_traj_safety,
                            ped_mask,
                            veh_mask,
                        )
                    teacher_risk_gate = (teacher_g_safety > 0.0).to(ego_metric_traj.dtype)
                    teacher_static_frame_violation = (
                        compute_static_detection_box_violation(
                            ego_traj=teacher_metric_traj,
                            static_states=static_states,
                            static_mask=static_mask,
                            rect_distance=safety_loss_module.rect_dist,
                            margin=cfg_runtime.get(
                                "static_teacher_gate_margin", 0.0
                            ),
                            smooth_temperature=0.0,
                            max_distance=cfg_runtime.get(
                                "static_safety_max_distance", 55.0
                            ),
                            ego_z_min=cfg_runtime.get(
                                "static_ego_z_min", -1.90
                            ),
                            ego_z_max=cfg_runtime.get(
                                "static_ego_z_max", 0.80
                            ),
                            time_weights=cfg_runtime.get(
                                "static_safety_time_weights",
                                (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                            ),
                            return_per_frame=True,
                        )
                    )
                    teacher_static_frame_gate = (
                        teacher_static_frame_violation
                        > float(
                            cfg_runtime.get(
                                "static_teacher_gate_threshold", 1e-6
                            )
                        )
                    ).to(ego_metric_traj.dtype)
                    teacher_static_risk_gate = (
                        teacher_static_frame_gate.amax(dim=1)
                    )
            rollout_collision_risk = compute_rollout_collision_risk(
                ego_traj=ego_metric_traj,
                ped_states=ped_states,
                veh_states=veh_states,
                ped_mask=ped_mask,
                veh_mask=veh_mask,
                safe_dist=cfg_runtime.get("safe_dist", 10.0),
                collision_dist=cfg_runtime.get("collision_dist", SAFE_MARGIN),
                sharpness=cfg_runtime.get("collision_risk_sharpness", 1.5),
            )

            scene_vec = build_scene_feature_vector_from_batch(
                ego_state=ego_state,
                ped_states=ped_states,
                veh_states=veh_states,
                lane_points=lane_for_weight,
                ped_mask=ped_mask,
                veh_mask=veh_mask,
                safe_dist=cfg_runtime.get("safe_dist", 10.0),
            )
            prior_out = build_scene_adaptive_cost_prior(
                scene_vec=scene_vec,
                base_weights=default_weights,
                collision_risk=rollout_collision_risk.detach(),
                dense_gain=cfg_runtime.get("prior_dense_gain", 1.0),
                turn_gain=cfg_runtime.get("prior_turn_gain", 0.8),
                high_speed_gain=cfg_runtime.get("prior_high_speed_gain", 0.7),
                high_speed_threshold=cfg_runtime.get("prior_high_speed_threshold", 12.0),
                high_speed_sharpness=cfg_runtime.get("prior_high_speed_sharpness", 3.0),
            )
            scene_prior_weights = sanitize_planner_cost_weights(
                prior_out["scene_prior_weights"],
                min_weight=cfg_runtime.get("planner_weight_min_vector", cfg_runtime.get("planner_weight_min", 1e-3)),
                max_weight=cfg_runtime.get("planner_weight_max_vector", cfg_runtime.get("planner_weight_max", 20.0)),
                fallback_weights=default_weights,
                renormalize_to_fallback_sum=cfg_runtime.get("prior_renormalize_to_default_sum", False),
            )

            weight_out = compute_direct_weightnet_prob(
                weight_model=weight_model,
                # WeightNet encoders are initialized from ControlNet and must
                # see the same robust-normalized feature distribution.
                ego_state=ego_state_n,
                ped_states=ped_states_n,
                veh_states=veh_states_n,
                lane_for_weight=lane_weight_n,
                ped_mask=ped_mask,
                veh_mask=veh_mask,
                safe_dist=cfg_runtime.get("safe_dist", 10.0),
                scene_prior_weights=scene_prior_weights,
                scene_vec=scene_vec,
            )
            weight_delta_raw = weight_out["weight_delta_raw"]
            refine_gate = weight_out["refine_gate"]

            weight_residual_out = compute_logspace_residual_cost_weights(
                residual_raw=weight_delta_raw,
                scene_prior_weights=scene_prior_weights,
                min_weight=cfg_runtime.get("planner_weight_min_vector", cfg_runtime.get("planner_weight_min", 1e-3)),
                max_weight=cfg_runtime.get("planner_weight_max_vector", cfg_runtime.get("planner_weight_max", 20.0)),
                delta_max=cfg_runtime.get("weight_delta_max", 0.7),
                renormalize_to_fallback_sum=cfg_runtime.get("planner_weight_renormalize_to_default_sum", False),
            )
            cost_weights = weight_residual_out["cost_weights"]
            pred_weights_prob = weight_residual_out["pred_weights_prob"]
            weight_delta = weight_residual_out["weight_delta"]

            min_dist = compute_min_agent_distance(
                ego_state=ego_state,
                ped_states=ped_states,
                veh_states=veh_states,
                ped_mask=ped_mask,
                veh_mask=veh_mask,
            )
            risk_score = distance_to_risk_score(
                min_dist,
                safe_dist=cfg_runtime.get("safe_dist", 10.0),
            ).squeeze(1)

            weight_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_loss_traj = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_rule_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_feedback_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_rank_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_sep_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_entropy_band_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_diversity_floor_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_extreme_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_hipad_risk_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_pnn_collision_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_dipp_safety_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_dipp_lane_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            weight_dipp_trust_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
            entropy = torch.zeros((), device=device, dtype=ego_state.dtype)
            diversity = torch.zeros((), device=device, dtype=ego_state.dtype)
            pairwise_l2 = torch.zeros((), device=device, dtype=ego_state.dtype)
            kl_to_prior = torch.zeros((), device=device, dtype=ego_state.dtype)
            lambda_weight_traj = 0.0
            lambda_weight_rule = 0.0
            lambda_weight_feedback = 0.0
            lambda_weight_rank = 0.0
            lambda_weight_sep = 0.0
            lambda_weight_extreme = 0.0
            lambda_entropy_band = 0.0
            lambda_diversity_floor = 0.0
            supervision_ramp = 0.0

            # Keep policy optimization detached from WeightNet. Directly minimizing
            # weighted_ego_loss w.r.t. weights lets the weight net "cheat" by
            # lowering difficult terms instead of improving the policy; WeightNet
            # receives control-branch feedback through compute_feedback_target_weights.
            with torch.no_grad():
                if epoch < cfg_runtime.get("control_weight_start_epoch", 0):
                    cost_weights_detached = default_weights.detach()
                else:
                    cost_weights_detached = cost_weights.detach()

            weighted_ego_comps = compute_weighted_ego_cost_components(
                init_control,
                ego_metric_traj,
                ego_state,
                lane_for_control,
                ped_traj_safety,
                veh_traj_safety,
                ped_mask,
                veh_mask,
                gt_reference_line=gt_reference_line,
                gt_reference_line_valid_mask=gt_reference_line_valid_mask,
            )

            weighted_ego_loss = sum(
                cost_weights_detached[:, i] * weighted_ego_comps[name]
                for i, name in enumerate(COST_NAMES)
            )
            gt_reference_lane_loss = (
                cost_weights_detached[:, COST_NAMES.index("lane_xy")] * weighted_ego_comps["lane_xy"]
                + cost_weights_detached[:, COST_NAMES.index("lane_theta")] * weighted_ego_comps["lane_theta"]
            ).mean()

            lane_loss_module.log_lambda.requires_grad_(False)
            safety_loss_module.log_lambda.requires_grad_(False)

            pnn_only_risk_gate = pnn_collision_gate * (1.0 - hipad_collision_gate)
            shared_risk_gate = pnn_collision_gate * hipad_collision_gate
            if cfg_runtime.get("use_pnn_only_risk_weighting", False):
                safety_sample_weight = float(cfg_runtime.get("lambda_ego_object_safety", 1.0)) * (
                    1.0
                    + float(cfg_runtime.get("pnn_only_safety_gain", 4.0)) * pnn_only_risk_gate
                    + float(cfg_runtime.get("shared_safety_gain", 2.0)) * shared_risk_gate
                )
            else:
                safety_sample_weight = float(cfg_runtime.get("lambda_ego_object_safety", 1.0)) * (
                    1.0
                    + float(cfg_runtime.get("risk_safety_gain", 0.0)) * teacher_risk_gate
                )
            residual_loss, residual_aux = compute_original_control_loss_with_adaptive_weights(
                init_control,
                u_peds,
                u_vehs,
                ego_metric_traj,
                ped_traj,
                veh_traj,
                ego_state,
                ped_states,
                veh_states,
                lane_points,
                ped_mask,
                veh_mask,
                device,
                lane_loss_module,
                safety_loss_module,
                soft_lambda_module,
                cost_weights_detached,
                default_weights,
                ped_traj_safety=ped_traj_safety,
                veh_traj_safety=veh_traj_safety,
                safety_loss_weight=safety_sample_weight,
                gt_actor_boxes_2hz=gt_actor_boxes_2hz,
                gt_actor_mask_2hz=gt_actor_mask_2hz,
                gt_obj_frame_valid_mask=gt_obj_frame_valid_mask,
                official_fut_valid_mask=official_fut_valid_mask,
                metric_safety_margin=cfg_runtime.get("metric_safety_margin", 0.0),
                metric_safety_topk=cfg_runtime.get("metric_safety_topk", 1),
                metric_safety_smooth_temperature=cfg_runtime.get(
                    "metric_safety_smooth_temperature", 0.0
                ),
                metric_safety_time_weights=cfg_runtime.get(
                    "metric_safety_time_weights", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                route_speed_loss_weight=cfg_runtime.get("lambda_route_speed_excess", 0.0),
                route_speed_margin=cfg_runtime.get("route_speed_margin", 1.0),
                route_speed_brake_weight=cfg_runtime.get("route_speed_brake_weight", 0.2),
                route_speed_brake_trigger_margin=cfg_runtime.get("route_speed_brake_trigger_margin", 1.0),
                route_speed_positive_accel_threshold=cfg_runtime.get("route_speed_positive_accel_threshold", 0.3),
                ego_track_time_weights=cfg_runtime.get(
                    "ego_track_time_weights",
                    (1.8, 2.5, 1.8, 1.2, 0.8, 0.5),
                ),
                lane_clearance_loss_weight=cfg_runtime.get("lane_clearance_loss_weight", 0.0),
                lane_clearance_margin=cfg_runtime.get("lane_clearance_margin", 0.8),
                gt_solid_lane_points=gt_solid_lane_points,
                gt_solid_lane_mask=gt_solid_lane_mask,
                gt_lane_frame_valid_mask=gt_lane_frame_valid_mask,
                gt_reference_line=gt_reference_line,
                metric_lane_loss_weight=cfg_runtime.get("lambda_metric_lane", 0.0),
                metric_lane_margin=cfg_runtime.get("metric_lane_margin", 0.05),
                metric_lane_use_gt_safe_side=cfg_runtime.get(
                    "metric_lane_use_gt_safe_side", False
                ),
                metric_lane_time_weights=cfg_runtime.get(
                    "metric_lane_time_weights", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                metric_lane_hard_max_weight=cfg_runtime.get(
                    "metric_lane_hard_max_weight", 0.0
                ),
                disable_predicted_lane_losses=cfg_runtime.get("disable_predicted_lane_losses", False),
                dense_speed_loss_weight=cfg_runtime.get("lambda_dense_route_speed", 0.0),
                dense_speed_margin=cfg_runtime.get("dense_route_speed_margin", 0.5),
                forward_overshoot_loss_weight=cfg_runtime.get("lambda_forward_overshoot", 0.0),
                forward_overshoot_margin=cfg_runtime.get("forward_overshoot_margin", 0.3),
                forward_overshoot_time_weights=cfg_runtime.get(
                    "forward_overshoot_time_weights",
                    (2.0, 1.5, 0.8),
                ),
                obstacle_clearance_loss_weight=cfg_runtime.get("lambda_obstacle_clearance", 0.0),
                obstacle_clearance_margin_veh=cfg_runtime.get("obstacle_clearance_margin_veh", 2.5),
                obstacle_clearance_margin_ped=cfg_runtime.get("obstacle_clearance_margin_ped", 1.8),
                obstacle_clearance_time_weights=cfg_runtime.get(
                    "obstacle_clearance_time_weights",
                    (1.5, 1.2, 1.0),
                ),
                obstacle_clearance_topk=cfg_runtime.get("obstacle_clearance_topk", 3),
                comfort_loss_weight=cfg_runtime.get("lambda_rollout_comfort", 0.0),
                comfort_acc_threshold=cfg_runtime.get("comfort_acc_threshold", 2.40),
                comfort_min_lon_accel=cfg_runtime.get("comfort_min_lon_accel", -4.05),
                comfort_lat_accel_threshold=cfg_runtime.get("comfort_lat_accel_threshold", 4.89),
                comfort_jerk_threshold=cfg_runtime.get("comfort_jerk_threshold", 4.13),
                comfort_yaw_rate_threshold=cfg_runtime.get("comfort_yaw_rate_threshold", 0.95),
                comfort_yaw_accel_threshold=cfg_runtime.get("comfort_yaw_accel_threshold", 1.93),
            )

            hipad_static_frame_gate = torch.zeros_like(
                teacher_static_frame_gate
            )
            if hipad_plan_2hz is not None:
                with torch.no_grad():
                    hipad_static_hard_violation = (
                        compute_static_detection_box_violation(
                            ego_traj=hipad_plan_2hz,
                            static_states=static_states,
                            static_mask=static_mask,
                            rect_distance=safety_loss_module.rect_dist,
                            margin=0.0,
                            smooth_temperature=0.0,
                            max_distance=cfg_runtime.get(
                                "static_safety_max_distance", 55.0
                            ),
                            ego_z_min=cfg_runtime.get(
                                "static_ego_z_min", -1.90
                            ),
                            ego_z_max=cfg_runtime.get(
                                "static_ego_z_max", 0.80
                            ),
                            time_weights=cfg_runtime.get(
                                "static_safety_time_weights",
                                (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                            ),
                            return_per_frame=True,
                        )
                    )
                    hipad_static_frame_gate = (
                        hipad_static_hard_violation
                        > float(
                            cfg_runtime.get(
                                "static_target_gate_threshold", 1e-6
                            )
                        )
                    ).to(ego_metric_traj.dtype)
            hipad_static_risk_gate = hipad_static_frame_gate.amax(dim=1)
            static_parent_pnn_only_gate = (
                teacher_static_risk_gate
                * (1.0 - hipad_static_risk_gate)
            ).detach()

            static_risk_gate_loss = torch.zeros(
                B, device=device, dtype=ego_metric_traj.dtype
            )
            static_gate_prediction = policy_core.module.last_static_gate
            static_gate_weight = float(
                cfg_runtime.get("lambda_static_risk_gate", 0.0)
            )
            if (
                static_gate_weight > 0.0
                and static_gate_prediction is not None
            ):
                gate_target_count = static_parent_pnn_only_gate.sum()
                gate_positive_scale = (
                    static_parent_pnn_only_gate.new_tensor(float(B))
                    / gate_target_count.clamp_min(1.0)
                ).clamp_max(
                    float(
                        cfg_runtime.get(
                            "static_risk_gate_positive_max_scale", 128.0
                        )
                    )
                )
                gate_prediction = static_gate_prediction.clamp(
                    min=1e-5, max=1.0 - 1e-5
                )
                gate_bce = -(
                    static_parent_pnn_only_gate
                    * gate_positive_scale
                    * torch.log(gate_prediction)
                    + (1.0 - static_parent_pnn_only_gate)
                    * torch.log1p(-gate_prediction)
                )
                static_risk_gate_loss = static_gate_weight * gate_bce
                residual_loss = residual_loss + static_risk_gate_loss
            residual_aux["loss_static_risk_gate"] = static_risk_gate_loss

            static_box_violation = compute_static_detection_box_violation(
                ego_traj=ego_metric_traj,
                static_states=static_states,
                static_mask=static_mask,
                rect_distance=safety_loss_module.rect_dist,
                margin=cfg_runtime.get("static_safety_margin", 0.60),
                smooth_temperature=cfg_runtime.get(
                    "static_safety_temperature", 0.20
                ),
                max_distance=cfg_runtime.get("static_safety_max_distance", 55.0),
                ego_z_min=cfg_runtime.get("static_ego_z_min", -1.90),
                ego_z_max=cfg_runtime.get("static_ego_z_max", 0.80),
                time_weights=cfg_runtime.get(
                    "static_safety_time_weights",
                    (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                ),
            )
            static_positive_gate = torch.ones_like(static_box_violation)
            static_positive_scale = torch.ones(
                (), device=device, dtype=static_box_violation.dtype
            )
            if cfg_runtime.get("static_positive_normalize", False):
                with torch.no_grad():
                    static_hard_violation = (
                        compute_static_detection_box_violation(
                            ego_traj=ego_metric_traj,
                            static_states=static_states,
                            static_mask=static_mask,
                            rect_distance=safety_loss_module.rect_dist,
                            margin=cfg_runtime.get(
                                "static_positive_margin",
                                cfg_runtime.get("static_safety_margin", 0.60),
                            ),
                            smooth_temperature=0.0,
                            max_distance=cfg_runtime.get(
                                "static_safety_max_distance", 55.0
                            ),
                            ego_z_min=cfg_runtime.get(
                                "static_ego_z_min", -1.90
                            ),
                            ego_z_max=cfg_runtime.get(
                                "static_ego_z_max", 0.80
                            ),
                            time_weights=cfg_runtime.get(
                                "static_safety_time_weights",
                                (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                            ),
                        )
                    )
                    static_positive_gate = (
                        static_hard_violation
                        > float(
                            cfg_runtime.get(
                                "static_positive_threshold", 1e-6
                            )
                        )
                    ).to(static_box_violation.dtype)
                    positive_count = static_positive_gate.sum()
                    static_positive_scale = (
                        static_box_violation.new_tensor(float(B))
                        / positive_count.clamp_min(1.0)
                    ).clamp_max(
                        float(
                            cfg_runtime.get(
                                "static_positive_max_scale", 16.0
                            )
                        )
                    )
                static_box_violation = (
                    static_box_violation
                    * static_positive_gate
                    * static_positive_scale
                )
            static_box_safety_loss = (
                float(cfg_runtime.get("lambda_static_box_safety", 0.0))
                * static_box_violation
            )
            residual_loss = residual_loss + static_box_safety_loss
            residual_aux["loss_static_box_safety"] = static_box_safety_loss

            static_target_scale = torch.ones(
                (), device=device, dtype=ego_metric_traj.dtype
            )
            static_any_hit_loss = torch.zeros(
                B, device=device, dtype=ego_metric_traj.dtype
            )
            static_any_hit_weight = float(
                cfg_runtime.get("lambda_static_any_hit", 0.0)
            )
            if static_any_hit_weight > 0.0:
                static_target_count = static_parent_pnn_only_gate.sum()
                static_target_scale = (
                    static_any_hit_loss.new_tensor(float(B))
                    / static_target_count.clamp_min(1.0)
                ).clamp_max(
                    float(
                        cfg_runtime.get(
                            "static_target_max_scale", 32.0
                        )
                    )
                )
                static_current_frame_violation = (
                    compute_static_detection_box_violation(
                        ego_traj=ego_metric_traj,
                        static_states=static_states,
                        static_mask=static_mask,
                        rect_distance=safety_loss_module.rect_dist,
                        margin=cfg_runtime.get(
                            "static_any_hit_margin", 0.60
                        ),
                        smooth_temperature=cfg_runtime.get(
                            "static_any_hit_temperature", 0.15
                        ),
                        max_distance=cfg_runtime.get(
                            "static_safety_max_distance", 55.0
                        ),
                        ego_z_min=cfg_runtime.get(
                            "static_ego_z_min", -1.90
                        ),
                        ego_z_max=cfg_runtime.get(
                            "static_ego_z_max", 0.80
                        ),
                        time_weights=cfg_runtime.get(
                            "static_safety_time_weights",
                            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                        ),
                        return_per_frame=True,
                    )
                )
                # A single colliding official frame fails the whole sample.
                # Max aggregation therefore matches the deployment objective
                # better than averaging collision duration over six frames.
                static_any_hit_violation = (
                    static_current_frame_violation.amax(dim=1)
                )
                static_any_hit_loss = (
                    static_any_hit_weight
                    * static_target_scale
                    * static_parent_pnn_only_gate
                    * static_any_hit_violation
                )
                residual_loss = residual_loss + static_any_hit_loss
            residual_aux["loss_static_any_hit"] = static_any_hit_loss

            static_hipad_anchor_loss = torch.zeros(
                B, device=device, dtype=ego_metric_traj.dtype
            )
            static_anchor_weight = float(
                cfg_runtime.get("lambda_static_hipad_anchor", 0.0)
            )
            if static_anchor_weight > 0.0 and hipad_plan_2hz is not None:
                frame_indices = [
                    min(i, ego_metric_traj.shape[1] - 1)
                    for i in (4, 9, 14, 19, 24, 29)
                ]
                pnn_2hz = ego_metric_traj[:, frame_indices, :2]
                anchor_steps = min(
                    pnn_2hz.shape[1], hipad_plan_2hz.shape[1], 6
                )
                anchor_time_weights = torch.as_tensor(
                    cfg_runtime.get(
                        "static_hipad_anchor_time_weights",
                        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                    ),
                    device=device,
                    dtype=ego_metric_traj.dtype,
                )[:anchor_steps]
                static_anchor_error = F.smooth_l1_loss(
                    pnn_2hz[:, :anchor_steps],
                    hipad_plan_2hz[:, :anchor_steps, :2],
                    reduction="none",
                    beta=float(
                        cfg_runtime.get(
                            "static_hipad_anchor_beta", 0.25
                        )
                    ),
                ).mean(dim=-1)
                static_anchor_error = (
                    static_anchor_error
                    * anchor_time_weights.view(1, -1)
                ).sum(dim=1) / anchor_time_weights.sum().clamp_min(1e-6)
                static_hipad_anchor_loss = (
                    static_anchor_weight
                    * static_target_scale
                    * static_parent_pnn_only_gate
                    * static_anchor_error
                )
                residual_loss = residual_loss + static_hipad_anchor_loss
            residual_aux[
                "loss_static_hipad_anchor"
            ] = static_hipad_anchor_loss

            teacher_trust_loss = torch.zeros(B, device=device, dtype=ego_metric_traj.dtype)
            if teacher_metric_traj is not None:
                teacher_xy_error = F.smooth_l1_loss(
                    ego_metric_traj[:, :, :2],
                    teacher_metric_traj[:, :, :2],
                    reduction="none",
                    beta=cfg_runtime.get("teacher_trust_beta", 0.25),
                ).mean(dim=(1, 2))
                trust_floor = float(cfg_runtime.get("teacher_trust_risk_floor", 0.1))
                if cfg_runtime.get(
                    "teacher_trust_uses_static_pnn_only", False
                ):
                    trust_risk_gate = static_parent_pnn_only_gate
                elif cfg_runtime.get("teacher_trust_uses_static_risk", False):
                    trust_risk_gate = teacher_static_risk_gate
                elif cfg_runtime.get("teacher_trust_uses_pnn_risk", False):
                    trust_risk_gate = pnn_collision_gate
                else:
                    trust_risk_gate = teacher_risk_gate
                trust_sample_weight = 1.0 - (1.0 - trust_floor) * trust_risk_gate
                teacher_trust_loss = (
                    float(cfg_runtime.get("lambda_teacher_trust", 0.0))
                    * trust_sample_weight
                    * teacher_xy_error
                )
                residual_loss = residual_loss + teacher_trust_loss
                residual_aux["loss_teacher_trust"] = teacher_trust_loss

            # In samples where PNN collides but the original HiPAD plan is safe,
            # HiPAD provides an observed safe direction. This conditional anchor
            # removes the left/right ambiguity of a symmetric actor-distance loss
            # without imitating HiPAD on already-safe PNN samples.
            pnn_only_hipad_anchor_loss = torch.zeros(
                B, device=device, dtype=ego_metric_traj.dtype
            )
            anchor_weight = float(cfg_runtime.get("lambda_pnn_only_hipad_anchor", 0.0))
            if anchor_weight > 0.0 and hipad_plan_2hz is not None:
                frame_indices = [min(i, ego_metric_traj.shape[1] - 1) for i in (4, 9, 14, 19, 24, 29)]
                pnn_2hz = ego_metric_traj[:, frame_indices, :2]
                anchor_steps = min(pnn_2hz.shape[1], hipad_plan_2hz.shape[1], 6)
                time_weights = torch.as_tensor(
                    cfg_runtime.get(
                        "pnn_only_hipad_anchor_time_weights",
                        (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                    ),
                    device=device,
                    dtype=ego_metric_traj.dtype,
                )[:anchor_steps]
                point_error = F.smooth_l1_loss(
                    pnn_2hz[:, :anchor_steps],
                    hipad_plan_2hz[:, :anchor_steps, :2],
                    reduction="none",
                    beta=float(cfg_runtime.get("pnn_only_hipad_anchor_beta", 0.25)),
                ).mean(dim=-1)
                weighted_error = (
                    point_error * time_weights[None]
                ).sum(dim=1) / time_weights.sum().clamp_min(1e-6)
                anchor_gate = pnn_only_risk_gate * official_fut_valid_mask.to(
                    dtype=ego_metric_traj.dtype
                )
                pnn_only_hipad_anchor_loss = anchor_weight * anchor_gate * weighted_error
                residual_loss = residual_loss + pnn_only_hipad_anchor_loss
            residual_aux["loss_pnn_only_hipad_anchor"] = pnn_only_hipad_anchor_loss

            official_frame_acr_loss = torch.zeros(
                B, device=device, dtype=ego_metric_traj.dtype
            )
            official_frame_anchor_loss = torch.zeros_like(official_frame_acr_loss)
            official_frame_lane_guard_loss = torch.zeros_like(
                official_frame_acr_loss
            )
            safe_parent_distill_loss = torch.zeros_like(official_frame_acr_loss)
            hipad_lane_collision_frame_gate = torch.zeros_like(
                pnn_collision_frame_gate
            )
            if (
                use_official_frame_acr
                and gt_actor_boxes_2hz is not None
                and gt_actor_mask_2hz is not None
            ):
                frame_safety = compute_gt_actor_box_violation(
                    ego_traj=ego_metric_traj,
                    gt_actor_boxes_2hz=gt_actor_boxes_2hz,
                    gt_actor_mask_2hz=gt_actor_mask_2hz,
                    rect_distance=safety_loss_module.rect_dist,
                    sample_valid_mask=official_fut_valid_mask,
                    frame_valid_mask=gt_obj_frame_valid_mask,
                    margin=cfg_runtime.get("official_frame_acr_margin", 1.0),
                    topk=cfg_runtime.get("metric_safety_topk", 1),
                    smooth_temperature=cfg_runtime.get(
                        "official_frame_acr_temperature", 0.25
                    ),
                    return_per_frame=True,
                )
                frame_steps = min(
                    frame_safety.shape[1], pnn_collision_frame_gate.shape[1]
                )
                frame_mask = pnn_collision_frame_gate[:, :frame_steps]
                if cfg_runtime.get("official_frame_acr_pnn_only", False):
                    if hipad_plan_2hz is None:
                        frame_mask = torch.zeros_like(frame_mask)
                    else:
                        frame_mask = frame_mask * (
                            1.0
                            - hipad_collision_frame_gate[:, :frame_steps]
                        )
                frame_signal = (
                    cost_weights_detached[:, COST_NAMES.index("safe")].unsqueeze(1)
                    / default_weights[:, COST_NAMES.index("safe")].clamp_min(1e-3).unsqueeze(1)
                    * safety_loss_module.lambda_val.detach()
                    * frame_safety[:, :frame_steps]
                )
                official_frame_acr_loss = float(
                    cfg_runtime.get("lambda_official_frame_acr", 0.0)
                ) * positive_normalized_frame_loss_per_sample(
                    frame_signal,
                    frame_mask,
                    time_weights=cfg_runtime.get(
                        "official_frame_acr_time_weights",
                        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                    ),
                )
                residual_loss = residual_loss + official_frame_acr_loss

                pnn_2hz = _sample_trajectory_at_official_2hz(ego_metric_traj)
                lane_guard_weight = float(
                    cfg_runtime.get("lambda_official_frame_lane_guard", 0.0)
                )
                if (
                    lane_guard_weight > 0.0
                    and gt_solid_lane_points is not None
                    and gt_solid_lane_mask is not None
                ):
                    pnn_lane_frame_violation = compute_metric_solid_lane_violation(
                        ego_traj=ego_metric_traj,
                        solid_lane_points=gt_solid_lane_points,
                        solid_lane_mask=gt_solid_lane_mask,
                        frame_valid_mask=gt_lane_frame_valid_mask,
                        gt_reference_line=gt_reference_line,
                        use_gt_safe_side=True,
                        margin=cfg_runtime.get(
                            "official_frame_lane_guard_margin", 0.35
                        ),
                        return_per_frame=True,
                    )
                    guard_steps = min(
                        pnn_lane_frame_violation.shape[1],
                        pnn_collision_frame_gate.shape[1],
                    )
                    lane_guard_frame_gate = pnn_collision_frame_gate[
                        :, :guard_steps
                    ]
                    if cfg_runtime.get(
                        "official_frame_lane_guard_include_previous", True
                    ):
                        previous = torch.zeros_like(lane_guard_frame_gate)
                        previous[:, :-1] = lane_guard_frame_gate[:, 1:]
                        lane_guard_frame_gate = torch.maximum(
                            lane_guard_frame_gate, previous
                        )
                    if cfg_runtime.get(
                        "official_frame_lane_guard_include_future", False
                    ):
                        # Steering away from an actor can remain lane-safe at
                        # the collision frame but cross a solid line later.
                        lane_guard_frame_gate = (
                            lane_guard_frame_gate.cumsum(dim=1).clamp_max(1.0)
                        )
                    official_frame_lane_guard_loss = (
                        lane_guard_weight
                        * positive_normalized_frame_loss_per_sample(
                            pnn_lane_frame_violation[:, :guard_steps],
                            lane_guard_frame_gate,
                            time_weights=cfg_runtime.get(
                                "official_frame_lane_guard_time_weights",
                                (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                            ),
                        )
                    )
                    residual_loss = residual_loss + official_frame_lane_guard_loss

                if hipad_plan_2hz is not None:
                    anchor_steps = min(
                        pnn_2hz.shape[1],
                        hipad_plan_2hz.shape[1],
                        pnn_collision_frame_gate.shape[1],
                        hipad_collision_frame_gate.shape[1],
                    )
                    pnn_only_frame_gate = (
                        pnn_collision_frame_gate[:, :anchor_steps]
                        * (1.0 - hipad_collision_frame_gate[:, :anchor_steps])
                    )
                    require_hipad_lane_safe = bool(
                        cfg_runtime.get(
                            "official_frame_anchor_require_lane_safe", False
                        )
                    )
                    if require_hipad_lane_safe:
                        if (
                            gt_solid_lane_points is not None
                            and gt_solid_lane_mask is not None
                        ):
                            with torch.no_grad():
                                hipad_lane_frame_violation = (
                                    compute_metric_solid_lane_violation(
                                        ego_traj=hipad_plan_2hz,
                                        solid_lane_points=gt_solid_lane_points,
                                        solid_lane_mask=gt_solid_lane_mask,
                                        frame_valid_mask=gt_lane_frame_valid_mask,
                                        gt_reference_line=gt_reference_line,
                                        use_gt_safe_side=True,
                                        margin=cfg_runtime.get(
                                            "official_frame_anchor_lane_safe_margin",
                                            0.0,
                                        ),
                                        return_per_frame=True,
                                    )
                                )
                            lane_steps = min(
                                anchor_steps,
                                hipad_lane_frame_violation.shape[1],
                            )
                            hipad_lane_collision_frame_gate[
                                :, :lane_steps
                            ] = (
                                hipad_lane_frame_violation[:, :lane_steps] > 0.0
                            ).to(ego_metric_traj.dtype)
                            anchor_lane_safe = (
                                1.0
                                - hipad_lane_collision_frame_gate[
                                    :, :anchor_steps
                                ]
                            )
                        else:
                            anchor_lane_safe = torch.zeros_like(
                                pnn_only_frame_gate
                            )
                        pnn_only_frame_gate = (
                            pnn_only_frame_gate * anchor_lane_safe
                        )
                    if cfg_runtime.get("official_frame_anchor_include_previous", True):
                        previous = torch.zeros_like(pnn_only_frame_gate)
                        previous[:, :-1] = pnn_only_frame_gate[:, 1:]
                        previous = previous * (
                            1.0 - hipad_collision_frame_gate[:, :anchor_steps]
                        )
                        if require_hipad_lane_safe:
                            previous = previous * anchor_lane_safe
                        pnn_only_frame_gate = torch.maximum(
                            pnn_only_frame_gate, previous
                        )
                    anchor_error = F.smooth_l1_loss(
                        pnn_2hz[:, :anchor_steps],
                        hipad_plan_2hz[:, :anchor_steps, :2],
                        reduction="none",
                        beta=float(
                            cfg_runtime.get("official_frame_anchor_beta", 0.25)
                        ),
                    ).mean(dim=-1)
                    official_frame_anchor_loss = float(
                        cfg_runtime.get("lambda_official_frame_anchor", 0.0)
                    ) * positive_normalized_frame_loss_per_sample(
                        anchor_error,
                        pnn_only_frame_gate,
                        time_weights=cfg_runtime.get(
                            "official_frame_anchor_time_weights",
                            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                        ),
                    )
                    residual_loss = residual_loss + official_frame_anchor_loss

                if teacher_metric_traj is not None:
                    teacher_2hz = _sample_trajectory_at_official_2hz(
                        teacher_metric_traj
                    )
                    distill_steps = min(
                        pnn_2hz.shape[1],
                        teacher_2hz.shape[1],
                        pnn_collision_frame_gate.shape[1],
                        teacher_collision_frame_gate.shape[1],
                    )
                    parent_safe = (
                        1.0
                        - teacher_collision_frame_gate[:, :distill_steps]
                    )
                    if cfg_runtime.get(
                        "safe_parent_distill_require_parent_lane_safe", False
                    ):
                        if (
                            gt_solid_lane_points is not None
                            and gt_solid_lane_mask is not None
                        ):
                            with torch.no_grad():
                                teacher_lane_frame_violation = (
                                    compute_metric_solid_lane_violation(
                                        ego_traj=teacher_metric_traj,
                                        solid_lane_points=gt_solid_lane_points,
                                        solid_lane_mask=gt_solid_lane_mask,
                                        frame_valid_mask=gt_lane_frame_valid_mask,
                                        gt_reference_line=gt_reference_line,
                                        use_gt_safe_side=True,
                                        margin=cfg_runtime.get(
                                            "safe_parent_distill_lane_safe_margin",
                                            0.0,
                                        ),
                                        return_per_frame=True,
                                    )
                                )
                            lane_steps = min(
                                distill_steps,
                                teacher_lane_frame_violation.shape[1],
                            )
                            teacher_lane_collision_frame_gate[
                                :, :lane_steps
                            ] = (
                                teacher_lane_frame_violation[:, :lane_steps]
                                > 0.0
                            ).to(ego_metric_traj.dtype)
                            parent_safe = parent_safe * (
                                1.0
                                - teacher_lane_collision_frame_gate[
                                    :, :distill_steps
                                ]
                            )
                        else:
                            parent_safe = torch.zeros_like(parent_safe)
                    if cfg_runtime.get(
                        "safe_parent_distill_require_current_actor_safe", True
                    ):
                        parent_safe = parent_safe * (
                            1.0
                            - pnn_collision_frame_gate[:, :distill_steps]
                        )
                    safe_error = F.smooth_l1_loss(
                        pnn_2hz[:, :distill_steps],
                        teacher_2hz[:, :distill_steps],
                        reduction="none",
                        beta=float(
                            cfg_runtime.get("safe_parent_distill_beta", 0.25)
                        ),
                    ).mean(dim=-1)
                    safe_count = parent_safe.sum(dim=1).clamp_min(1.0)
                    safe_parent_distill_loss = float(
                        cfg_runtime.get("lambda_safe_parent_distill", 0.0)
                    ) * (safe_error * parent_safe).sum(dim=1) / safe_count
                    residual_loss = residual_loss + safe_parent_distill_loss

            residual_aux["loss_official_frame_acr"] = official_frame_acr_loss
            residual_aux["loss_official_frame_anchor"] = official_frame_anchor_loss
            residual_aux[
                "loss_official_frame_lane_guard"
            ] = official_frame_lane_guard_loss
            residual_aux["loss_safe_parent_distill"] = safe_parent_distill_loss

            g_lane, g_safety = residual_aux["g_lane"], residual_aux["g_safety"]
            # WeightNet supervision follows current PNN failures. Object gates
            # use the corrected actor boxes; lane gates use the metric-aligned
            # full-footprint GT-solid surrogate.
            weight_obj_collision_gate = pnn_collision_gate.detach()
            weight_lane_collision_gate = (
                residual_aux["g_metric_lane"].detach()
                > float(cfg_runtime.get("weight_pnn_lane_gate_threshold", 0.0))
            ).to(dtype=ego_metric_traj.dtype)
            weight_any_collision_gate = torch.maximum(
                weight_obj_collision_gate,
                weight_lane_collision_gate,
            )
            aug_term = 0.5 * rho_aug * ((g_lane ** 2).mean() + (g_safety ** 2).mean())
            train_proxy_metrics = {}
            if cfg_runtime.get("train_proxy_metrics", True):
                train_proxy_metrics, _train_proxy_valid_count = compute_training_planning_proxy_metrics(
                    ego_traj=ego_metric_traj,
                    ego_state=ego_state,
                    ego_future_gt_new=ego_future_gt_new,
                    ego_future_gt_valid_mask=ego_future_gt_valid_mask,
                    ped_states=ped_states,
                    veh_states=veh_states,
                    ped_mask=ped_mask,
                    veh_mask=veh_mask,
                    control=init_control,
                    g_lane=g_lane,
                    g_safety=g_safety,
                    safety_loss_module=safety_loss_module,
                    comfort_max_lon_accel=cfg_runtime.get("comfort_acc_threshold", 2.40),
                    comfort_min_lon_accel=cfg_runtime.get("comfort_min_lon_accel", -4.05),
                    comfort_max_lat_accel=cfg_runtime.get("comfort_lat_accel_threshold", 4.89),
                    comfort_jerk_threshold=cfg_runtime.get("comfort_jerk_threshold", 4.13),
                    comfort_yaw_rate_threshold=cfg_runtime.get("comfort_yaw_rate_threshold", 0.95),
                    comfort_yaw_accel_threshold=cfg_runtime.get("comfort_yaw_accel_threshold", 1.93),
                )

            if update_weight:
                traj_ramp = linear_ramp(
                    epoch,
                    cfg_runtime.get("weight_traj_warmup_epochs", 2),
                    cfg_runtime.get("weight_traj_ramp_epochs", 1),
                )
                lambda_weight_traj = cfg_runtime.get("lambda_weight_traj", 0.4) * traj_ramp
                lambda_weight_dipp_safety = float(cfg_runtime.get("lambda_weight_dipp_safety", 0.0))
                lambda_weight_dipp_lane = float(cfg_runtime.get("lambda_weight_dipp_lane", 0.0))
                lambda_weight_dipp_trust = float(cfg_runtime.get("lambda_weight_dipp_trust", 0.0))
                supervision_start = cfg_runtime.get(
                    "weight_supervision_start_epoch",
                    cfg_runtime.get("weight_update_start_epoch", 0),
                )
                supervision_ramp = linear_ramp(
                    epoch,
                    supervision_start,
                    cfg_runtime.get("weight_supervision_ramp_epochs", 1),
                )

                dipp_failed_local = False
                need_dipp_weight_rollout = (
                    lambda_weight_traj > 0.0
                    or lambda_weight_dipp_safety > 0.0
                    or lambda_weight_dipp_lane > 0.0
                    or lambda_weight_dipp_trust > 0.0
                )
                if need_dipp_weight_rollout and update_weight_dipp and not dipp_traj_disabled:
                    init_control_for_weight = (
                        init_control.detach()
                        if cfg_runtime.get("detach_init_control_for_weight", True)
                        else init_control
                    )
                    try:
                        best_control_w, ego_dipp_traj_w, _ = run_upper_dipp(
                            upper_dipp,
                            ego_state,
                            lane_for_control,
                            init_control_for_weight,
                            cost_weights,
                            ped_states,
                            veh_states,
                            ped_mask,
                            veh_mask,
                            planner_weight_min=cfg_runtime.get(
                                "planner_weight_min_vector", cfg_runtime.get("planner_weight_min", 1e-3)
                            ),
                            planner_weight_max=cfg_runtime.get(
                                "planner_weight_max_vector", cfg_runtime.get("planner_weight_max", 20.0)
                            ),
                            planner_weight_renormalize_to_default_sum=cfg_runtime.get(
                                "planner_weight_renormalize_to_default_sum", False
                            ),
                            ped_safety_distance=cfg_runtime.get(
                                "planner_ped_safety_distance", 2.5
                            ),
                            veh_safety_distance=cfg_runtime.get(
                                "planner_veh_safety_distance", 4.0
                            ),
                            ped_lateral_safety_distance=cfg_runtime.get(
                                "planner_ped_lateral_safety_distance", 1.2
                            ),
                            veh_lateral_safety_distance=cfg_runtime.get(
                                "planner_veh_lateral_safety_distance", 1.8
                            ),
                            control_anchor_weight=cfg_runtime.get(
                                "planner_control_anchor_weight", 500.0
                            ),
                            control_anchor_risk_floor=cfg_runtime.get(
                                "planner_control_anchor_risk_floor", 0.05
                            ),
                        )
                        # WeightNet and Theseus are training-only modules. Use
                        # the complete optimized trajectory here so WeightNet
                        # must improve the eight objective weights instead of
                        # reducing a deployment-time refinement gate.
                        ego_dipp_metric_traj_w = apply_forward_offset_to_traj(
                            ego_dipp_traj_w,
                            -reference_forward_offset,
                        )
                        weight_loss_traj = compute_weight_loss_differentiable(
                            ego_dipp_metric_traj_w,
                            ego_future_gt_new,
                            ego_future_gt_valid_mask,
                        )
                        if gt_actor_boxes_2hz is not None and gt_actor_mask_2hz is not None:
                            dipp_g_safety = compute_gt_actor_box_violation(
                                ego_traj=ego_dipp_metric_traj_w,
                                gt_actor_boxes_2hz=gt_actor_boxes_2hz,
                                gt_actor_mask_2hz=gt_actor_mask_2hz,
                                rect_distance=safety_loss_module.rect_dist,
                                sample_valid_mask=official_fut_valid_mask,
                                frame_valid_mask=gt_obj_frame_valid_mask,
                                margin=cfg_runtime.get("weight_dipp_safety_margin", 0.25),
                                topk=cfg_runtime.get("metric_safety_topk", 1),
                            )
                            dipp_risk_weight = 1.0 + float(
                                cfg_runtime.get("weight_dipp_risk_gain", 8.0)
                            ) * weight_obj_collision_gate
                            weight_dipp_safety_loss = (
                                dipp_risk_weight * dipp_g_safety
                            ).sum() / dipp_risk_weight.sum().clamp_min(1e-8)

                        if lambda_weight_dipp_lane > 0.0:
                            dipp_g_lane = compute_metric_solid_lane_violation(
                                ego_traj=ego_dipp_metric_traj_w,
                                solid_lane_points=gt_solid_lane_points,
                                solid_lane_mask=gt_solid_lane_mask,
                                frame_valid_mask=gt_lane_frame_valid_mask,
                                gt_reference_line=gt_reference_line,
                                use_gt_safe_side=cfg_runtime.get(
                                    "metric_lane_use_gt_safe_side", False
                                ),
                                time_weights=cfg_runtime.get(
                                    "metric_lane_time_weights",
                                    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                                ),
                                hard_max_weight=cfg_runtime.get(
                                    "metric_lane_hard_max_weight", 0.0
                                ),
                                margin=cfg_runtime.get("weight_dipp_lane_margin", 0.05),
                            )
                            dipp_lane_weight = 1.0 + float(
                                cfg_runtime.get("weight_dipp_lane_risk_gain", 8.0)
                            ) * weight_lane_collision_gate
                            weight_dipp_lane_loss = (
                                dipp_lane_weight * dipp_g_lane
                            ).sum() / dipp_lane_weight.sum().clamp_min(1e-8)

                        dipp_trust_per_sample = F.smooth_l1_loss(
                            ego_dipp_metric_traj_w[:, :, :2],
                            ego_metric_traj.detach()[:, :, :2],
                            reduction="none",
                            beta=cfg_runtime.get("weight_dipp_trust_beta", 0.25),
                        ).mean(dim=(1, 2))
                        trust_floor = float(cfg_runtime.get("weight_dipp_trust_risk_floor", 0.1))
                        trust_weight = 1.0 - (1.0 - trust_floor) * weight_any_collision_gate
                        weight_dipp_trust_loss = (trust_weight * dipp_trust_per_sample).mean()
                    except RuntimeError as err:
                        if not is_theseus_singular_error(err):
                            raise
                        dipp_failed_local = True
                        weight_loss_traj = torch.zeros((), device=device, dtype=ego_state.dtype)
                        weight_dipp_safety_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
                        weight_dipp_lane_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
                        weight_dipp_trust_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError:
                            pass
                        if rank == 0:
                            old_idx = batch.get("old_index", None)
                            old_idx_msg = old_idx[: min(8, old_idx.numel())].tolist() if old_idx is not None else []
                            print(
                                f"[WARN] skip DIPP trajectory term because Theseus planner failed | "
                                f"epoch={epoch} | batch={batch_idx} | global_step={global_step} | "
                                f"cost_min={cost_weights.detach().min().item():.6g} | "
                                f"cost_max={cost_weights.detach().max().item():.6g} | "
                                f"old_index_head={old_idx_msg} | err={str(err).splitlines()[0]}"
                            )
                else:
                    weight_loss_traj = torch.zeros((), device=device, dtype=ego_state.dtype)

                dipp_failed_flag = torch.tensor(float(dipp_failed_local), device=device)
                if dist.is_initialized():
                    dist.all_reduce(dipp_failed_flag, op=dist.ReduceOp.MAX)
                if dipp_failed_flag.item() > 0.0:
                    weight_loss_traj = torch.zeros((), device=device, dtype=ego_state.dtype)
                    weight_dipp_safety_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
                    weight_dipp_lane_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
                    weight_dipp_trust_loss = torch.zeros((), device=device, dtype=ego_state.dtype)
                    if cfg_runtime.get("disable_dipp_traj_after_failure", True):
                        dipp_traj_disabled = True

                rule_target_prob = build_semantic_rule_target_weights(
                    scene_vec=scene_vec,
                    risk_score=risk_score,
                    cfg_runtime=cfg_runtime,
                )
                feedback_target_prob = compute_feedback_target_weights(
                    weighted_ego_comps=weighted_ego_comps,
                    g_lane=g_lane,
                    g_safety=g_safety,
                    risk_score=risk_score,
                    scene_prior_weights=scene_prior_weights,
                    cfg_runtime=cfg_runtime,
                )
                weight_rule_loss = compute_rule_kl_loss(pred_weights_prob, rule_target_prob)
                hipad_risk_target_prob = build_hipad_collision_target_weights(
                    scene_prior_weights=scene_prior_weights,
                    hipad_collision_gate=hipad_collision_gate,
                    cfg_runtime=cfg_runtime,
                )
                hipad_risk_sample_weight = 1.0 + float(
                    cfg_runtime.get("weight_hipad_risk_positive_weight", 16.0)
                ) * hipad_collision_gate
                weight_hipad_risk_loss = compute_sample_weighted_kl_loss(
                    pred_weights_prob,
                    hipad_risk_target_prob,
                    hipad_risk_sample_weight,
                )
                pnn_collision_target_prob = build_pnn_collision_target_weights(
                    scene_prior_weights=scene_prior_weights,
                    obj_collision_gate=weight_obj_collision_gate,
                    lane_collision_gate=weight_lane_collision_gate,
                    cfg_runtime=cfg_runtime,
                )
                pnn_collision_sample_weight = 1.0 + float(
                    cfg_runtime.get("weight_pnn_collision_positive_weight", 12.0)
                ) * weight_any_collision_gate
                weight_pnn_collision_loss = compute_sample_weighted_kl_loss(
                    pred_weights_prob,
                    pnn_collision_target_prob,
                    pnn_collision_sample_weight,
                )
                weight_feedback_loss = compute_rule_kl_loss(pred_weights_prob, feedback_target_prob)
                weight_rank_loss = compute_risk_conditioned_ranking_loss(pred_weights_prob, risk_score, cfg_runtime)

                weight_sep_loss = compute_weight_separation_loss(
                    pred_weights_prob,
                    risk_score,
                    base_margin=cfg_runtime.get("weight_sep_base_margin", 0.03),
                    scalar_scale=cfg_runtime.get("weight_sep_scalar_scale", 0.20),
                    min_risk_gap=cfg_runtime.get("weight_sep_min_risk_gap", 0.10),
                )

                weight_extreme_loss = compute_weight_extreme_penalty(
                    pred_weights_prob,
                    max_allowed_prob=cfg_runtime.get("weight_max_allowed_prob", 0.55),
                    min_allowed_prob=cfg_runtime.get("weight_min_allowed_prob", 0.0),
                )
                weight_entropy_band_loss, entropy = compute_entropy_band_loss(
                    pred_weights_prob,
                    entropy_low=cfg_runtime.get("entropy_band_low", 1.30),
                    entropy_high=cfg_runtime.get("entropy_band_high", 2.00),
                )
                weight_diversity_floor_loss, pairwise_l2 = compute_pairwise_diversity_floor_loss(
                    pred_weights_prob,
                    target_pairwise_l2=cfg_runtime.get("diversity_floor_pairwise_l2", 0.015),
                )

                entropy, diversity, kl_to_prior = compute_weight_regularizers(
                    pred_weights_prob,
                    scene_prior_weights.detach(),
                )

                decay = cfg_runtime.get("weight_reg_decay", 0.99) ** epoch
                lambda_entropy = max(base_lambda_entropy * decay, cfg_runtime.get("lambda_entropy_min", 1e-4))
                lambda_diversity = max(base_lambda_diversity * decay, cfg_runtime.get("lambda_diversity_min", 1e-4))
                lambda_kl = max(base_lambda_kl * decay, cfg_runtime.get("lambda_kl_min", 1e-4))
                lambda_entropy *= supervision_ramp
                lambda_diversity *= supervision_ramp
                lambda_kl *= supervision_ramp
                lambda_weight_rule = cfg_runtime.get("lambda_weight_rule", 0.20) * supervision_ramp
                lambda_weight_feedback = cfg_runtime.get("lambda_weight_feedback", 0.15) * supervision_ramp
                lambda_weight_rank = cfg_runtime.get("lambda_weight_rank", 0.10) * supervision_ramp
                lambda_weight_sep = cfg_runtime.get("lambda_weight_sep", 0.03) * supervision_ramp
                lambda_weight_extreme = cfg_runtime.get("lambda_weight_extreme", 0.05) * supervision_ramp
                lambda_entropy_band = cfg_runtime.get("lambda_entropy_band", 0.02) * supervision_ramp
                lambda_diversity_floor = cfg_runtime.get("lambda_diversity_floor", 0.02) * supervision_ramp

                total_weight_loss = (
                    lambda_weight_traj * weight_loss_traj
                    + lambda_weight_dipp_safety * weight_dipp_safety_loss
                    + lambda_weight_dipp_lane * weight_dipp_lane_loss
                    + lambda_weight_dipp_trust * weight_dipp_trust_loss
                    + float(cfg_runtime.get("lambda_weight_hipad_risk", 0.0)) * weight_hipad_risk_loss
                    + float(cfg_runtime.get("lambda_weight_pnn_collision", 0.0))
                    * weight_pnn_collision_loss
                    + lambda_weight_rule * weight_rule_loss
                    + lambda_weight_feedback * weight_feedback_loss
                    + lambda_weight_rank * weight_rank_loss
                    + lambda_weight_sep * weight_sep_loss
                    + lambda_weight_extreme * weight_extreme_loss
                    + lambda_entropy_band * weight_entropy_band_loss
                    + lambda_diversity_floor * weight_diversity_floor_loss
                    - lambda_entropy * entropy
                    - lambda_diversity * diversity
                    + lambda_kl * kl_to_prior
                    # Keep the legacy gate parameters DDP-compatible when
                    # resuming epochs 0-1; the gate no longer affects training.
                    + 0.0 * refine_gate.sum()
                )
                total_weight_loss.backward()
                torch.nn.utils.clip_grad_norm_(weight_optimized_params, max_norm=1.0)
                optimizer_weight.step()
                optimizer_control.zero_grad()
                weight_loss = total_weight_loss.detach()

            control_core = residual_loss.mean()
            control_total = (
                control_core
                + aug_term
                + float(cfg_runtime.get("lambda_gt_reference_lane", 1.0)) * gt_reference_lane_loss
            )

            if update_control_epoch:
                local_bad_loss = not bool(torch.isfinite(control_total.detach()).item())
                bad_loss_flag = torch.tensor(
                    float(local_bad_loss), device=device, dtype=torch.float32
                )
                if dist.is_initialized():
                    dist.all_reduce(bad_loss_flag, op=dist.ReduceOp.MAX)
                if bad_loss_flag.item() > 0:
                    if local_bad_loss:
                        component_values = {
                            "control_core": float(control_core.detach().item()),
                            "aug_term": float(aug_term.detach().item()),
                            "gt_reference_lane": float(gt_reference_lane_loss.detach().item()),
                            **{
                                name: float(value.detach().mean().item())
                                for name, value in residual_aux.items()
                                if name.startswith("loss_") and torch.is_tensor(value)
                            },
                        }
                        print(
                            f"[v10] non-finite control loss at epoch={epoch} "
                            f"batch={batch_idx} rank={rank}: {component_values}",
                            flush=True,
                        )
                    raise FloatingPointError(
                        f"non-finite control loss at epoch={epoch} batch={batch_idx}"
                    )
                control_total.backward()
                local_bad_grad = any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all().item())
                    for parameter in nnc_dyn.parameters()
                )
                bad_grad_flag = torch.tensor(
                    float(local_bad_grad), device=device, dtype=torch.float32
                )
                if dist.is_initialized():
                    dist.all_reduce(bad_grad_flag, op=dist.ReduceOp.MAX)
                if bad_grad_flag.item() > 0:
                    optimizer_control.zero_grad(set_to_none=True)
                    if local_bad_grad:
                        print(
                            f"[v10] non-finite ControlNet gradient at epoch={epoch} "
                            f"batch={batch_idx} rank={rank}",
                            flush=True,
                        )
                    raise FloatingPointError(
                        f"non-finite ControlNet gradient at epoch={epoch} batch={batch_idx}"
                    )
                if dist.is_initialized():
                    if train_soft_constraint_lambdas:
                        for p in soft_lambda_module.parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                                p.grad /= world_size

                torch.nn.utils.clip_grad_norm_(nnc_dyn.parameters(), max_norm=5.0)
                if train_soft_constraint_lambdas:
                    torch.nn.utils.clip_grad_norm_(soft_lambda_module.parameters(), max_norm=1.0)
                optimizer_control.step()

            if (
                update_control_epoch
                and ema_update_interval > 0
                and global_step >= cfg_runtime.get("ema_update_start_step", 1)
                and global_step % ema_update_interval == 0
            ):
                ema_update_weight_encoder_from_control(
                    unwrap_module(weight_model),
                    policy_core,
                    beta=cfg_runtime["ema_beta"],
                    names=cfg_runtime.get(
                        "ema_weightnet_modules",
                        ("ego_encoder", "ped_encoder", "veh_encoder", "map_encoder"),
                    ),
                )

            if lane_loss_module.log_lambda.grad is not None:
                lane_loss_module.log_lambda.grad.zero_()
            lane_loss_module.log_lambda.requires_grad_(True)

            if safety_loss_module.log_lambda.grad is not None:
                safety_loss_module.log_lambda.grad.zero_()
            safety_loss_module.log_lambda.requires_grad_(True)

            with torch.no_grad():
                g_lane_global = g_lane.detach().mean().clamp(min=0.0)
                g_safety_global = g_safety.detach().mean().clamp(min=0.0)
                if dist.is_initialized():
                    dist.all_reduce(g_lane_global, op=dist.ReduceOp.SUM)
                    dist.all_reduce(g_safety_global, op=dist.ReduceOp.SUM)
                    g_lane_global /= world_size
                    g_safety_global /= world_size

                g_lane_mean = g_lane_global.item()
                g_ema_lane = ema_momentum * g_ema_lane + (1 - ema_momentum) * g_lane_mean
                if g_ema_lane > eps_dead_lane:
                    delta_log_lane = eta_dual_lane * (g_ema_lane - eps_dead_lane) / max(
                        lane_loss_module.lambda_val.item(), 1e-8
                    )
                    lane_loss_module.log_lambda.add_(delta_log_lane)
                else:
                    lane_loss_module.log_lambda.mul_(1.0 - decay_dual)
                lane_loss_module.log_lambda.clamp_(min=np.log(1e-6), max=np.log(lambda_cap))

                g_safety_mean = g_safety_global.item()
                g_ema_safety = ema_momentum * g_ema_safety + (1 - ema_momentum) * g_safety_mean
                if g_ema_safety > eps_dead_safety:
                    delta_log_safety = eta_dual_safety * (g_ema_safety - eps_dead_safety) / max(
                        safety_loss_module.lambda_val.item(), 1e-8
                    )
                    safety_loss_module.log_lambda.add_(delta_log_safety)
                else:
                    safety_loss_module.log_lambda.mul_(1.0 - decay_dual)
                safety_loss_module.log_lambda.clamp_(min=np.log(1e-6), max=np.log(lambda_cap))

            if writer is not None:
                if update_weight:
                    writer.add_scalar("loss/weight_total", weight_loss.item(), global_step)
                    writer.add_scalar("loss/weight_traj", weight_loss_traj.item(), global_step)
                    writer.add_scalar("loss/weight_rule", weight_rule_loss.item(), global_step)
                    writer.add_scalar("loss/weight_feedback", weight_feedback_loss.item(), global_step)
                    writer.add_scalar("loss/weight_rank", weight_rank_loss.item(), global_step)
                    writer.add_scalar("loss/weight_sep", weight_sep_loss.item(), global_step)
                    writer.add_scalar("loss/weight_entropy_band", weight_entropy_band_loss.item(), global_step)
                    writer.add_scalar("loss/weight_diversity_floor", weight_diversity_floor_loss.item(), global_step)
                    writer.add_scalar("loss/weight_extreme", weight_extreme_loss.item(), global_step)
                    writer.add_scalar("loss/weight_hipad_risk", weight_hipad_risk_loss.item(), global_step)
                    writer.add_scalar(
                        "loss/weight_pnn_collision", weight_pnn_collision_loss.item(), global_step
                    )
                    writer.add_scalar("loss/weight_dipp_safety", weight_dipp_safety_loss.item(), global_step)
                    writer.add_scalar("loss/weight_dipp_lane", weight_dipp_lane_loss.item(), global_step)
                    writer.add_scalar("loss/weight_dipp_trust", weight_dipp_trust_loss.item(), global_step)
                    writer.add_scalar("weight/lambda_weight_traj", lambda_weight_traj, global_step)
                    writer.add_scalar("weight/lambda_weight_rule", lambda_weight_rule, global_step)
                    writer.add_scalar("weight/lambda_weight_feedback", lambda_weight_feedback, global_step)
                    writer.add_scalar("weight/lambda_weight_rank", lambda_weight_rank, global_step)
                    writer.add_scalar("weight/lambda_weight_sep", lambda_weight_sep, global_step)
                    writer.add_scalar("weight/lambda_weight_extreme", lambda_weight_extreme, global_step)
                    writer.add_scalar("weight/supervision_ramp", supervision_ramp, global_step)
                    writer.add_scalar(
                        "weight/dipp_update", float(update_weight_dipp), global_step
                    )
                    writer.add_scalar("weight/entropy", entropy.item(), global_step)
                    writer.add_scalar("weight/diversity", diversity.item(), global_step)
                    writer.add_scalar("weight/pairwise_l2", pairwise_l2.item(), global_step)
                    writer.add_scalar("weight/kl_to_prior", kl_to_prior.item(), global_step)
                    writer.add_scalar("weight/min_dist_mean", min_dist.mean().item(), global_step)
                    writer.add_scalar("weight/risk_score_mean", risk_score.mean().item(), global_step)
                    writer.add_scalar("weight/rollout_collision_risk_mean", rollout_collision_risk.mean().item(), global_step)
                    writer.add_scalar("weight/prior_safe_score_mean", prior_out["safe_score"].mean().item(), global_step)
                    writer.add_scalar("weight/hipad_gt_collision_rate", hipad_collision_gate.mean().item(), global_step)
                    writer.add_scalar(
                        "weight/pnn_obj_collision_gate_rate",
                        weight_obj_collision_gate.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "weight/pnn_lane_collision_gate_rate",
                        weight_lane_collision_gate.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "weight/pnn_any_collision_gate_rate",
                        weight_any_collision_gate.mean().item(),
                        global_step,
                    )
                    for gate_name, gate in (
                        ("obj", weight_obj_collision_gate),
                        ("lane", weight_lane_collision_gate),
                        ("any", weight_any_collision_gate),
                    ):
                        gate_mask = gate > 0.5
                        if gate_mask.any():
                            for cost_name in COST_NAMES:
                                writer.add_scalar(
                                    f"weight_by_collision/{gate_name}_{cost_name}",
                                    cost_weights[gate_mask, COST_NAMES.index(cost_name)].mean().item(),
                                    global_step,
                                )
                    risk_mask = hipad_collision_gate > 0.5
                    free_mask = ~risk_mask
                    if risk_mask.any():
                        writer.add_scalar(
                            "weight/safe_prob_hipad_collision",
                            pred_weights_prob[risk_mask, COST_NAMES.index("safe")].mean().item(),
                            global_step,
                        )
                    if free_mask.any():
                        writer.add_scalar(
                            "weight/safe_prob_hipad_safe",
                            pred_weights_prob[free_mask, COST_NAMES.index("safe")].mean().item(),
                            global_step,
                        )
                    writer.add_histogram("weight/scene_vec", scene_vec.detach().cpu(), global_step)
                    writer.add_histogram("weight/scene_prior_weights", scene_prior_weights.detach().cpu(), global_step)
                    writer.add_histogram("weight/logspace_delta", weight_delta.detach().cpu(), global_step)
                    writer.add_histogram("weight/feedback_target_prob", feedback_target_prob.detach().cpu(), global_step)
                    writer.add_histogram("weight/rule_target_prob", rule_target_prob.detach().cpu(), global_step)

                writer.add_scalar(
                    "weight/control_uses_adaptive_weights",
                    float(epoch >= cfg_runtime.get("control_weight_start_epoch", 0)),
                    global_step,
                )
                writer.add_scalar("weight/max_prob", pred_weights_prob.detach().max(dim=-1).values.mean().item(), global_step)
                writer.add_histogram("weight/cost_weights", cost_weights_detached.detach().cpu(), global_step)
                writer.add_histogram("weight/pred_weights_prob", pred_weights_prob.detach().cpu(), global_step)

                for idx, name in enumerate(COST_NAMES):
                    writer.add_scalar(f"weight/{name}", cost_weights_detached[:, idx].mean().item(), global_step)
                    writer.add_scalar(f"weight_std/{name}", pred_weights_prob[:, idx].std(unbiased=False).item(), global_step)
                    writer.add_scalar(f"ego_component/{name}", weighted_ego_comps[name].mean().item(), global_step)

                writer.add_scalar("constraint/g_lane_mean", g_lane.mean().item(), global_step)
                writer.add_scalar(
                    "constraint/g_metric_lane_mean",
                    residual_aux["g_metric_lane"].mean().item(),
                    global_step,
                )
                writer.add_scalar("constraint/g_safety_mean", g_safety.mean().item(), global_step)
                writer.add_scalar("constraint/teacher_risk_rate", teacher_risk_gate.mean().item(), global_step)
                writer.add_scalar(
                    "static/teacher_risk_rate",
                    teacher_static_risk_gate.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "static/positive_rate",
                    static_positive_gate.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "static/positive_scale",
                    static_positive_scale.item(),
                    global_step,
                )
                writer.add_scalar(
                    "static/hipad_risk_rate",
                    hipad_static_risk_gate.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "static/parent_pnn_only_rate",
                    static_parent_pnn_only_gate.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "static/target_scale",
                    static_target_scale.item(),
                    global_step,
                )
                if static_gate_prediction is not None:
                    writer.add_scalar(
                        "static/predicted_gate_mean",
                        static_gate_prediction.mean().item(),
                        global_step,
                    )
                    target_mask = static_parent_pnn_only_gate > 0.5
                    if target_mask.any():
                        writer.add_scalar(
                            "static/predicted_gate_target",
                            static_gate_prediction[target_mask].mean().item(),
                            global_step,
                        )
                    non_target_mask = ~target_mask
                    if non_target_mask.any():
                        writer.add_scalar(
                            "static/predicted_gate_non_target",
                            static_gate_prediction[non_target_mask].mean().item(),
                            global_step,
                        )
                writer.add_scalar("acr/pnn_risk_rate", pnn_collision_gate.mean().item(), global_step)
                writer.add_scalar("acr/hipad_risk_rate", hipad_collision_gate.mean().item(), global_step)
                writer.add_scalar("acr/pnn_only_rate", pnn_only_risk_gate.mean().item(), global_step)
                writer.add_scalar("acr/shared_rate", shared_risk_gate.mean().item(), global_step)
                if use_official_frame_acr:
                    writer.add_scalar(
                        "acr/frame_pnn_rate",
                        pnn_collision_frame_gate.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "acr/frame_hipad_rate",
                        hipad_collision_frame_gate.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "acr/frame_pnn_only_rate",
                        (
                            pnn_collision_frame_gate
                            * (1.0 - hipad_collision_frame_gate)
                        ).mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "acr/frame_hipad_lane_rate",
                        hipad_lane_collision_frame_gate.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "acr/frame_teacher_lane_rate",
                        teacher_lane_collision_frame_gate.mean().item(),
                        global_step,
                    )
                for name, value in train_proxy_metrics.items():
                    if torch.isfinite(value):
                        writer.add_scalar(f"train_proxy/{name}", value.item(), global_step)

            epoch_control_total += float(control_total.item())
            epoch_control_core += float(control_core.item())
            epoch_gt_reference_lane += float(gt_reference_lane_loss.item())
            for name in MONITORED_CONTROL_LOSS_COMPONENT_NAMES:
                epoch_control_components[name] += float(residual_aux[name].mean().item())
            epoch_weight_loss += float(weight_loss.item())
            epoch_weight_traj_loss += float(weight_loss_traj.item())
            epoch_weight_rule_loss += float(weight_rule_loss.item())
            epoch_weight_feedback_loss += float(weight_feedback_loss.item())
            epoch_weight_rank_loss += float(weight_rank_loss.item())
            epoch_weight_sep_loss += float(weight_sep_loss.item())
            epoch_weight_entropy_band_loss += float(weight_entropy_band_loss.item())
            epoch_weight_diversity_floor_loss += float(weight_diversity_floor_loss.item())
            epoch_weight_extreme_loss += float(weight_extreme_loss.item())
            epoch_weight_pnn_collision_loss += float(weight_pnn_collision_loss.item())
            epoch_weight_dipp_safety_loss += float(weight_dipp_safety_loss.item())
            epoch_weight_dipp_lane_loss += float(weight_dipp_lane_loss.item())
            epoch_weight_dipp_trust_loss += float(weight_dipp_trust_loss.item())
            epoch_aug += float(aug_term.item())
            epoch_entropy += float(entropy.item())
            epoch_diversity += float(diversity.item())
            epoch_kl += float(kl_to_prior.item())
            epoch_costw_sum += cost_weights_detached.mean(dim=0).detach()
            epoch_costw_sq_sum += cost_weights_detached.pow(2).mean(dim=0).detach()
            epoch_pairwise_l2_sum += float(pairwise_l2.item())
            for name, value in train_proxy_metrics.items():
                value = value.detach()
                finite = torch.isfinite(value)
                if name not in epoch_proxy_sums:
                    epoch_proxy_sums[name] = torch.zeros((), device=device, dtype=torch.float32)
                    epoch_proxy_counts[name] = torch.zeros((), device=device, dtype=torch.float32)
                epoch_proxy_sums[name] += torch.where(
                    finite,
                    value.to(device=device, dtype=torch.float32),
                    torch.zeros((), device=device, dtype=torch.float32),
                )
                epoch_proxy_counts[name] += finite.to(device=device, dtype=torch.float32)
            epoch_steps += 1

            if rank == 0 and batch_idx % 100 == 0:
                proxy_msg = ""
                if train_proxy_metrics:
                    l2_gt_3s = train_proxy_metrics.get("l2_gt_3s")
                    obj_col_3s = train_proxy_metrics.get("obj_col_proxy_3s")
                    lane_rate = train_proxy_metrics.get("lane_violation_rate")
                    if l2_gt_3s is not None and obj_col_3s is not None and lane_rate is not None:
                        proxy_msg = (
                            f" | proxy_l2_gt_3s={l2_gt_3s.item():.4f}"
                            f" | proxy_obj_col_3s={obj_col_3s.item() * 100:.2f}%"
                            f" | proxy_lane_rate={lane_rate.item() * 100:.2f}%"
                        )
                print(
                    f"epoch={epoch} | step={global_step} | update_weight={int(update_weight)} | "
                    f"update_dipp={int(update_weight_dipp)} | "
                    f"total={control_total.item():.2f} | gt_ref_lane={gt_reference_lane_loss.item():.4f} | "
                    f"weight_total={weight_loss.item():.4f} | "
                    f"weight_traj={weight_loss_traj.item():.4f} | rule={weight_rule_loss.item():.4f} | "
                    f"hipad_risk={weight_hipad_risk_loss.item():.4f} | "
                    f"pnn_collision={weight_pnn_collision_loss.item():.4f} | "
                    f"dipp_safe={weight_dipp_safety_loss.item():.4f} | "
                    f"dipp_lane={weight_dipp_lane_loss.item():.4f} | "
                    f"dipp_trust={weight_dipp_trust_loss.item():.4f} | "
                    f"pnn_obj/lane_col={weight_obj_collision_gate.mean().item() * 100:.2f}%/"
                    f"{weight_lane_collision_gate.mean().item() * 100:.2f}% | "
                    f"feedback={weight_feedback_loss.item():.4f} | rank={weight_rank_loss.item():.4f} | "
                    f"weight_sep={weight_sep_loss.item():.4f} | "
                    f"weight_extreme={weight_extreme_loss.item():.4f} | "
                    f"max_prob={pred_weights_prob.detach().max(dim=-1).values.mean().item():.4f} | "
                    f"ent={entropy.item():.4f} | div={diversity.item():.4f} | "
                    f"pair_l2={pairwise_l2.item():.4f} | kl={kl_to_prior.item():.4f}"
                    f"{proxy_msg}"
                )

        epoch_control_total = reduce_mean_scalar(epoch_control_total / max(epoch_steps, 1), device, world_size)
        epoch_control_core = reduce_mean_scalar(epoch_control_core / max(epoch_steps, 1), device, world_size)
        epoch_gt_reference_lane = reduce_mean_scalar(
            epoch_gt_reference_lane / max(epoch_steps, 1), device, world_size
        )
        epoch_control_components = {
            name: reduce_mean_scalar(value / max(epoch_steps, 1), device, world_size)
            for name, value in epoch_control_components.items()
        }
        epoch_weight_loss = reduce_mean_scalar(epoch_weight_loss / max(epoch_steps, 1), device, world_size)
        epoch_weight_traj_loss = reduce_mean_scalar(epoch_weight_traj_loss / max(epoch_steps, 1), device, world_size)
        epoch_weight_rule_loss = reduce_mean_scalar(epoch_weight_rule_loss / max(epoch_steps, 1), device, world_size)
        epoch_weight_feedback_loss = reduce_mean_scalar(epoch_weight_feedback_loss / max(epoch_steps, 1), device, world_size)
        epoch_weight_rank_loss = reduce_mean_scalar(epoch_weight_rank_loss / max(epoch_steps, 1), device, world_size)
        epoch_weight_sep_loss = reduce_mean_scalar(epoch_weight_sep_loss / max(epoch_steps, 1), device, world_size)
        epoch_weight_entropy_band_loss = reduce_mean_scalar(epoch_weight_entropy_band_loss / max(epoch_steps, 1), device, world_size)
        epoch_weight_diversity_floor_loss = reduce_mean_scalar(epoch_weight_diversity_floor_loss / max(epoch_steps, 1), device, world_size)
        epoch_weight_extreme_loss = reduce_mean_scalar(epoch_weight_extreme_loss / max(epoch_steps, 1), device, world_size)
        epoch_weight_pnn_collision_loss = reduce_mean_scalar(
            epoch_weight_pnn_collision_loss / max(epoch_steps, 1), device, world_size
        )
        epoch_weight_dipp_safety_loss = reduce_mean_scalar(
            epoch_weight_dipp_safety_loss / max(epoch_steps, 1), device, world_size
        )
        epoch_weight_dipp_lane_loss = reduce_mean_scalar(
            epoch_weight_dipp_lane_loss / max(epoch_steps, 1), device, world_size
        )
        epoch_weight_dipp_trust_loss = reduce_mean_scalar(
            epoch_weight_dipp_trust_loss / max(epoch_steps, 1), device, world_size
        )
        epoch_aug = reduce_mean_scalar(epoch_aug / max(epoch_steps, 1), device, world_size)
        epoch_entropy = reduce_mean_scalar(epoch_entropy / max(epoch_steps, 1), device, world_size)
        epoch_diversity = reduce_mean_scalar(epoch_diversity / max(epoch_steps, 1), device, world_size)
        epoch_kl = reduce_mean_scalar(epoch_kl / max(epoch_steps, 1), device, world_size)
        epoch_costw_mean = reduce_mean_tensor(epoch_costw_sum / max(epoch_steps, 1), world_size)
        epoch_costw_sq_mean = reduce_mean_tensor(epoch_costw_sq_sum / max(epoch_steps, 1), world_size)
        epoch_costw_std = (epoch_costw_sq_mean - epoch_costw_mean.pow(2)).clamp_min(0.0).sqrt()
        epoch_pairwise_l2 = reduce_mean_scalar(epoch_pairwise_l2_sum / max(epoch_steps, 1), device, world_size)
        epoch_proxy_metrics: Dict[str, float] = {}
        for name in sorted(epoch_proxy_sums.keys()):
            total = reduce_sum_tensor(epoch_proxy_sums[name])
            count = reduce_sum_tensor(epoch_proxy_counts[name])
            if count.item() > 0:
                epoch_proxy_metrics[name] = float((total / count.clamp_min(1.0)).item())
            else:
                epoch_proxy_metrics[name] = float("nan")
        prior_vec = torch.tensor(
            cfg_runtime.get("weight_prior", (1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0)),
            device=epoch_costw_mean.device,
            dtype=epoch_costw_mean.dtype,
        )
        relative_shift = ((epoch_costw_mean - prior_vec).abs() / prior_vec.clamp_min(1e-3)).mean().item()
        std_ratio = (epoch_costw_std / epoch_costw_mean.abs().clamp_min(1e-3)).mean().item()
        weight_variation_score = float(relative_shift + 2.0 * std_ratio + 3.0 * epoch_pairwise_l2)

        scheduler.step(epoch_control_total)

        stop_training_flag = torch.zeros((), device=device)

        if rank == 0:
            row = {
                "epoch": epoch,
                "control_total": epoch_control_total,
                "control_core": epoch_control_core,
                "gt_reference_lane_loss": epoch_gt_reference_lane,
                "weight_loss": epoch_weight_loss,
                "weight_traj_loss": epoch_weight_traj_loss,
                "weight_rule_loss": epoch_weight_rule_loss,
                "weight_feedback_loss": epoch_weight_feedback_loss,
                "weight_rank_loss": epoch_weight_rank_loss,
                "weight_sep_loss": epoch_weight_sep_loss,
                "weight_entropy_band_loss": epoch_weight_entropy_band_loss,
                "weight_diversity_floor_loss": epoch_weight_diversity_floor_loss,
                "weight_extreme_loss": epoch_weight_extreme_loss,
                "weight_pnn_collision_loss": epoch_weight_pnn_collision_loss,
                "weight_dipp_safety_loss": epoch_weight_dipp_safety_loss,
                "weight_dipp_lane_loss": epoch_weight_dipp_lane_loss,
                "weight_dipp_trust_loss": epoch_weight_dipp_trust_loss,
                "aug_term": epoch_aug,
                "entropy": epoch_entropy,
                "diversity": epoch_diversity,
                "kl_to_prior": epoch_kl,
                "lr_control": optimizer_control.param_groups[0]["lr"],
                "lr_weight": optimizer_weight.param_groups[0]["lr"],
            }
            for name, value in epoch_control_components.items():
                row[name] = value
            if writer is not None:
                writer.add_scalar("epoch_loss/control_core", epoch_control_core, epoch)
                writer.add_scalar("epoch_loss/gt_reference_lane", epoch_gt_reference_lane, epoch)
                writer.add_scalar(
                    "epoch_weight/pnn_collision", epoch_weight_pnn_collision_loss, epoch
                )
                writer.add_scalar(
                    "epoch_weight/dipp_safety", epoch_weight_dipp_safety_loss, epoch
                )
                writer.add_scalar("epoch_weight/dipp_lane", epoch_weight_dipp_lane_loss, epoch)
                writer.add_scalar(
                    "epoch_weight/dipp_trust", epoch_weight_dipp_trust_loss, epoch
                )
                for name, value in epoch_control_components.items():
                    writer.add_scalar(
                        f"epoch_loss/{name[len('loss_'):]}", value, epoch
                    )
                writer.add_scalar("learning_rate/control", optimizer_control.param_groups[0]["lr"], epoch)
                writer.add_scalar("learning_rate/weight", optimizer_weight.param_groups[0]["lr"], epoch)
            for idx, name in enumerate(COST_NAMES):
                row[f"costw_{name}"] = float(epoch_costw_mean[idx].item())
                row[f"costw_std_{name}"] = float(epoch_costw_std[idx].item())
            row["weight_pairwise_l2"] = float(epoch_pairwise_l2)
            row["weight_relative_shift"] = float(relative_shift)
            row["weight_std_ratio"] = float(std_ratio)
            row["weight_variation_score"] = float(weight_variation_score)
            for name, value in epoch_proxy_metrics.items():
                row[f"train_proxy_{name}"] = value
                if writer is not None:
                    writer.add_scalar(f"epoch_train_proxy/{name}", value, epoch)
            history_rows.append(row)

            ckpt = {
                "epoch": epoch,
                "history_rows": history_rows,
                "best_control_total": best_control_total,
                "neural_net": policy_core.module.state_dict(),
                "weight_model": unwrap_module(weight_model).state_dict(),
                "lane_loss_module": lane_loss_module.state_dict(),
                "safety_loss_module": safety_loss_module.state_dict(),
                "soft_lambda_module": soft_lambda_module.state_dict(),
                "optimizer_control": optimizer_control.state_dict(),
                "optimizer_weight": optimizer_weight.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg_runtime": cfg_runtime,
            }
            epoch_ckpt_path = os.path.join(save_dir, "checkpoints", f"epoch_{epoch:04d}.pth")
            torch.save(ckpt, epoch_ckpt_path)
            torch.save(ckpt, os.path.join(save_dir, "checkpoints", "last.pth"))

            if epoch_control_total < best_control_total:
                best_control_total = epoch_control_total
                ckpt["best_control_total"] = best_control_total
                torch.save(ckpt, epoch_ckpt_path)
                torch.save(ckpt, os.path.join(save_dir, "checkpoints", "last.pth"))
                torch.save(
                    ckpt,
                    os.path.join(save_dir, "checkpoints", "best_train_loss.pth"),
                )

            eval_row = run_l2_eval_for_checkpoint(
                ckpt_path=epoch_ckpt_path,
                save_dir=save_dir,
                epoch=epoch,
                weight_variation_score=weight_variation_score,
                cfg_runtime=cfg_runtime,
            )
            l2_avg = float(eval_row.get("L2", float("nan"))) if eval_row else float("nan")
            row["eval_l2_avg"] = l2_avg
            row["eval_obj_col"] = float(eval_row.get("obj_col", float("nan"))) if eval_row else float("nan")
            row["eval_obj_box_col"] = float(eval_row.get("obj_box_col", float("nan"))) if eval_row else float("nan")
            ckpt["history_rows"] = history_rows
            ckpt["eval_row"] = eval_row
            ckpt["eval_l2_avg"] = l2_avg
            ckpt["weight_variation_score"] = float(weight_variation_score)
            torch.save(ckpt, epoch_ckpt_path)
            torch.save(ckpt, os.path.join(save_dir, "checkpoints", "last.pth"))
            min_l2_save_variation = cfg_runtime.get("best_l2_min_weight_variation_score", 1.0)
            if eval_row and should_save_best_l2_checkpoint(
                l2_avg=l2_avg,
                weight_variation_score=weight_variation_score,
                best_l2_with_variation=best_l2_with_variation,
                min_weight_variation_score=min_l2_save_variation,
            ):
                best_l2_with_variation = l2_avg
                ckpt["eval_row"] = eval_row
                ckpt["best_l2_with_variation"] = best_l2_with_variation
                ckpt["best_l2_min_weight_variation_score"] = min_l2_save_variation
                best_l2_epoch_path = os.path.join(
                    save_dir,
                    "checkpoints",
                    f"best_l2_with_variation_epoch_{epoch:04d}.pth",
                )
                torch.save(ckpt, best_l2_epoch_path)
                torch.save(ckpt, os.path.join(save_dir, "checkpoints", "best_l2_with_variation.pth"))
                print(
                    f"[epoch {epoch}] new best L2 with variation: "
                    f"L2={l2_avg:.6f} | variation={weight_variation_score:.6f} | "
                    f"saved {best_l2_epoch_path}"
                )
            if (
                eval_row
                and l2_avg < cfg_runtime.get("target_l2_avg", 0.7)
                and weight_variation_score >= cfg_runtime.get("target_weight_variation_score", 0.30)
            ):
                satisfied_path = os.path.join(save_dir, "checkpoints", f"satisfied_epoch_{epoch:04d}.pth")
                ckpt["eval_row"] = eval_row
                torch.save(ckpt, satisfied_path)
                torch.save(ckpt, os.path.join(save_dir, "checkpoints", "satisfied_latest.pth"))
                print(f"[epoch {epoch}] satisfied target, saved {satisfied_path}")
                if cfg_runtime.get("stop_on_satisfied", True):
                    stop_training_flag.fill_(1.0)

            plot_history(history_rows, os.path.join(save_dir, "plots"))
            print(
                f"[epoch {epoch}] control_total={epoch_control_total:.6f} | "
                f"weight_loss={epoch_weight_loss:.6f} | weight_traj={epoch_weight_traj_loss:.6f} | "
                f"rule={epoch_weight_rule_loss:.6f} | feedback={epoch_weight_feedback_loss:.6f} | "
                f"rank={epoch_weight_rank_loss:.6f} | weight_sep={epoch_weight_sep_loss:.6f} | "
                f"weight_extreme={epoch_weight_extreme_loss:.6f} | "
                f"pnn_collision={epoch_weight_pnn_collision_loss:.6f} | "
                f"dipp_safe/lane={epoch_weight_dipp_safety_loss:.6f}/"
                f"{epoch_weight_dipp_lane_loss:.6f} | "
                f"aug={epoch_aug:.6f} | "
                f"entropy={epoch_entropy:.6f} | diversity={epoch_diversity:.6f} | "
                f"pair_l2={epoch_pairwise_l2:.6f} | variation={weight_variation_score:.6f} | "
                f"eval_l2={l2_avg:.6f} | kl={epoch_kl:.6f} | "
                f"best={best_control_total:.6f} | best_l2_var={best_l2_with_variation:.6f}"
            )
            if epoch_proxy_metrics:
                print(
                    f"[epoch {epoch}] train_proxy | "
                    f"l2_gt={epoch_proxy_metrics.get('l2_gt_1s', float('nan')):.4f}/"
                    f"{epoch_proxy_metrics.get('l2_gt_2s', float('nan')):.4f}/"
                    f"{epoch_proxy_metrics.get('l2_gt_3s', float('nan')):.4f} | "
                    f"l2_route={epoch_proxy_metrics.get('l2_route_1s', float('nan')):.4f}/"
                    f"{epoch_proxy_metrics.get('l2_route_2s', float('nan')):.4f}/"
                    f"{epoch_proxy_metrics.get('l2_route_3s', float('nan')):.4f} | "
                    f"obj_col_proxy={epoch_proxy_metrics.get('obj_col_proxy_1s', float('nan')) * 100:.2f}/"
                    f"{epoch_proxy_metrics.get('obj_col_proxy_2s', float('nan')) * 100:.2f}/"
                    f"{epoch_proxy_metrics.get('obj_col_proxy_3s', float('nan')) * 100:.2f}% | "
                    f"lane_rate={epoch_proxy_metrics.get('lane_violation_rate', float('nan')) * 100:.2f}% | "
                    f"comfort={epoch_proxy_metrics.get('comfort_score_3s', float('nan')) * 100:.2f}%"
                )

        if dist.is_initialized():
            dist.broadcast(stop_training_flag, src=0)
        if stop_training_flag.item() > 0.0:
            if rank == 0:
                print("[train_v7] target satisfied, stopping training early.")
            break

    if writer is not None:
        writer.close()
    cleanup()


def main():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(find_free_port()))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,7,8,9"
    torch.manual_seed(27)

    print(
        f"[DDP] MASTER_ADDR={os.environ['MASTER_ADDR']} "
        f"MASTER_PORT={os.environ['MASTER_PORT']} "
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
    )

    cfg_runtime = {
        "old_train_data_path": os.path.join(PROJECT_ROOT, "data", "pnn", "train_old.pt"),
        "new_train_data_path": os.path.join(PROJECT_ROOT, "data", "pnn", "train_new.pt"),
        "control_ckpt_path": os.path.join(PROJECT_ROOT, "checkpoints", "pnn_control.pth"),
        "save_dir": os.path.join(PROJECT_ROOT, "outputs", "pnn_static_v1"),
        "resume_ckpt_path": None,
        "batch_size": 48,
        "num_workers": 4,
        "epochs": 10,
        "lr_control": 2e-5,
        "lr_weight": 8e-4,
        "ema_beta": 0.995,
        "embed_dim": 128,
        "num_heads": 4,
        "weight_temperature": 0.7,
        "weightnet_use_prior_context": True,
        "weightnet_prior_context_mode": "log",
        "weight_prior": (1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0),
        "weightnet_outputs_residual": True,
        "weight_delta_max": (1.3, 1.4, 1.1, 1.4, 1.2, 1.2, 1.5, 1.4),
        "prior_dense_gain": 1.4,
        "prior_turn_gain": 1.2,
        "prior_high_speed_gain": 0.9,
        "prior_high_speed_threshold": 12.0,
        "prior_high_speed_sharpness": 3.0,
        "lambda_entropy": 2e-3,
        "lambda_diversity": 1e-3,
        "lambda_kl": 3e-3,
        "lambda_entropy_min": 1e-4,
        "lambda_diversity_min": 1e-4,
        "lambda_kl_min": 1e-4,
        "weight_reg_decay": 0.99,
        "weight_update_interval": 1,
        "ema_update_interval": 100,
        "ema_update_start_step": 1,
        "ema_weightnet_modules": ("ego_encoder", "ped_encoder", "veh_encoder", "map_encoder"),
        "weight_decay_weightnet": 5e-5,
        "safe_dist": 10.0,
        "collision_dist": 6.0,
        "collision_risk_sharpness": 1.5,
        "control_weight_start_epoch": 3,
        "weight_update_start_epoch": 0,
        "weight_supervision_start_epoch": 0,
        "weight_supervision_ramp_epochs": 5,
        "lambda_weight_traj": 0.35,
        "weight_traj_warmup_epochs": 2,
        "weight_traj_ramp_epochs": 4,
        "lambda_weight_rule": 0.38,
        "lambda_weight_feedback": 0.25,
        "lambda_weight_rank": 0.22,
        "lambda_weight_sep": 0.10,
        "lambda_weight_extreme": 0.03,
        "lambda_entropy_band": 0.03,
        "lambda_diversity_floor": 0.12,
        "entropy_band_low": 1.05,
        "entropy_band_high": 1.92,
        "diversity_floor_pairwise_l2": 0.060,
        "feedback_component_gain": 0.45,
        "rank_high_risk_th": 0.45,
        "rank_low_risk_th": 0.30,
        "rank_margin_safe": 0.035,
        "rank_margin_route": 0.030,
        "rank_margin_comfort": 0.025,
        "weight_max_allowed_prob": 0.68,
        "weight_min_allowed_prob": 0.0,
        "weight_sep_base_margin": 0.08,
        "weight_sep_scalar_scale": 0.45,
        "weight_sep_min_risk_gap": 0.06,
        "weak_free_weights": (2.4, 4.2, 1.1, 4.4, 0.45, 0.45, 7.0, 0.45),
        "weak_risky_weights": (0.35, 2.2, 0.35, 2.2, 4.8, 4.5, 1.2, 8.0),
        "detach_init_control_for_weight": True,
        "disable_dipp_traj_after_failure": True,
        "planner_optimizer": "levenberg_marquardt",
        "planner_max_iterations": 10,
        "planner_step_size": 0.10,
        "planner_weight_min": 1e-3,
        "planner_weight_max": 20.0,
        "planner_weight_min_vector": (0.05, 0.05, 0.02, 0.02, 0.20, 0.20, 0.20, 0.50),
        "planner_weight_max_vector": (8.0, 8.0, 5.0, 6.0, 14.0, 14.0, 16.0, 16.0),
        "prior_renormalize_to_default_sum": False,
        "planner_weight_renormalize_to_default_sum": False,
        "eval_each_epoch": False,
        "eval_cuda_visible_devices": "6",
        "eval_nnplanner_python": sys.executable,
        "eval_pinn_python": sys.executable,
        "target_l2_avg": 0.68,
        "target_weight_variation_score": 0.30,
        "best_l2_min_weight_variation_score": 1.0,
        "stop_on_satisfied": False,
    }

    world_size = 8
    mp.spawn(train, args=(world_size, cfg_runtime), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
