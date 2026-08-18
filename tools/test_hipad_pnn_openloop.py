import argparse
import copy
import os
import sys
import time
from os import path as osp
from pathlib import Path

import mmcv
import numpy as np
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor, build_dataset
from mmdet.models import build_detector

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.apis.test import collect_results_cpu, collect_results_gpu


PROJECT_ROOT = Path(
    os.environ.get("HIPAD_PNN_ROOT", Path(__file__).resolve().parents[1])
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pnn_temporal_alignment import (  # noqa: E402
    ALIGNMENT_VERSION,
    HIPAD_MOTION_DT,
    align_hipad_motion_future,
)
from hipad_pnn_adapter import (  # noqa: E402
    PNNAdapterConfig,
    PNNOptimizerAdapter,
    select_left_right_lane_boundaries,
)


DEFAULT_PNN_CKPT = str(PROJECT_ROOT / "checkpoints" / "pnn_control.pth")
DEFAULT_PNN_STATS = str(PROJECT_ROOT / "checkpoints" / "pnn_stats.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HiP-AD test with PNN open-loop planning replacement"
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="HiP-AD checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--fuse-conv-bn", action="store_true")
    parser.add_argument("--format-only", action="store_true")
    parser.add_argument("--eval", type=str, nargs="+")
    parser.add_argument("--gpu-collect", action="store_true")
    parser.add_argument("--tmpdir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--eval-options", nargs="+", action=DictAction)
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--show_only", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-dir")

    parser.add_argument("--pnn-ckpt", default=os.environ.get("PNN_OPENLOOP_CKPT", DEFAULT_PNN_CKPT))
    parser.add_argument("--pnn-control-ckpt", default=os.environ.get("PNN_OPENLOOP_CONTROL_CKPT"))
    parser.add_argument("--pnn-weight-ckpt", default=os.environ.get("PNN_OPENLOOP_WEIGHT_CKPT"))
    parser.add_argument("--pnn-stats", default=os.environ.get("PNN_OPENLOOP_STATS", DEFAULT_PNN_STATS))
    parser.add_argument(
        "--pnn-no-weight",
        action="store_true",
        default=os.environ.get("PNN_OPENLOOP_USE_WEIGHT_NET", "0") == "0",
        help="disable WeightNet",
    )
    parser.add_argument(
        "--pnn-use-theseus",
        action="store_true",
        default=os.environ.get("PNN_OPENLOOP_USE_THESEUS", "0") == "1",
        help="apply learned cost weights through the DIPP/Theseus optimizer",
    )
    parser.add_argument(
        "--pnn-planner-max-iterations",
        type=int,
        default=int(os.environ.get("PNN_PLANNER_MAX_ITERATIONS", "5")),
    )
    parser.add_argument("--pnn-score-thr", type=float, default=float(os.environ.get("PNN_OPENLOOP_SCORE_THR", "0.3")))
    parser.add_argument("--pnn-debug-limit", type=int, default=int(os.environ.get("PNN_OPENLOOP_DEBUG_LIMIT", "0")))
    parser.add_argument(
        "--pnn-planning-only",
        action="store_true",
        default=os.environ.get("PNN_OPENLOOP_PLANNING_ONLY", "0") == "1",
        help=(
            "only collect/print planning metrics. This avoids collecting large "
            "detection/vector outputs and skips perception formatting/evaluation."
        ),
    )
    parser.add_argument(
        "--pnn-max-batches",
        type=int,
        default=int(os.environ.get("PNN_OPENLOOP_MAX_BATCHES", "0")),
        help="debug/smoke-test limit per distributed rank; 0 evaluates the full dataset",
    )
    parser.add_argument(
        "--pnn-keep-plans",
        action="store_true",
        default=os.environ.get("PNN_OPENLOOP_KEEP_PLANS", "1") == "1",
        help=(
            "when --pnn-planning-only is enabled, keep lightweight PNN/HiP-AD "
            "planning tensors for visualization/diagnostics"
        ),
    )
    parser.add_argument(
        "--pnn-output-forward-offset",
        type=float,
        default=float(os.environ.get("PNN_OUTPUT_FORWARD_OFFSET", "0.0")),
        help="meters; post-rollout shift along each predicted heading before metric/PID output",
    )
    parser.add_argument(
        "--pnn-reference-forward-offset",
        type=float,
        default=float(os.environ.get("PNN_REFERENCE_FORWARD_OFFSET", "0.0")),
        help=(
            "meters; PNN internal ego reference point forward offset relative "
            "to HiP-AD/GT ego reference. Positive means PNN state point is "
            "ahead; outputs are shifted back before metric/PID exposure."
        ),
    )
    parser.add_argument(
        "--pnn-coord-convention",
        choices=["hipad_xy", "pnn_xy"],
        default=os.environ.get("PNN_COORD_CONVENTION", "hipad_xy"),
        help="coordinate convention used inside PNN network/dynamics",
    )
    parser.add_argument(
        "--pnn-stats-quantile-low",
        type=float,
        default=float(os.environ.get("PNN_STATS_QUANTILE_LOW", "0.0")),
        help="lower quantile for robust PNN min/max normalization stats",
    )
    parser.add_argument(
        "--pnn-stats-quantile-high",
        type=float,
        default=float(os.environ.get("PNN_STATS_QUANTILE_HIGH", "1.0")),
        help="upper quantile for robust PNN min/max normalization stats",
    )
    parser.add_argument(
        "--pnn-clamp-normalized-inputs",
        action="store_true",
        default=os.environ.get("PNN_CLAMP_NORMALIZED_INPUTS", "0") == "1",
        help="clamp robust-normalized PNN inputs to [-1, 1]",
    )
    parser.add_argument(
        "--pnn-route-source",
        choices=["navigation", "hipad_plan"],
        default=os.environ.get("PNN_OPENLOOP_ROUTE_SOURCE", "navigation"),
    )
    parser.add_argument("--pnn-nav-min-speed", type=float, default=float(os.environ.get("PNN_NAV_MIN_SPEED", "1.0")))
    parser.add_argument("--pnn-nav-max-speed", type=float, default=float(os.environ.get("PNN_NAV_MAX_SPEED", "15.0")))
    parser.add_argument(
        "--pnn-nav-distance-scale",
        default=os.environ.get("PNN_NAV_DISTANCE_SCALE", "1.0"),
        help="scalar or comma-separated per-horizon scales, e.g. 0.96,1.03,1.12",
    )
    parser.add_argument(
        "--pnn-nav-interpolation",
        choices=["spline", "polyline"],
        default=os.environ.get("PNN_NAV_INTERPOLATION", "spline"),
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def _as_numpy(x):
    if x is None:
        return None
    if hasattr(x, "data") and not torch.is_tensor(x):
        x = x.data
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
    if hasattr(x, "tensor"):
        x = x.tensor
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_tensor_on(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _get_img_bbox(result_item):
    return result_item.get("img_bbox", result_item)


def _extract_plan(det):
    for key in ("plan_temp_2hz", "plan_speed_2hz", "plan_temp_5hz", "plan_speed_5hz"):
        if key in det:
            plan = _as_numpy(det[key]).astype(np.float32)
            if plan.ndim == 3:
                plan = plan[0]
            if plan.shape[0] >= 6:
                return plan[:6]
    raise KeyError("No HiP-AD planning key found in result")


def _extract_speed(data, bs):
    if "ego_status" not in data:
        return 0.0
    status = data["ego_status"]
    if hasattr(status, "data") and not torch.is_tensor(status):
        status = status.data
    if isinstance(status, (list, tuple)):
        status = status[0]
    if torch.is_tensor(status):
        return float(status[bs, 0].detach().cpu().item())
    arr = np.asarray(status)
    return float(arr[bs, 0])


def _extract_navigation_points(data, bs):
    points = []
    for key in ("target_point_near", "target_point", "target_point_next"):
        if key not in data:
            continue
        value = _unwrap_data_container(data[key])
        if torch.is_tensor(value):
            arr = value[bs].detach().cpu().numpy()
        else:
            arr = np.asarray(value)[bs]
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 2:
            point = arr[:2]
            if not points or float(np.linalg.norm(point - points[-1])) > 1e-3:
                points.append(point)
    if not points:
        raise KeyError(
            "PNN navigation route requires target_point in the open-loop data pipeline"
        )
    return np.stack(points, axis=0).astype(np.float32)


def _unwrap_data_container(value):
    if hasattr(value, "data") and not torch.is_tensor(value):
        value = value.data
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _unwrap_batch_data(data):
    return {k: _unwrap_data_container(v) for k, v in data.items()}


def _extract_lane_points(det, reference_plan=None):
    if not all(k in det for k in ("vectors", "scores", "labels")):
        return None
    vectors = _as_numpy(det["vectors"])
    scores = _as_numpy(det["scores"])
    labels = _as_numpy(det["labels"])
    if vectors is None or len(vectors) == 0:
        return None
    return select_left_right_lane_boundaries(
        vectors,
        scores=scores,
        labels=labels,
        reference_plan=reference_plan,
    )


def _agent_goal_from_traj(
    trajs, traj_scores, idx, fallback_xy, measured_velocity_xy, max_speed
):
    if trajs is None:
        return fallback_xy
    arr = _as_numpy(trajs)
    try:
        fut = arr[idx]
        if fut.ndim == 3:
            scores = _as_numpy(traj_scores)
            mode = int(np.argmax(scores[idx])) if scores is not None else 0
            fut = fut[mode]
        while fut.ndim > 2:
            fut = fut[0]
        if fut.ndim == 2 and fut.shape[-1] >= 2 and fut.shape[0] > 0:
            aligned = align_hipad_motion_future(
                current_xy=fallback_xy,
                future=fut[:, :2],
                source_dt=HIPAD_MOTION_DT,
                measured_velocity_xy=measured_velocity_xy,
                max_speed=max_speed,
            )
            return aligned[-1].astype(np.float32)
    except Exception:
        pass
    return fallback_xy


def _extract_agents(det, score_thr=0.3):
    boxes = _as_numpy(det.get("boxes_3d"))
    labels = _as_numpy(det.get("labels_3d"))
    scores = _as_numpy(det.get("scores_3d"))
    trajs = det.get("trajs_3d", None)
    traj_scores = det.get("trajs_score", None)
    ped_agents, veh_agents = [], []
    if boxes is None or labels is None or scores is None:
        return ped_agents, veh_agents

    for i in range(len(boxes)):
        if float(scores[i]) <= score_thr:
            continue
        label = int(labels[i])
        x, y = float(boxes[i, 0]), float(boxes[i, 1])
        yaw = float(boxes[i, 6]) if boxes.shape[1] > 6 else np.pi / 2
        if boxes.shape[1] > 8:
            vx, vy = float(boxes[i, 7]), float(boxes[i, 8])
            speed = float(np.hypot(vx, vy))
        else:
            speed = 0.0
        velocity_xy = boxes[i, 7:9] if boxes.shape[1] > 8 else None
        gx, gy = _agent_goal_from_traj(
            trajs,
            traj_scores,
            i,
            np.array([x, y], dtype=np.float32),
            velocity_xy,
            5.0 if label == 7 else 20.0,
        )
        item = {
            "x": x,
            "y": y,
            "yaw": yaw,
            "speed": speed,
            "goal": [float(gx), float(gy)],
            "score": float(scores[i]),
            "label": label,
            "length": float(boxes[i, 3]) if boxes.shape[1] > 3 else 0.0,
            "width": float(boxes[i, 4]) if boxes.shape[1] > 4 else 0.0,
            "alignment_version": ALIGNMENT_VERSION,
        }
        if label == 7:
            ped_agents.append(item)
        elif label in (0, 1, 2, 3):
            veh_agents.append(item)
    return ped_agents, veh_agents


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _apply_pnn_to_result(adapter, model, result, data, args, score_thr=0.3, debug_limit=0, seen=None):
    core_model = _unwrap_model(model)
    head = core_model.head
    metric_data = _unwrap_batch_data(data)
    dets = []
    for bs, item in enumerate(result):
        det = _get_img_bbox(item)
        dets.append(det)
        hipad_plan = _extract_plan(det)
        lane_points = _extract_lane_points(det, reference_plan=hipad_plan)
        ped_agents, veh_agents = _extract_agents(det, score_thr=score_thr)
        ego_speed = _extract_speed(data, bs)

        if args.pnn_route_source == "navigation":
            navigation_points = _extract_navigation_points(metric_data, bs)
            refined = adapter.refine_navigation_route(
                navigation_points=navigation_points,
                hipad_plan=hipad_plan,
                ego_speed=ego_speed,
                ped_agents=ped_agents,
                veh_agents=veh_agents,
                lane_points=lane_points,
                navigation_min_speed=args.pnn_nav_min_speed,
                navigation_max_speed=args.pnn_nav_max_speed,
                navigation_distance_scale=args.pnn_nav_distance_scale,
                navigation_interpolation=args.pnn_nav_interpolation,
            )
        else:
            refined = adapter.refine_hipad_plan(
                hipad_plan=hipad_plan,
                ego_speed=ego_speed,
                ped_agents=ped_agents,
                veh_agents=veh_agents,
                lane_points=lane_points,
            )
        pnn_plan = torch.as_tensor(refined["final_planning"], dtype=torch.float32)
        det["plan_temp_2hz_hipad"] = det["plan_temp_2hz"]
        det["plan_temp_2hz"] = pnn_plan
        det["plan_pnn_cost_weights"] = torch.as_tensor(refined["cost_weights"], dtype=torch.float32)
        det["plan_pnn_input_agents"] = {
            "score_threshold": float(score_thr),
            "pedestrians": ped_agents,
            "vehicles": veh_agents,
        }
        det["plan_pnn_lane_boundaries"] = torch.as_tensor(lane_points[:2], dtype=torch.float32)

        if seen is not None and (debug_limit <= 0 or seen[0] < debug_limit):
            print(
                "[PNN openloop] "
                f"sample={seen[0]} speed={ego_speed:.3f} "
                f"agents(ped/veh)={len(ped_agents)}/{len(veh_agents)} "
                f"hipad_3s={hipad_plan[-1].tolist()} pnn_3s={refined['final_planning'][-1].tolist()} "
                f"weights={np.round(refined['cost_weights'], 3).tolist()}",
                flush=True,
            )
            seen[0] += 1
    if hasattr(head, "compute_planner_metric_stp3"):
        for bs, item in enumerate(result):
            item["metric_results"] = head.compute_planner_metric_stp3(bs, dets, metric_data)
    if args.pnn_planning_only:
        compact = []
        for item in result:
            entry = {"metric_results": item["metric_results"]}
            if args.pnn_keep_plans:
                det = _get_img_bbox(item)
                entry["img_bbox"] = {
                    "plan_temp_2hz": det.get("plan_temp_2hz"),
                    "plan_temp_2hz_hipad": det.get("plan_temp_2hz_hipad"),
                    "plan_pnn_cost_weights": det.get("plan_pnn_cost_weights"),
                    "plan_pnn_input_agents": det.get("plan_pnn_input_agents"),
                    "plan_pnn_lane_boundaries": det.get("plan_pnn_lane_boundaries"),
                }
            compact.append(entry)
        result = compact
    return result


def print_planning_metrics(outputs):
    if not outputs or "metric_results" not in outputs[0]:
        print("[PNN openloop] no metric_results found; cannot print planning metrics")
        return {}

    print("-------------- Planning --------------")
    metric_sum = None
    num_valid = 0
    for res in outputs:
        metric_results = res["metric_results"]
        if not metric_results.get("fut_valid_flag", False):
            continue
        num_valid += 1
        if metric_sum is None:
            metric_sum = copy.deepcopy(metric_results)
        else:
            for k, v in metric_results.items():
                metric_sum[k] += v

    if metric_sum is None or num_valid == 0:
        print("[PNN openloop] no valid future GT samples")
        return {}

    metric_dict = {}
    for k, v in metric_sum.items():
        metric_dict[k] = v / num_valid

    for k in metric_dict:
        if "plan_L2" in k:
            print("{}: {:.4f}".format(k, metric_dict[k]))
        if "plan_obj_box_col" in k:
            print("{}: {:.4f} % ".format(k, metric_dict[k] * 100))
        if "plan_lane_edge_col" in k:
            print("{}: {:.4f} % ".format(k, metric_dict[k] * 100))
        if "plan_comfort_score" in k:
            print("{}: {:.4f} % ".format(k, metric_dict[k] * 100))

    comparison_keys = (
        "plan_recomputed_masked_obj_box_col_{}s",
        "plan_recomputed_masked_lane_edge_col_{}s",
        "hipad_plan_obj_box_col_{}s",
        "hipad_plan_recomputed_masked_obj_box_col_{}s",
        "hipad_plan_lane_edge_col_{}s",
        "hipad_plan_recomputed_masked_lane_edge_col_{}s",
    )
    if all(pattern.format(sec) in metric_dict for pattern in comparison_keys for sec in (1, 2, 3)):
        print("------ PNN / HiPAD collision comparison ------")
        for sec in (1, 2, 3):
            values = {
                "pnn_obj_raw": metric_dict[f"plan_obj_box_col_{sec}s"],
                "pnn_obj_masked": metric_dict[f"plan_recomputed_masked_obj_box_col_{sec}s"],
                "hipad_obj_raw": metric_dict[f"hipad_plan_obj_box_col_{sec}s"],
                "hipad_obj_masked": metric_dict[f"hipad_plan_recomputed_masked_obj_box_col_{sec}s"],
                "pnn_lane_raw": metric_dict[f"plan_lane_edge_col_{sec}s"],
                "pnn_lane_masked": metric_dict[f"plan_recomputed_masked_lane_edge_col_{sec}s"],
                "hipad_lane_raw": metric_dict[f"hipad_plan_lane_edge_col_{sec}s"],
                "hipad_lane_masked": metric_dict[f"hipad_plan_recomputed_masked_lane_edge_col_{sec}s"],
            }
            print(
                f"{sec}s obj raw/masked PNN={values['pnn_obj_raw'] * 100:.4f}%/"
                f"{values['pnn_obj_masked'] * 100:.4f}% "
                f"HiPAD={values['hipad_obj_raw'] * 100:.4f}%/"
                f"{values['hipad_obj_masked'] * 100:.4f}%"
            )
            print(
                f"{sec}s lane raw/masked PNN={values['pnn_lane_raw'] * 100:.4f}%/"
                f"{values['pnn_lane_masked'] * 100:.4f}% "
                f"HiPAD={values['hipad_lane_raw'] * 100:.4f}%/"
                f"{values['hipad_lane_masked'] * 100:.4f}%"
            )
    comfort_raw_keys = [
        "plan_max_abs_lon_accel_3s",
        "plan_max_abs_lat_accel_3s",
        "plan_max_abs_jerk_3s",
        "plan_max_abs_lon_jerk_3s",
        "plan_max_abs_yaw_rate_3s",
        "plan_max_abs_yaw_accel_3s",
    ]
    for k in comfort_raw_keys:
        if k in metric_dict:
            print("{}: {:.4f}".format(k, metric_dict[k]))
    print(f"[PNN openloop] valid planning samples: {num_valid}/{len(outputs)}")
    return metric_dict


def build_pnn_adapter(args):
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    control_ckpt = args.pnn_control_ckpt or args.pnn_ckpt
    weight_ckpt = args.pnn_weight_ckpt or args.pnn_ckpt
    cfg = PNNAdapterConfig(
        stats_path=args.pnn_stats,
        control_ckpt_path=control_ckpt,
        weight_ckpt_path=weight_ckpt,
        device=device,
        use_weight_net=not args.pnn_no_weight,
        use_theseus_refine=args.pnn_use_theseus,
        planner_max_iterations=args.pnn_planner_max_iterations,
        planner_step_size=float(os.environ.get("PNN_PLANNER_STEP_SIZE", "0.30")),
        planner_ped_safety_distance=float(
            os.environ.get("PNN_PLANNER_PED_SAFETY_DISTANCE", "2.5")
        ),
        planner_veh_safety_distance=float(
            os.environ.get("PNN_PLANNER_VEH_SAFETY_DISTANCE", "4.0")
        ),
        planner_ped_lateral_safety_distance=float(
            os.environ.get("PNN_PLANNER_PED_LATERAL_SAFETY_DISTANCE", "1.2")
        ),
        planner_veh_lateral_safety_distance=float(
            os.environ.get("PNN_PLANNER_VEH_LATERAL_SAFETY_DISTANCE", "1.8")
        ),
        planner_control_anchor_weight=float(
            os.environ.get("PNN_PLANNER_CONTROL_ANCHOR_WEIGHT", "500.0")
        ),
        planner_control_anchor_risk_floor=float(
            os.environ.get("PNN_PLANNER_CONTROL_ANCHOR_RISK_FLOOR", "0.05")
        ),
        weight_temperature=0.7,
        weight_initial_refine_gate=float(
            os.environ.get("PNN_WEIGHT_INITIAL_REFINE_GATE", "0.01")
        ),
        collision_dist=float(os.environ.get("PNN_COLLISION_DIST", "3.0")),
        prior_dense_gain=float(os.environ.get("PNN_PRIOR_DENSE_GAIN", "0.5")),
        prior_turn_gain=float(os.environ.get("PNN_PRIOR_TURN_GAIN", "0.8")),
        prior_high_speed_gain=float(os.environ.get("PNN_PRIOR_HIGH_SPEED_GAIN", "0.3")),
        default_cost_weights=tuple(
            float(x) for x in os.environ.get(
                "PNN_DEFAULT_COST_WEIGHTS", "1.0,2.0,0.6,2.0,3.0,2.0,1.2,10.0"
            ).replace(",", " ").split()
        ),
        weight_delta_max=tuple(
            float(x) for x in os.environ.get(
                "PNN_WEIGHT_DELTA_MAX", "0.4,0.5,0.4,0.5,0.5,0.5,0.6,0.6"
            ).replace(",", " ").split()
        ),
        min_weight=tuple(
            float(x) for x in os.environ.get(
                "PNN_PLANNER_WEIGHT_MIN", "0.05,0.05,0.02,0.02,0.2,0.2,0.2,0.5"
            ).replace(",", " ").split()
        ),
        max_weight=tuple(
            float(x) for x in os.environ.get(
                "PNN_PLANNER_WEIGHT_MAX", "5,8,4,8,12,10,5,24"
            ).replace(",", " ").split()
        ),
        output_forward_offset=args.pnn_output_forward_offset,
        reference_forward_offset=args.pnn_reference_forward_offset,
        coord_convention=args.pnn_coord_convention,
        stats_quantile_low=args.pnn_stats_quantile_low,
        stats_quantile_high=args.pnn_stats_quantile_high,
        clamp_normalized_inputs=args.pnn_clamp_normalized_inputs,
    )
    return PNNOptimizerAdapter(cfg)


def custom_multi_gpu_test_pnn(model, data_loader, args):
    model.eval()
    bbox_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        progress_count = 0

    adapter = build_pnn_adapter(args)
    time.sleep(2)
    seen = [0]
    for batch_idx, data in enumerate(data_loader):
        if args.pnn_max_batches > 0 and batch_idx >= args.pnn_max_batches:
            break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            result = _apply_pnn_to_result(
                adapter,
                model,
                result,
                data,
                args,
                score_thr=args.pnn_score_thr,
                debug_limit=args.pnn_debug_limit,
                seen=seen if rank == 0 else None,
            )
            batch_size = len(result)
            bbox_results.extend(result)

        if rank == 0:
            progress_step = min(batch_size * world_size, len(dataset) - progress_count)
            for _ in range(progress_step):
                prog_bar.update()
            progress_count += progress_step

    collect_size = len(dataset)
    if args.pnn_max_batches > 0:
        collect_size = min(len(dataset), len(bbox_results) * world_size)
    if args.gpu_collect:
        return collect_results_gpu(bbox_results, collect_size)
    return collect_results_cpu(bbox_results, collect_size, args.tmpdir)


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.result_file
    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        plugin_dir = cfg.plugin_dir
        module_dir = os.path.dirname(plugin_dir).split("/")
        module_path = module_dir[0]
        for m in module_dir[1:]:
            module_path += "." + m
        print(module_path)
        importlib.import_module(module_path)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.data.test.work_dir = cfg.work_dir
    print("work_dir: ", cfg.work_dir)
    print("[PNN openloop] ckpt:", args.pnn_ckpt)
    print("[PNN openloop] control ckpt:", args.pnn_control_ckpt or args.pnn_ckpt)
    print("[PNN openloop] weight enabled:", not args.pnn_no_weight)
    if not args.pnn_no_weight:
        print("[PNN openloop] weight ckpt:", args.pnn_weight_ckpt or args.pnn_ckpt)
    print("[PNN openloop] stats:", args.pnn_stats)
    print("[PNN openloop] planning only:", args.pnn_planning_only)

    dataset = build_dataset(cfg.data.test)
    print("distributed:", distributed)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=dict(type="DistributedSampler"),
    )

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if args.result_file is not None:
        outputs = mmcv.load(args.result_file)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = custom_multi_gpu_test_pnn(model, data_loader, args)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        if args.pnn_planning_only:
            print_planning_metrics(outputs)
            return
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        elif args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(eval_kwargs)
            results_dict = dataset.evaluate(outputs, **eval_kwargs)
            print(results_dict)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
