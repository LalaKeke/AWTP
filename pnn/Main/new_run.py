import os
import sys
import time
import pickle
import warnings
from typing import Dict, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


sys.path.append("../")

import nnc.controllers.baselines.dynamics_v4 as dynamics
from nnc.controllers.neural_network.nnc_controllers import NeuralNetworkController, NNCDynamics
from PCC_helpers_v8 import normalize, inverse_normalize, EluTimeControlEnhanced
from LaneBoundaryLagrangianLoss_dual_final import (
    LaneBoundaryLagrangianLoss,
    SafetyConstraintLoss,
    SoftConstraintLambdas,
)


# ============================================================
# 0. 配置区
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_PATH = os.environ.get(
    "PNN_CONTROL_CKPT", os.path.join(PROJECT_ROOT, "checkpoints", "pnn_control.pth")
)
#FILE_PATH = "data/simulated_dataset_0424_mutilane_6019.pt"
FILE_PATH = os.environ.get("PNN_EVAL_DATA", os.path.join(PROJECT_ROOT, "data", "pnn", "eval.pt"))

# 必须和训练时 old_train_data_path 保持一致。
# 如果你训练 v13 用的是 28130_pred_trainv2.pt，这里也改成 v2。
STATS_DATA_PATH = os.environ.get(
    "PNN_STATS_PATH", os.path.join(PROJECT_ROOT, "checkpoints", "pnn_stats.pt")
)
GPU_ID = 5 
DT = 0.1
WHEEL_BASE = 2.588

SAVE_BASE_DIR = "result_view_new"
EVAL_LIMIT = None
SAVE_IMAGES = False

# 如果后续要用 SparseDrive 官方 planning_eval，可以直接拿这个 pkl 去替换 results.pkl。
SAVE_SPARSEDRIVE_STYLE = True


# ============================================================
# 1. 工具函数
# ============================================================

def get_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > gpu_id:
        return torch.device(f"cuda:{gpu_id}")

    if torch.cuda.is_available():
        warnings.warn(
            f"当前机器只有 {torch.cuda.device_count()} 张 GPU，无法使用 cuda:{gpu_id}，自动切换到 cuda:0。"
        )
        return torch.device("cuda:0")

    warnings.warn("未检测到 CUDA，自动使用 CPU。")
    return torch.device("cpu")


def safe_torch_load(path: str, map_location):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def remove_prefix_from_state_dict(state_dict, prefix: str):
    return {
        k[len(prefix):] if isinstance(k, str) and k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def sanitize_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError(f"state_dict 类型错误: {type(state_dict)}")

    prefixes = [
        "module.",
        "model.",
        "net.",
        "network.",
        "neural_net.",
        "module.model.",
        "module.net.",
        "module.network.",
        "module.neural_net.",
    ]

    candidates = [state_dict]
    for prefix in prefixes:
        candidates.append(remove_prefix_from_state_dict(state_dict, prefix))
    return candidates


def state_dict_overlap_score(candidate, target_keys):
    return len(set(candidate.keys()).intersection(target_keys))


def extract_neural_net_state_dict(ckpt, neural_net: torch.nn.Module):
    target_keys = set(neural_net.state_dict().keys())
    direct_keys = [
        "neural_net",
        "neural_net_state_dict",
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "network",
    ]

    candidates = []
    if isinstance(ckpt, dict):
        for key in direct_keys:
            if key in ckpt and isinstance(ckpt[key], dict):
                candidates.extend(sanitize_state_dict(ckpt[key]))
        candidates.extend(sanitize_state_dict(ckpt))

    best_candidate = None
    best_score = 0
    for candidate in candidates:
        score = state_dict_overlap_score(candidate, target_keys)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None or best_score == 0:
        if isinstance(ckpt, dict):
            raise KeyError(
                "无法从 checkpoint 中找到可匹配 EluTimeControlEnhanced 的权重。\n"
                f"checkpoint 顶层 keys: {list(ckpt.keys())[:50]}"
            )
        raise TypeError(f"checkpoint 类型异常: {type(ckpt)}")

    return best_candidate


def load_optional_module(ckpt, key: str, module: torch.nn.Module):
    if not isinstance(ckpt, dict) or key not in ckpt:
        print(f"checkpoint 中没有 {key}，跳过该模块加载。")
        return

    try:
        candidates = sanitize_state_dict(ckpt[key])
        target_keys = set(module.state_dict().keys())
        best_candidate = None
        best_score = 0

        for candidate in candidates:
            score = state_dict_overlap_score(candidate, target_keys)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None or best_score == 0:
            print(f"{key} 的 state_dict 与当前模块不匹配，跳过。")
            return

        missing, unexpected = module.load_state_dict(best_candidate, strict=False)
        if missing:
            print(f"{key} 缺失参数: {missing}")
        if unexpected:
            print(f"{key} 多余参数: {unexpected}")
        print(f"已加载 {key}")
    except Exception as exc:
        print(f"加载 {key} 失败，已跳过。原因: {exc}")


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def make_save_dir(ckpt_path: str, save_base_dir: str) -> str:
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path))) or "unknown_run"
    save_dir = os.path.join(save_base_dir, f"test_{run_dir}_{ckpt_name}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def masked_agent_minmax(
    states: torch.Tensor,
    mask: Optional[torch.Tensor],
    feature_dim: int = 6,
):
    states = states.reshape(-1, feature_dim)
    if mask is not None:
        valid = mask.reshape(-1).bool()
        if valid.any():
            states = states[valid]
    return states.min(0).values, states.max(0).values


@torch.no_grad()
def build_stats_from_data(data: Dict[str, torch.Tensor], device: torch.device):
    min_ped, max_ped = masked_agent_minmax(data["ped_states"], data.get("ped_mask"))
    min_veh, max_veh = masked_agent_minmax(data["veh_states"], data.get("veh_mask"))
    return {
        "min_ego": data["ego_state"].min(0).values.to(device),
        "max_ego": data["ego_state"].max(0).values.to(device),
        "min_ped": min_ped.to(device),
        "max_ped": max_ped.to(device),
        "min_veh": min_veh.to(device),
        "max_veh": max_veh.to(device),
        "min_lane": data["lane_points"][:, 0:2].reshape(-1, 2).min(0).values.to(device),
        "max_lane": data["lane_points"][:, 0:2].reshape(-1, 2).max(0).values.to(device),
    }


def validate_dynamic_parameters(dp: dict):
    required_keys = [
        "max_acceleration",
        "max_jerk",
        "max_lateral_acceleration",
    ]
    for key in required_keys:
        if key not in dp:
            raise KeyError(f"dynamic_parameters 缺少字段: {key}。当前字段为: {list(dp.keys())}")


def compute_dynamic_parameters_from_traj(ego_traj: torch.Tensor, dt: float = DT):
    traj = ego_traj.detach().float()
    v = traj[:, 3]

    if v.numel() < 2:
        return {
            "max_acceleration": 0.0,
            "max_jerk": 0.0,
            "max_lateral_acceleration": 0.0,
        }

    acc = torch.diff(v, dim=0) / dt
    max_acc = acc.abs().max().item()

    if acc.numel() >= 2:
        jerk = torch.diff(acc, dim=0) / dt
        max_jerk = jerk.abs().max().item()
    else:
        max_jerk = 0.0

    if traj.shape[0] >= 2:
        theta = traj[:, 2]
        yaw_rate = torch.diff(theta, dim=0)
        yaw_rate = torch.atan2(torch.sin(yaw_rate), torch.cos(yaw_rate)) / dt
        v_mid = v[1:]
        lat_acc = v_mid * yaw_rate
        max_latacc = lat_acc.abs().max().item()
    else:
        max_latacc = 0.0

    return {
        "max_acceleration": max_acc,
        "max_jerk": max_jerk,
        "max_lateral_acceleration": max_latacc,
    }


def rollout_all_agents_like_training(
    dynamics_model,
    ego_state: torch.Tensor,
    ped_states: torch.Tensor,
    veh_states: torch.Tensor,
    u_ego: torch.Tensor,
    u_peds: torch.Tensor,
    u_vehs: torch.Tensor,
    n_step: Optional[int] = None,
    dt: float = DT,
):
    """Roll out ego, pedestrian, and vehicle states exactly in the lower-branch training style.

    State convention: [x, y, theta, v]
    Control convention: [acceleration, steering]

    The returned trajectory contains predicted states after applying controls.
    Therefore traj[:, 0] corresponds to the state after u[:, 0], not the initial state.
    """
    if n_step is None:
        n_step = int(u_ego.shape[1])

    B = ego_state.size(0)
    Np = ped_states.size(1)
    Nv = veh_states.size(1)

    ego = ego_state[:, :4]
    ped = ped_states[:, :, :4]
    veh = veh_states[:, :, :4]

    ego_list = []
    ped_list = []
    veh_list = []

    for t in range(n_step):
        dx_ego = dynamics_model(ego, u_ego[:, t])
        ego = ego + dt * dx_ego

        if Np > 0:
            dx_ped = dynamics_model(
                ped.reshape(B * Np, 4),
                u_peds[:, :, t].reshape(B * Np, 2),
            )
            ped = ped + dt * dx_ped.view(B, Np, 4)

        if Nv > 0:
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

        if Np > 0:
            ped = torch.stack(
                [
                    ped[:, :, 0],
                    ped[:, :, 1],
                    torch.atan2(torch.sin(ped[:, :, 2]), torch.cos(ped[:, :, 2])),
                    torch.clamp(ped[:, :, 3], min=0.0),
                ],
                dim=2,
            )

        if Nv > 0:
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

    if Np > 0:
        ped_traj = torch.stack(ped_list, dim=2)
    else:
        ped_traj = ped_states.new_zeros(B, 0, n_step, 4)

    if Nv > 0:
        veh_traj = torch.stack(veh_list, dim=2)
    else:
        veh_traj = veh_states.new_zeros(B, 0, n_step, 4)

    return ego_traj, ped_traj, veh_traj


# ============================================================
# 2. 推理与可视化
# ============================================================

def visualize_results(
    data,
    nnc_dyn,
    stats,
    index=0,
    save_path=None,
    device=None,
    dt=0.1,
    wheel_base=4.508,
):
    if device is None:
        device = next(nnc_dyn.parameters()).device

    ego_state_t = data["ego_state"][index:index + 1].to(device).float()
    ped_states_t = data["ped_states"][index:index + 1].to(device).float()
    veh_states_t = data["veh_states"][index:index + 1].to(device).float()
    lane_points_t = data["lane_points"][index:index + 1].to(device).float()

    ped_mask_t = data["ped_mask"][index:index + 1].to(device).bool()
    veh_mask_t = data["veh_mask"][index:index + 1].to(device).bool()

    ego_state = to_numpy(ego_state_t[0])
    ped_states = to_numpy(ped_states_t[0])
    veh_states = to_numpy(veh_states_t[0])
    lane_points = to_numpy(lane_points_t[0, 0:2])
    ped_mask_np = to_numpy(ped_mask_t[0]).astype(bool)
    veh_mask_np = to_numpy(veh_mask_t[0]).astype(bool)

    ego_state_normalized = normalize(ego_state_t, stats["min_ego"], stats["max_ego"])
    ped_states_normalized = normalize(ped_states_t, stats["min_ped"], stats["max_ped"])
    veh_states_normalized = normalize(veh_states_t, stats["min_veh"], stats["max_veh"])
    lane_points_normalized = normalize(
        lane_points_t[:, 0:2].reshape(1, -1, 2),
        stats["min_lane"],
        stats["max_lane"],
    ).reshape(1, 2, lane_points_t.shape[2], 2)

    # 训练时传给模型的是 padding mask，即 True 表示无效位置。
    ped_padding_mask = ~ped_mask_t.bool()
    veh_padding_mask = ~veh_mask_t.bool()

    with torch.no_grad():
        u_ego, u_peds, u_vehs = nnc_dyn(
            ego_state_normalized,
            ped_states_normalized,
            veh_states_normalized,
            lane_points_normalized,
            ped_padding_mask,
            veh_padding_mask,
        )

    n_peds = ped_states.shape[0]
    n_vehs = veh_states.shape[0]

    a_ego = torch.tensor([-10.0, -1.066], device=u_ego.device).view(1, 1, 2)
    b_ego = torch.tensor([10.0, 1.066], device=u_ego.device).view(1, 1, 2)
    a_peds = torch.tensor([-1.0, -np.pi / 4], device=u_peds.device).view(1, 1, 1, 2)
    b_peds = torch.tensor([1.0, np.pi / 4], device=u_peds.device).view(1, 1, 1, 2)
    a_vehs = torch.tensor([-10.0, -1.066], device=u_vehs.device).view(1, 1, 1, 2)
    b_vehs = torch.tensor([10.0, 1.066], device=u_vehs.device).view(1, 1, 1, 2)

    u_ego = inverse_normalize(u_ego, a_ego, b_ego)
    u_peds = inverse_normalize(u_peds, a_peds, b_peds)
    u_vehs = inverse_normalize(u_vehs, a_vehs, b_vehs)

    horizon = u_ego.shape[1]
    model = dynamics.BicycleModel(
        torch.tensor(wheel_base, device=u_ego.device, dtype=u_ego.dtype),
        dt=dt,
    )

    ego_traj, ped_traj, veh_traj = rollout_all_agents_like_training(
        dynamics_model=model,
        ego_state=ego_state_t,
        ped_states=ped_states_t,
        veh_states=veh_states_t,
        u_ego=u_ego,
        u_peds=u_peds,
        u_vehs=u_vehs,
        n_step=horizon,
        dt=dt,
    )

    dynamic_para = compute_dynamic_parameters_from_traj(ego_traj[0], dt=dt)
    timestamp = (np.arange(horizon) + 1) * dt

    trajectory_dict = {
        "ego": {
            "trajectory": to_numpy(ego_traj[0]),
            "position": to_numpy(ego_traj[0, :, :2]),
            "timestamp": timestamp,
        },
        "pedestrians": [],
        "vehicles": [],
    }

    for i in range(n_peds):
        trajectory_dict["pedestrians"].append(
            {
                "id": i,
                "trajectory": to_numpy(ped_traj[0, i]),
                "position": to_numpy(ped_traj[0, i, :, :2]),
                "timestamp": timestamp,
                "valid": bool(ped_mask_np[i]),
            }
        )

    for i in range(n_vehs):
        trajectory_dict["vehicles"].append(
            {
                "id": i,
                "trajectory": to_numpy(veh_traj[0, i]),
                "position": to_numpy(veh_traj[0, i, :, :2]),
                "timestamp": timestamp,
                "valid": bool(veh_mask_np[i]),
            }
        )

    if save_path is not None:
        plot_sample(
            data=data,
            index=index,
            lane_points=lane_points,
            ego_state=ego_state,
            ped_states=ped_states,
            veh_states=veh_states,
            ped_mask_np=ped_mask_np,
            veh_mask_np=veh_mask_np,
            ego_traj=ego_traj,
            ped_traj=ped_traj,
            veh_traj=veh_traj,
            save_path=save_path,
        )

    return {
        "dynamic_parameters": dynamic_para,
        "trajectories": trajectory_dict,
    }


def plot_sample(
    data,
    index,
    lane_points,
    ego_state,
    ped_states,
    veh_states,
    ped_mask_np,
    veh_mask_np,
    ego_traj,
    ped_traj,
    veh_traj,
    save_path,
):
    fig, ax = plt.subplots(figsize=(10, 10))

    b0 = torch.as_tensor(lane_points[0], dtype=torch.float32)
    b1 = torch.as_tensor(lane_points[1], dtype=torch.float32)
    center_line = (b0 + b1) / 2.0
    center_dir = center_line[1] - center_line[0]
    center_dir = center_dir / (torch.norm(center_dir) + 1e-6)

    def cross2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    side0 = cross2d(center_dir, b0[0] - center_line[0])
    left_boundary, right_boundary = (b0, b1) if side0 > 0 else (b1, b0)

    ax.plot(to_numpy(left_boundary[:, 0]), to_numpy(left_boundary[:, 1]), "-", color="#00008B", label="Left Boundary")
    ax.plot(to_numpy(right_boundary[:, 0]), to_numpy(right_boundary[:, 1]), "-", color="#00008B", label="Right Boundary")

    if "lane_points" in data and data["lane_points"].shape[1] > 2:
        for line in range(2, min(10, data["lane_points"].shape[1])):
            pts = to_numpy(data["lane_points"][index, line])
            ax.plot(pts[:, 0], pts[:, 1], "-", color="#00008B")

    ego_xy = to_numpy(ego_traj[0, :, :2])
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], color="blue", linewidth=2, label="Ego Traj")
    ax.scatter(ego_state[0], ego_state[1], color="blue", s=100, marker="o", label="Ego Start")

    for tx, ty in [
        (ego_state[4], ego_state[5]),
        (ego_state[6], ego_state[7]),
        (ego_state[8], ego_state[9]),
    ]:
        ax.scatter(tx, ty, color="blue", s=60, marker="x")

    for i in range(ped_states.shape[0]):
        if ped_mask_np[i]:
            ax.scatter(ped_states[i, 0], ped_states[i, 1], color="green", s=80, marker="^",
                       label="Pedestrian Start" if i == 0 else None)
            ax.plot(to_numpy(ped_traj[0, i, :, 0]), to_numpy(ped_traj[0, i, :, 1]), color="green", linewidth=1,
                    label="Pedestrian Trajectory" if i == 0 else None)

    for i in range(veh_states.shape[0]):
        if veh_mask_np[i]:
            ax.scatter(veh_states[i, 0], veh_states[i, 1], color="red", s=80, marker="s",
                       label="Vehicle Start" if i == 0 else None)
            ax.plot(to_numpy(veh_traj[0, i, :, 0]), to_numpy(veh_traj[0, i, :, 1]), color="red", linewidth=1,
                    label="Vehicle Trajectory" if i == 0 else None)

    ax.legend()
    ax.set_title(f"Sample #{index} Visualization")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(False)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.axis("equal")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. 主流程
# ============================================================

def main():
    device = get_device(GPU_ID)
    print(f"使用设备: {device}")

    save_dir = make_save_dir(CKPT_PATH, SAVE_BASE_DIR)
    print(f"结果保存目录: {save_dir}")

    print(f"加载模型: {CKPT_PATH}")
    checkpoint = safe_torch_load(CKPT_PATH, map_location=device)

    neural_net = EluTimeControlEnhanced(
        embed_dim=128,
        num_heads=4,
        future_steps=30,
    ).to(device)

    neural_state_dict = extract_neural_net_state_dict(checkpoint, neural_net)
    neural_net.load_state_dict(neural_state_dict, strict=True)
    neural_net.eval()
    print("neural_net 已严格加载。")

    lane_loss_module = LaneBoundaryLagrangianLoss(init_lambda=10.0).to(device)
    safety_loss_module = SafetyConstraintLoss(init_lambda=10.0).to(device)
    soft_lambda_module = SoftConstraintLambdas().to(device)

    load_optional_module(checkpoint, "lane_loss_module", lane_loss_module)
    load_optional_module(checkpoint, "safety_loss_module", safety_loss_module)
    load_optional_module(checkpoint, "soft_lambda_module", soft_lambda_module)

    linear_dynamics = dynamics.BicycleModel(torch.tensor(WHEEL_BASE, device=device), DT)
    nnc_dyn = NNCDynamics(linear_dynamics, NeuralNetworkController(neural_net)).to(device)
    nnc_dyn.eval()

    print(f"加载测试数据: {FILE_PATH}")
    data = safe_torch_load(FILE_PATH, map_location="cpu")

    print(f"加载训练 stats 数据: {STATS_DATA_PATH}")
    stats_data = safe_torch_load(STATS_DATA_PATH, map_location="cpu")
    stats = build_stats_from_data(stats_data, device)

    required_data_keys = [
        "ego_state",
        "ped_states",
        "veh_states",
        "lane_points",
        "ped_mask",
        "veh_mask",
    ]
    for key in required_data_keys:
        if key not in data:
            raise KeyError(f"测试数据缺少字段: {key}")

    total_num = data["ego_state"].size(0)
    eval_num = total_num if EVAL_LIMIT is None else min(int(EVAL_LIMIT), total_num)
    print(f"测试样本数量: {eval_num} / {total_num}")

    max_acc = 0.0
    max_jerk = 0.0
    max_latacc = 0.0
    sum_acc = 0.0
    sum_jerk = 0.0
    sum_latacc = 0.0

    all_results = []
    sparsedrive_results = []

    print("开始逐样本测试与可视化...")

    for i in tqdm(range(eval_num), desc="Evaluating"):
        save_path = os.path.join(save_dir, f"sample_{i}.png") if SAVE_IMAGES else None

        result = visualize_results(
            data=data,
            nnc_dyn=nnc_dyn,
            stats=stats,
            index=i,
            save_path=save_path,
            device=device,
            dt=DT,
            wheel_base=WHEEL_BASE,
        )

        all_results.append(result)

        if SAVE_SPARSEDRIVE_STYLE:
            ego_pos = torch.as_tensor(result["trajectories"]["ego"]["position"], dtype=torch.float32)
            final_planning = ego_pos[[4, 9, 14, 19, 24, 29], :2]
            sparsedrive_results.append({"img_bbox": {"final_planning": final_planning}})

        dp = result["dynamic_parameters"]
        validate_dynamic_parameters(dp)

        sample_max_acc = float(dp["max_acceleration"])
        sample_max_jerk = float(dp["max_jerk"])
        sample_max_latacc = float(dp["max_lateral_acceleration"])

        max_acc = max(max_acc, sample_max_acc)
        max_jerk = max(max_jerk, sample_max_jerk)
        max_latacc = max(max_latacc, sample_max_latacc)

        sum_acc += sample_max_acc
        sum_jerk += sample_max_jerk
        sum_latacc += sample_max_latacc

    avg_acc = sum_acc / max(eval_num, 1)
    avg_jerk = sum_jerk / max(eval_num, 1)
    avg_latacc = sum_latacc / max(eval_num, 1)

    summary = {
        "eval_num": eval_num,
        "total_num": total_num,
        "max_acceleration": max_acc,
        "max_jerk": max_jerk,
        "max_lateral_acceleration": max_latacc,
        "average_max_acceleration": avg_acc,
        "average_max_jerk": avg_jerk,
        "average_max_lateral_acceleration": avg_latacc,
        "ckpt_path": CKPT_PATH,
        "file_path": FILE_PATH,
        "stats_data_path": STATS_DATA_PATH,
        "device": str(device),
        "dt": DT,
        "wheel_base": WHEEL_BASE,
        "save_images": SAVE_IMAGES,
        "mask_mode": "padding_mask = ~valid_mask",
    }

    print("\n===== 预测轨迹动态参数统计 =====")
    print(f"最大加速度: {max_acc:.4f} m/s^2")
    print(f"最大加加速度: {max_jerk:.4f} m/s^3")
    print(f"最大横向加速度: {max_latacc:.4f} m/s^2")
    print(f"平均最大加速度: {avg_acc:.4f} m/s^2")
    print(f"平均最大加加速度: {avg_jerk:.4f} m/s^3")
    print(f"平均最大横向加速度: {avg_latacc:.4f} m/s^2")

    date_suffix = time.strftime("%Y%m%d")
    results_pkl_path = os.path.join(save_dir, f"all_results_{date_suffix}.pkl")
    summary_pkl_path = os.path.join(save_dir, f"summary_{date_suffix}.pkl")
    sparsedrive_pkl_path = os.path.join(save_dir, f"sparsedrive_style_results_{date_suffix}0513_0_noad_new_test.pkl")

    with open(results_pkl_path, "wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(summary_pkl_path, "wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n{eval_num} 个逐样本结果已保存到: {results_pkl_path}")
    print(f"统计摘要已保存到: {summary_pkl_path}")

    if SAVE_SPARSEDRIVE_STYLE:
        with open(sparsedrive_pkl_path, "wb") as f:
            pickle.dump(sparsedrive_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"SparseDrive 风格 final_planning 结果已保存到: {sparsedrive_pkl_path}")


if __name__ == "__main__":
    main()
