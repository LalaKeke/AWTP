#!/usr/bin/env python3
"""Run PNN_EV_v12 train_v10 on converted HiP-AD/Bench2Drive tensors.

This file is a thin wrapper so the original PNN project code does not need to
be edited. Configure it through environment variables, for example:

  PNN_GPUS=0 PNN_EPOCHS=1 PNN_BATCH_SIZE=4 python run_pnn_train_b2d.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("HIPAD_PNN_ROOT", Path(__file__).resolve().parents[2])
).resolve()
PNN_MAIN = str(PROJECT_ROOT / "pnn" / "Main")
DEFAULT_OLD = str(PROJECT_ROOT / "data" / "pnn" / "train_old.pt")
DEFAULT_NEW = str(PROJECT_ROOT / "data" / "pnn" / "train_new.pt")
DEFAULT_CONTROL = ""
DEFAULT_SAVE = str(PROJECT_ROOT / "outputs" / "pnn_train")


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float_tuple(name: str, default, length: int = 8):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return tuple(default)
    parts = [p for p in value.replace(",", " ").split() if p]
    vals = tuple(float(p) for p in parts)
    if len(vals) != length:
        raise ValueError(f"{name} must contain {length} floats, got {len(vals)}: {value!r}")
    return vals


def main() -> None:
    os.chdir(PNN_MAIN)
    sys.path.insert(0, PNN_MAIN)

    gpus = os.environ.get("PNN_GPUS", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # torch 1.x does not recognize newer allocator options such as
    # expandable_segments, so leave PYTORCH_CUDA_ALLOC_CONF to the caller.

    import torch
    import torch.multiprocessing as mp
    import train_v10

    torch.manual_seed(env_int("PNN_SEED", 27))
    world_size = len([gpu for gpu in gpus.split(",") if gpu.strip()])
    if world_size <= 0:
        raise ValueError(f"PNN_GPUS={gpus!r} does not contain any GPU ids")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(train_v10.find_free_port())

    cfg_runtime = {
        "old_train_data_path": os.environ.get("PNN_OLD_DATA", DEFAULT_OLD),
        "new_train_data_path": os.environ.get("PNN_NEW_DATA", DEFAULT_NEW),
        "supervision_data_path": os.environ.get("PNN_SUPERVISION_DATA") or None,
        "coord_convention": os.environ.get("PNN_COORD_CONVENTION") or None,
        "control_ckpt_path": os.environ.get("PNN_CONTROL_CKPT", DEFAULT_CONTROL) or None,
        "teacher_ckpt_path": os.environ.get("PNN_TEACHER_CKPT") or None,
        "save_dir": os.environ.get("PNN_SAVE_DIR", DEFAULT_SAVE),
        "resume_ckpt_path": os.environ.get("PNN_RESUME_CKPT") or None,
        "batch_size": env_int("PNN_BATCH_SIZE", 24),
        "num_workers": env_int("PNN_NUM_WORKERS", 4),
        "epochs": env_int("PNN_EPOCHS", 10),
        "max_train_batches": env_int("PNN_MAX_TRAIN_BATCHES", 0),
        "lr_control": env_float("PNN_LR_CONTROL", 2e-5),
        "lr_weight": env_float("PNN_LR_WEIGHT", 8e-4),
        "resume_optimizer_state": env_bool("PNN_RESUME_OPTIMIZER_STATE", True),
        "override_resume_lr": env_bool("PNN_OVERRIDE_RESUME_LR", False),
        "ema_beta": 0.995,
        "embed_dim": 128,
        "num_heads": 4,
        "weight_temperature": 0.7,
        "weight_initial_refine_gate": env_float("PNN_WEIGHT_INITIAL_REFINE_GATE", 0.01),
        "weightnet_use_prior_context": True,
        "weightnet_prior_context_mode": "log",
        "default_cost_weights": env_float_tuple(
            "PNN_DEFAULT_COST_WEIGHTS",
            (1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 3.0, 2.0),
        ),
        "weight_prior": env_float_tuple(
            "PNN_WEIGHT_PRIOR",
            (1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0),
        ),
        "progress_overshoot_weight": env_float("PNN_PROGRESS_OVERSHOOT_WEIGHT", 0.25),
        "weightnet_outputs_residual": True,
        "weight_delta_max": env_float_tuple(
            "PNN_WEIGHT_DELTA_MAX",
            (1.3, 1.4, 1.1, 1.4, 1.2, 1.2, 1.5, 1.4),
        ),
        "prior_dense_gain": env_float("PNN_PRIOR_DENSE_GAIN", 1.4),
        "prior_turn_gain": env_float("PNN_PRIOR_TURN_GAIN", 1.2),
        "prior_high_speed_gain": env_float("PNN_PRIOR_HIGH_SPEED_GAIN", 0.9),
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
        "collision_dist": env_float("PNN_COLLISION_DIST", 6.0),
        "collision_risk_sharpness": 1.5,
        "lambda_control_pred_target": 0.0,
        "lambda_gt_reference_lane": env_float("PNN_LAMBDA_GT_REFERENCE_LANE", 1.0),
        "lambda_ego_object_safety": env_float("PNN_LAMBDA_EGO_OBJECT_SAFETY", 1.0),
        "metric_safety_margin": env_float("PNN_METRIC_SAFETY_MARGIN", 1.5),
        "metric_safety_topk": env_int("PNN_METRIC_SAFETY_TOPK", 3),
        "metric_safety_smooth_temperature": env_float("PNN_METRIC_SAFETY_TEMPERATURE", 0.0),
        "metric_safety_time_weights": env_float_tuple(
            "PNN_METRIC_SAFETY_TIME_WEIGHTS",
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            length=6,
        ),
        "risk_gate_margin": env_float("PNN_RISK_GATE_MARGIN", 0.0),
        "risk_safety_gain": env_float("PNN_RISK_SAFETY_GAIN", 0.0),
        "use_pnn_only_risk_weighting": env_bool("PNN_USE_PNN_ONLY_RISK_WEIGHTING", False),
        "pnn_only_safety_gain": env_float("PNN_PNN_ONLY_SAFETY_GAIN", 4.0),
        "shared_safety_gain": env_float("PNN_SHARED_SAFETY_GAIN", 2.0),
        "lambda_teacher_trust": env_float("PNN_LAMBDA_TEACHER_TRUST", 0.0),
        "teacher_trust_beta": env_float("PNN_TEACHER_TRUST_BETA", 0.25),
        "teacher_trust_risk_floor": env_float("PNN_TEACHER_TRUST_RISK_FLOOR", 0.1),
        "teacher_trust_uses_pnn_risk": env_bool("PNN_TEACHER_TRUST_USES_PNN_RISK", False),
        "lambda_pnn_only_hipad_anchor": env_float("PNN_LAMBDA_PNN_ONLY_HIPAD_ANCHOR", 0.0),
        "pnn_only_hipad_anchor_beta": env_float("PNN_PNN_ONLY_HIPAD_ANCHOR_BETA", 0.25),
        "pnn_only_hipad_anchor_time_weights": env_float_tuple(
            "PNN_PNN_ONLY_HIPAD_ANCHOR_TIME_WEIGHTS",
            (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
            length=6,
        ),
        "official_frame_acr_enabled": env_bool("PNN_OFFICIAL_FRAME_ACR_ENABLED", False),
        "official_raster_actor_padding": env_float("PNN_OFFICIAL_RASTER_ACTOR_PADDING", 0.12),
        "official_raster_actor_dilation_pixels": env_float(
            "PNN_OFFICIAL_RASTER_ACTOR_DILATION_PIXELS", 0.0
        ),
        "lambda_official_frame_acr": env_float("PNN_LAMBDA_OFFICIAL_FRAME_ACR", 0.0),
        "official_frame_acr_margin": env_float("PNN_OFFICIAL_FRAME_ACR_MARGIN", 1.0),
        "official_frame_acr_temperature": env_float("PNN_OFFICIAL_FRAME_ACR_TEMPERATURE", 0.25),
        "official_frame_acr_time_weights": env_float_tuple(
            "PNN_OFFICIAL_FRAME_ACR_TIME_WEIGHTS",
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            length=6,
        ),
        "official_frame_acr_pnn_only": env_bool(
            "PNN_OFFICIAL_FRAME_ACR_PNN_ONLY", False
        ),
        "official_frame_acr_require_perceived_actor": env_bool(
            "PNN_OFFICIAL_FRAME_ACR_REQUIRE_PERCEIVED_ACTOR", False
        ),
        "perceived_actor_match_radius": env_float(
            "PNN_PERCEIVED_ACTOR_MATCH_RADIUS", 3.0
        ),
        "perceived_actor_collision_margin": env_float(
            "PNN_PERCEIVED_ACTOR_COLLISION_MARGIN", 0.75
        ),
        "lambda_perception_gated_brake": env_float(
            "PNN_LAMBDA_PERCEPTION_GATED_BRAKE", 0.0
        ),
        "perception_brake_min_decel": env_float(
            "PNN_PERCEPTION_BRAKE_MIN_DECEL", 0.5
        ),
        "perception_brake_min_speed": env_float(
            "PNN_PERCEPTION_BRAKE_MIN_SPEED", 1.0
        ),
        "perception_brake_reaction_time": env_float(
            "PNN_PERCEPTION_BRAKE_REACTION_TIME", 0.20
        ),
        "perception_brake_decel_gain": env_float(
            "PNN_PERCEPTION_BRAKE_DECEL_GAIN", 0.60
        ),
        "perception_brake_decay_steps": env_float(
            "PNN_PERCEPTION_BRAKE_DECAY_STEPS", 6.0
        ),
        "perception_brake_time_weights": env_float_tuple(
            "PNN_PERCEPTION_BRAKE_TIME_WEIGHTS",
            (8.0, 8.0, 6.0, 4.0, 2.0, 1.0),
            length=6,
        ),
        "lambda_actuator_feasibility": env_float("PNN_LAMBDA_ACTUATOR_FEASIBILITY", 0.0),
        "actuator_max_accel": env_float("PNN_ACTUATOR_MAX_ACCEL", 2.40),
        "actuator_max_decel": env_float("PNN_ACTUATOR_MAX_DECEL", 4.05),
        "control_decode_min_accel": env_float("PNN_CONTROL_DECODE_MIN_ACCEL", -10.0),
        "control_decode_max_accel": env_float("PNN_CONTROL_DECODE_MAX_ACCEL", 10.0),
        "control_decode_max_steer": env_float("PNN_CONTROL_DECODE_MAX_STEER", 1.066),
        "actuator_max_steer": env_float("PNN_ACTUATOR_MAX_STEER", 1.066),
        "lambda_risk_early_brake": env_float("PNN_LAMBDA_RISK_EARLY_BRAKE", 0.0),
        "risk_early_brake_steps": env_int("PNN_RISK_EARLY_BRAKE_STEPS", 5),
        "risk_early_brake_accel_ceiling": env_float(
            "PNN_RISK_EARLY_BRAKE_ACCEL_CEILING", 0.0
        ),
        "lambda_official_frame_anchor": env_float("PNN_LAMBDA_OFFICIAL_FRAME_ANCHOR", 0.0),
        "official_frame_anchor_beta": env_float("PNN_OFFICIAL_FRAME_ANCHOR_BETA", 0.25),
        "official_frame_anchor_include_previous": env_bool(
            "PNN_OFFICIAL_FRAME_ANCHOR_INCLUDE_PREVIOUS", True
        ),
        "official_frame_anchor_time_weights": env_float_tuple(
            "PNN_OFFICIAL_FRAME_ANCHOR_TIME_WEIGHTS",
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            length=6,
        ),
        "official_frame_anchor_require_lane_safe": env_bool(
            "PNN_OFFICIAL_FRAME_ANCHOR_REQUIRE_LANE_SAFE", False
        ),
        "official_frame_anchor_lane_safe_margin": env_float(
            "PNN_OFFICIAL_FRAME_ANCHOR_LANE_SAFE_MARGIN", 0.0
        ),
        "lambda_official_frame_lane_guard": env_float(
            "PNN_LAMBDA_OFFICIAL_FRAME_LANE_GUARD", 0.0
        ),
        "official_frame_lane_guard_margin": env_float(
            "PNN_OFFICIAL_FRAME_LANE_GUARD_MARGIN", 0.35
        ),
        "official_frame_lane_guard_include_previous": env_bool(
            "PNN_OFFICIAL_FRAME_LANE_GUARD_INCLUDE_PREVIOUS", True
        ),
        "official_frame_lane_guard_include_future": env_bool(
            "PNN_OFFICIAL_FRAME_LANE_GUARD_INCLUDE_FUTURE", False
        ),
        "official_frame_lane_guard_include_lane_hits": env_bool(
            "PNN_OFFICIAL_FRAME_LANE_GUARD_INCLUDE_LANE_HITS", False
        ),
        "official_frame_lane_guard_self_gate_threshold": env_float(
            "PNN_OFFICIAL_FRAME_LANE_GUARD_SELF_GATE_THRESHOLD", 0.0
        ),
        "official_frame_lane_guard_time_weights": env_float_tuple(
            "PNN_OFFICIAL_FRAME_LANE_GUARD_TIME_WEIGHTS",
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            length=6,
        ),
        "lambda_safe_parent_distill": env_float("PNN_LAMBDA_SAFE_PARENT_DISTILL", 0.0),
        "safe_parent_distill_beta": env_float("PNN_SAFE_PARENT_DISTILL_BETA", 0.25),
        "safe_parent_distill_require_current_actor_safe": env_bool(
            "PNN_SAFE_PARENT_DISTILL_REQUIRE_CURRENT_ACTOR_SAFE", True
        ),
        "safe_parent_distill_require_parent_lane_safe": env_bool(
            "PNN_SAFE_PARENT_DISTILL_REQUIRE_PARENT_LANE_SAFE", False
        ),
        "safe_parent_distill_lane_safe_margin": env_float(
            "PNN_SAFE_PARENT_DISTILL_LANE_SAFE_MARGIN", 0.0
        ),
        "lambda_route_speed_excess": env_float("PNN_LAMBDA_ROUTE_SPEED_EXCESS", 0.0),
        "route_speed_margin": env_float("PNN_ROUTE_SPEED_MARGIN", 1.0),
        "route_speed_brake_weight": env_float("PNN_ROUTE_SPEED_BRAKE_WEIGHT", 0.2),
        "route_speed_brake_trigger_margin": env_float("PNN_ROUTE_SPEED_BRAKE_TRIGGER_MARGIN", 1.0),
        "route_speed_positive_accel_threshold": env_float("PNN_ROUTE_SPEED_POS_ACCEL_THRESHOLD", 0.3),
        "ego_track_time_weights": env_float_tuple(
            "PNN_EGO_TRACK_TIME_WEIGHTS",
            (1.8, 2.5, 1.8, 1.2, 0.8, 0.5),
            length=6,
        ),
        "route_track_risk_gate_gain": env_float(
            "PNN_ROUTE_TRACK_RISK_GATE_GAIN", 0.0
        ),
        "route_track_risk_min_weight": env_float(
            "PNN_ROUTE_TRACK_RISK_MIN_WEIGHT", 1.0
        ),
        "route_track_lane_risk_gain": env_float(
            "PNN_ROUTE_TRACK_LANE_RISK_GAIN", 0.0
        ),
        "route_target_aug_probability": env_float(
            "PNN_ROUTE_TARGET_AUG_PROBABILITY", 0.0
        ),
        "route_target_longitudinal_scale_std": env_float(
            "PNN_ROUTE_TARGET_LONGITUDINAL_SCALE_STD", 0.0
        ),
        "route_target_lateral_jitter_std": env_float(
            "PNN_ROUTE_TARGET_LATERAL_JITTER_STD", 0.0
        ),
        "route_target_middle_dropout": env_float(
            "PNN_ROUTE_TARGET_MIDDLE_DROPOUT", 0.0
        ),
        "lane_clearance_loss_weight": env_float("PNN_LAMBDA_LANE_CLEARANCE", 0.0),
        "lane_clearance_margin": env_float("PNN_LANE_CLEARANCE_MARGIN", 0.8),
        "lambda_metric_lane": env_float("PNN_LAMBDA_METRIC_LANE", 0.0),
        "metric_lane_margin": env_float("PNN_METRIC_LANE_MARGIN", 0.05),
        "metric_lane_use_gt_safe_side": env_bool(
            "PNN_METRIC_LANE_USE_GT_SAFE_SIDE", False
        ),
        "metric_lane_time_weights": env_float_tuple(
            "PNN_METRIC_LANE_TIME_WEIGHTS",
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            length=6,
        ),
        "metric_lane_hard_max_weight": env_float("PNN_METRIC_LANE_HARD_MAX_WEIGHT", 0.0),
        "disable_predicted_lane_losses": env_bool("PNN_DISABLE_PREDICTED_LANE_LOSSES", False),
        "lambda_dense_route_speed": env_float("PNN_LAMBDA_DENSE_ROUTE_SPEED", 0.0),
        "dense_route_speed_margin": env_float("PNN_DENSE_ROUTE_SPEED_MARGIN", 0.5),
        "lambda_forward_overshoot": env_float("PNN_LAMBDA_FORWARD_OVERSHOOT", 0.0),
        "forward_overshoot_margin": env_float("PNN_FORWARD_OVERSHOOT_MARGIN", 0.3),
        "forward_overshoot_time_weights": env_float_tuple(
            "PNN_FORWARD_OVERSHOOT_TIME_WEIGHTS",
            (2.0, 1.5, 0.8),
            length=3,
        ),
        "lambda_obstacle_clearance": env_float("PNN_LAMBDA_OBS_CLEARANCE", 0.0),
        "obstacle_clearance_margin_veh": env_float("PNN_OBS_CLEARANCE_MARGIN_VEH", 2.5),
        "obstacle_clearance_margin_ped": env_float("PNN_OBS_CLEARANCE_MARGIN_PED", 1.8),
        "obstacle_clearance_topk": env_int("PNN_OBS_CLEARANCE_TOPK", 3),
        "obstacle_clearance_time_weights": env_float_tuple(
            "PNN_OBS_CLEARANCE_TIME_WEIGHTS",
            (1.5, 1.2, 1.0),
            length=3,
        ),
        "lambda_rollout_comfort": env_float("PNN_LAMBDA_ROLLOUT_COMFORT", 0.0),
        "comfort_acc_threshold": env_float("PNN_COMFORT_ACC_THRESHOLD", 2.40),
        "comfort_min_lon_accel": env_float("PNN_COMFORT_MIN_LON_ACCEL", -4.05),
        "comfort_lat_accel_threshold": env_float("PNN_COMFORT_LAT_ACCEL_THRESHOLD", 4.89),
        "comfort_jerk_threshold": env_float("PNN_COMFORT_JERK_THRESHOLD", 4.13),
        "comfort_yaw_rate_threshold": env_float("PNN_COMFORT_YAW_RATE_THRESHOLD", 0.95),
        "comfort_yaw_accel_threshold": env_float("PNN_COMFORT_YAW_ACCEL_THRESHOLD", 1.93),
        "train_soft_constraint_lambdas": env_bool("PNN_TRAIN_SOFT_CONSTRAINT_LAMBDAS", False),
        "control_weight_start_epoch": env_int("PNN_CONTROL_WEIGHT_START_EPOCH", 6),
        "weight_update_start_epoch": env_int("PNN_WEIGHT_UPDATE_START_EPOCH", 0),
        "weight_update_interval": env_int("PNN_WEIGHT_UPDATE_INTERVAL", 1),
        "weight_supervision_start_epoch": env_int("PNN_WEIGHT_SUPERVISION_START_EPOCH", 0),
        "weight_supervision_ramp_epochs": env_int("PNN_WEIGHT_SUPERVISION_RAMP_EPOCHS", 5),
        "lambda_weight_traj": env_float("PNN_LAMBDA_WEIGHT_TRAJ", 0.35),
        "weight_traj_warmup_epochs": env_int("PNN_WEIGHT_TRAJ_WARMUP_EPOCHS", 2),
        "weight_traj_ramp_epochs": env_int("PNN_WEIGHT_TRAJ_RAMP_EPOCHS", 4),
        "lambda_weight_rule": env_float("PNN_LAMBDA_WEIGHT_RULE", 0.38),
        "lambda_weight_feedback": env_float("PNN_LAMBDA_WEIGHT_FEEDBACK", 0.25),
        "lambda_weight_rank": env_float("PNN_LAMBDA_WEIGHT_RANK", 0.22),
        "lambda_weight_sep": env_float("PNN_LAMBDA_WEIGHT_SEP", 0.10),
        "lambda_weight_extreme": env_float("PNN_LAMBDA_WEIGHT_EXTREME", 0.03),
        "lambda_entropy_band": env_float("PNN_LAMBDA_ENTROPY_BAND", 0.03),
        "lambda_diversity_floor": env_float("PNN_LAMBDA_DIVERSITY_FLOOR", 0.12),
        "entropy_band_low": 1.05,
        "entropy_band_high": 1.92,
        "diversity_floor_pairwise_l2": 0.060,
        "feedback_component_gain": env_float("PNN_FEEDBACK_COMPONENT_GAIN", 0.45),
        "rank_high_risk_th": env_float("PNN_RANK_HIGH_RISK_TH", 0.45),
        "rank_low_risk_th": env_float("PNN_RANK_LOW_RISK_TH", 0.30),
        "rank_margin_safe": env_float("PNN_RANK_MARGIN_SAFE", 0.035),
        "rank_margin_route": env_float("PNN_RANK_MARGIN_ROUTE", 0.030),
        "rank_margin_comfort": env_float("PNN_RANK_MARGIN_COMFORT", 0.025),
        "weight_max_allowed_prob": 0.68,
        "weight_min_allowed_prob": 0.0,
        "weight_sep_base_margin": 0.08,
        "weight_sep_scalar_scale": 0.45,
        "weight_sep_min_risk_gap": 0.06,
        "weak_free_weights": env_float_tuple(
            "PNN_WEAK_FREE_WEIGHTS",
            (2.4, 4.2, 1.1, 4.4, 0.45, 0.45, 7.0, 0.45),
        ),
        "weak_risky_weights": env_float_tuple(
            "PNN_WEAK_RISKY_WEIGHTS",
            (0.35, 2.2, 0.35, 2.2, 4.8, 4.5, 1.2, 8.0),
        ),
        "detach_init_control_for_weight": True,
        "disable_dipp_traj_after_failure": True,
        "planner_optimizer": os.environ.get("PNN_PLANNER_OPTIMIZER", "levenberg_marquardt"),
        "planner_max_iterations": env_int("PNN_PLANNER_MAX_ITERATIONS", 10),
        "planner_step_size": env_float("PNN_PLANNER_STEP_SIZE", 0.10),
        "planner_ped_safety_distance": env_float("PNN_PLANNER_PED_SAFETY_DISTANCE", 2.5),
        "planner_veh_safety_distance": env_float("PNN_PLANNER_VEH_SAFETY_DISTANCE", 4.0),
        "planner_ped_lateral_safety_distance": env_float(
            "PNN_PLANNER_PED_LATERAL_SAFETY_DISTANCE", 1.2
        ),
        "planner_veh_lateral_safety_distance": env_float(
            "PNN_PLANNER_VEH_LATERAL_SAFETY_DISTANCE", 1.8
        ),
        "planner_control_anchor_weight": env_float(
            "PNN_PLANNER_CONTROL_ANCHOR_WEIGHT", 500.0
        ),
        "planner_control_anchor_risk_floor": env_float(
            "PNN_PLANNER_CONTROL_ANCHOR_RISK_FLOOR", 0.05
        ),
        "planner_weight_min": 1e-3,
        "planner_weight_max": 20.0,
        "planner_weight_min_vector": env_float_tuple(
            "PNN_PLANNER_WEIGHT_MIN_VECTOR",
            (0.05, 0.05, 0.02, 0.02, 0.20, 0.20, 0.20, 0.50),
        ),
        "planner_weight_max_vector": env_float_tuple(
            "PNN_PLANNER_WEIGHT_MAX_VECTOR",
            (8.0, 8.0, 5.0, 6.0, 14.0, 14.0, 16.0, 16.0),
        ),
        "prior_renormalize_to_default_sum": False,
        "planner_weight_renormalize_to_default_sum": False,
        "eval_each_epoch": env_bool("PNN_EVAL_EACH_EPOCH", False),
        "train_proxy_metrics": env_bool("PNN_TRAIN_PROXY_METRICS", True),
        "require_gt_actor_boxes": env_bool("PNN_REQUIRE_GT_ACTOR_BOXES", False),
        "require_metric_supervision": env_bool("PNN_REQUIRE_METRIC_SUPERVISION", False),
        "require_solid_lane_supervision": env_bool(
            "PNN_REQUIRE_SOLID_LANE_SUPERVISION", False
        ),
        "require_hipad_plan_2hz": env_bool("PNN_REQUIRE_HIPAD_PLAN_2HZ", False),
        "freeze_control_policy": env_bool("PNN_FREEZE_CONTROL_POLICY", False),
        "control_update_start_epoch": env_int("PNN_CONTROL_UPDATE_START_EPOCH", 0),
        "train_raw_collisions": env_bool("PNN_TRAIN_RAW_COLLISIONS", False),
        "weight_hipad_risk_margin": env_float("PNN_WEIGHT_HIPAD_RISK_MARGIN", 0.0),
        "lambda_weight_hipad_risk": env_float("PNN_LAMBDA_WEIGHT_HIPAD_RISK", 0.0),
        "weight_hipad_risk_positive_weight": env_float("PNN_WEIGHT_HIPAD_RISK_POSITIVE_WEIGHT", 8.0),
        "weight_hipad_safe_log_gain": env_float("PNN_WEIGHT_HIPAD_SAFE_LOG_GAIN", 0.8),
        "weight_hipad_route_log_decay": env_float("PNN_WEIGHT_HIPAD_ROUTE_LOG_DECAY", 0.25),
        "lambda_weight_pnn_collision": env_float("PNN_LAMBDA_WEIGHT_PNN_COLLISION", 0.0),
        "weight_pnn_collision_positive_weight": env_float(
            "PNN_WEIGHT_PNN_COLLISION_POSITIVE_WEIGHT", 12.0
        ),
        "weight_pnn_obj_safe_log_gain": env_float("PNN_WEIGHT_PNN_OBJ_SAFE_LOG_GAIN", 1.0),
        "weight_pnn_lane_xy_log_gain": env_float("PNN_WEIGHT_PNN_LANE_XY_LOG_GAIN", 0.9),
        "weight_pnn_lane_theta_log_gain": env_float(
            "PNN_WEIGHT_PNN_LANE_THETA_LOG_GAIN", 0.7
        ),
        "weight_pnn_comfort_log_decay": env_float(
            "PNN_WEIGHT_PNN_COMFORT_LOG_DECAY", 0.45
        ),
        "weight_pnn_longitudinal_log_decay": env_float(
            "PNN_WEIGHT_PNN_LONGITUDINAL_LOG_DECAY",
            env_float("PNN_WEIGHT_PNN_COMFORT_LOG_DECAY", 0.45),
        ),
        "weight_pnn_lateral_log_decay": env_float(
            "PNN_WEIGHT_PNN_LATERAL_LOG_DECAY",
            env_float("PNN_WEIGHT_PNN_COMFORT_LOG_DECAY", 0.45),
        ),
        "weight_pnn_route_log_decay": env_float("PNN_WEIGHT_PNN_ROUTE_LOG_DECAY", 0.55),
        "weight_application_gate_mode": os.environ.get(
            "PNN_WEIGHT_APPLICATION_GATE_MODE", "none"
        ),
        "weight_pnn_lane_gate_threshold": env_float(
            "PNN_WEIGHT_PNN_LANE_GATE_THRESHOLD", 0.0
        ),
        "lambda_weight_dipp_safety": env_float("PNN_LAMBDA_WEIGHT_DIPP_SAFETY", 0.0),
        "weight_dipp_safety_margin": env_float("PNN_WEIGHT_DIPP_SAFETY_MARGIN", 0.25),
        "weight_dipp_risk_gain": env_float("PNN_WEIGHT_DIPP_RISK_GAIN", 8.0),
        "lambda_weight_dipp_lane": env_float("PNN_LAMBDA_WEIGHT_DIPP_LANE", 0.0),
        "weight_dipp_lane_margin": env_float("PNN_WEIGHT_DIPP_LANE_MARGIN", 0.05),
        "weight_dipp_lane_risk_gain": env_float("PNN_WEIGHT_DIPP_LANE_RISK_GAIN", 8.0),
        "lambda_weight_dipp_trust": env_float("PNN_LAMBDA_WEIGHT_DIPP_TRUST", 0.0),
        "weight_dipp_trust_beta": env_float("PNN_WEIGHT_DIPP_TRUST_BETA", 0.25),
        "weight_dipp_trust_risk_floor": env_float("PNN_WEIGHT_DIPP_TRUST_RISK_FLOOR", 0.1),
        "weight_dipp_start_epoch": env_int("PNN_WEIGHT_DIPP_START_EPOCH", 0),
        "weight_dipp_update_interval": env_int("PNN_WEIGHT_DIPP_UPDATE_INTERVAL", 1),
        "lambda_dipp_control_distill": env_float(
            "PNN_LAMBDA_DIPP_CONTROL_DISTILL", 0.0
        ),
        "dipp_control_distill_start_epoch": env_int(
            "PNN_DIPP_CONTROL_DISTILL_START_EPOCH", 0
        ),
        "dipp_control_distill_beta": env_float(
            "PNN_DIPP_CONTROL_DISTILL_BETA", 0.10
        ),
        "dipp_control_distill_steer_weight": env_float(
            "PNN_DIPP_CONTROL_DISTILL_STEER_WEIGHT", 0.20
        ),
        "dipp_control_distill_time_weights": env_float_tuple(
            "PNN_DIPP_CONTROL_DISTILL_TIME_WEIGHTS",
            (3.0, 2.0, 1.0),
            length=3,
        ),
        "dipp_control_distill_require_lane_nonregression": env_bool(
            "PNN_DIPP_CONTROL_DISTILL_REQUIRE_LANE_NONREGRESSION", True
        ),
        "dipp_control_distill_lane_tolerance": env_float(
            "PNN_DIPP_CONTROL_DISTILL_LANE_TOLERANCE", 0.02
        ),
        "lambda_safe_parent_control_distill": env_float(
            "PNN_LAMBDA_SAFE_PARENT_CONTROL_DISTILL", 0.0
        ),
        "safe_parent_control_distill_beta": env_float(
            "PNN_SAFE_PARENT_CONTROL_DISTILL_BETA", 0.10
        ),
        "safe_parent_control_distill_steer_weight": env_float(
            "PNN_SAFE_PARENT_CONTROL_DISTILL_STEER_WEIGHT", 1.0
        ),
        "reference_forward_offset": env_float("PNN_REFERENCE_FORWARD_OFFSET", 0.0),
        "stats_quantile_low": env_float("PNN_STATS_QUANTILE_LOW", 0.0),
        "stats_quantile_high": env_float("PNN_STATS_QUANTILE_HIGH", 1.0),
        "clamp_normalized_inputs": env_bool("PNN_CLAMP_NORMALIZED_INPUTS", False),
        "eval_cuda_visible_devices": os.environ.get("PNN_EVAL_GPU", "0"),
        "eval_nnplanner_python": os.environ.get("PYTHON_BIN", sys.executable),
        "eval_pinn_python": os.environ.get("PYTHON_BIN", sys.executable),
        "target_l2_avg": 0.68,
        "target_weight_variation_score": 0.30,
        "best_l2_min_weight_variation_score": 1.0,
        "stop_on_satisfied": False,
    }

    print(
        f"[PNN-B2D] gpus={gpus} world_size={world_size} "
        f"batch_size={cfg_runtime['batch_size']} epochs={cfg_runtime['epochs']} "
        f"save_dir={cfg_runtime['save_dir']} "
        f"coord_convention={cfg_runtime['coord_convention'] or '<infer-from-data>'} "
        f"reference_forward_offset={cfg_runtime['reference_forward_offset']} "
        f"stats_quantile=({cfg_runtime['stats_quantile_low']},{cfg_runtime['stats_quantile_high']}) "
        f"clamp_normalized_inputs={cfg_runtime['clamp_normalized_inputs']} "
        f"resume_optimizer_state={cfg_runtime['resume_optimizer_state']}"
    )
    print(
        "[PNN-B2D] "
        f"default_cost_weights={cfg_runtime['default_cost_weights']} "
        f"weight_prior={cfg_runtime['weight_prior']} "
        f"weight_delta_max={cfg_runtime['weight_delta_max']} "
        f"progress_overshoot_weight={cfg_runtime['progress_overshoot_weight']} "
        f"control_weight_start_epoch={cfg_runtime['control_weight_start_epoch']} "
        f"lambda_gt_reference_lane={cfg_runtime['lambda_gt_reference_lane']} "
        f"lambda_ego_object_safety={cfg_runtime['lambda_ego_object_safety']} "
        f"metric_safety_margin={cfg_runtime['metric_safety_margin']} "
        f"metric_safety_topk={cfg_runtime['metric_safety_topk']} "
        f"metric_safety_time_weights={cfg_runtime['metric_safety_time_weights']} "
        f"risk_safety_gain={cfg_runtime['risk_safety_gain']} "
        f"pnn_only_weighting={cfg_runtime['use_pnn_only_risk_weighting']} "
        f"pnn_only/shared_gain={cfg_runtime['pnn_only_safety_gain']}/{cfg_runtime['shared_safety_gain']} "
        f"lambda_teacher_trust={cfg_runtime['lambda_teacher_trust']} "
        f"pnn_only_hipad_anchor={cfg_runtime['lambda_pnn_only_hipad_anchor']} "
        f"anchor_time_weights={cfg_runtime['pnn_only_hipad_anchor_time_weights']} "
        f"official_frame_acr={cfg_runtime['official_frame_acr_enabled']} "
        f"official_frame_acr_pnn_only="
        f"{cfg_runtime['official_frame_acr_pnn_only']} "
        f"require_perceived_actor="
        f"{cfg_runtime['official_frame_acr_require_perceived_actor']} "
        f"actor_match_radius={cfg_runtime['perceived_actor_match_radius']} "
        f"perception_brake={cfg_runtime['lambda_perception_gated_brake']} "
        f"official_frame_acr/anchor/safe_distill="
        f"{cfg_runtime['lambda_official_frame_acr']}/"
        f"{cfg_runtime['lambda_official_frame_anchor']}/"
        f"{cfg_runtime['lambda_safe_parent_distill']} "
        f"official_frame_lane_guard="
        f"{cfg_runtime['lambda_official_frame_lane_guard']}@"
        f"{cfg_runtime['official_frame_lane_guard_margin']} "
        f"anchor_requires_lane_safe="
        f"{cfg_runtime['official_frame_anchor_require_lane_safe']} "
        f"official_raster_padding={cfg_runtime['official_raster_actor_padding']} "
        f"official_raster_dilation_px="
        f"{cfg_runtime['official_raster_actor_dilation_pixels']} "
        f"teacher_ckpt={cfg_runtime['teacher_ckpt_path'] or '<none>'} "
        f"safety_check_frames=(4,9,14,19,24,29) "
        f"safety_actor_source={'metric_sidecar' if cfg_runtime.get('supervision_data_path') else 'new_data_or_fixed_interpolation'}"
    )
    print(
        "[PNN-B2D] "
        f"lambda_route_speed_excess={cfg_runtime['lambda_route_speed_excess']} "
        f"route_speed_margin={cfg_runtime['route_speed_margin']} "
        f"route_speed_brake_trigger_margin={cfg_runtime['route_speed_brake_trigger_margin']} "
        f"route_speed_pos_accel_threshold={cfg_runtime['route_speed_positive_accel_threshold']} "
        f"route_speed_brake_weight={cfg_runtime['route_speed_brake_weight']} "
        f"route_track_gate=gain:{cfg_runtime['route_track_risk_gate_gain']},"
        f"floor:{cfg_runtime['route_track_risk_min_weight']},"
        f"lane_gain:{cfg_runtime['route_track_lane_risk_gain']} "
        f"route_target_aug=p:{cfg_runtime['route_target_aug_probability']},"
        f"lon_std:{cfg_runtime['route_target_longitudinal_scale_std']},"
        f"lat_std:{cfg_runtime['route_target_lateral_jitter_std']},"
        f"mid_drop:{cfg_runtime['route_target_middle_dropout']}"
    )
    print(
        "[PNN-B2D] "
        f"lambda_dense_route_speed={cfg_runtime['lambda_dense_route_speed']} "
        f"dense_route_speed_margin={cfg_runtime['dense_route_speed_margin']} "
        f"lambda_forward_overshoot={cfg_runtime['lambda_forward_overshoot']} "
        f"forward_overshoot_margin={cfg_runtime['forward_overshoot_margin']} "
        f"forward_overshoot_time_weights={cfg_runtime['forward_overshoot_time_weights']} "
        f"lambda_obstacle_clearance={cfg_runtime['lambda_obstacle_clearance']} "
        f"obs_clearance_margins="
        f"veh:{cfg_runtime['obstacle_clearance_margin_veh']},"
        f"ped:{cfg_runtime['obstacle_clearance_margin_ped']} "
        f"obs_clearance_topk={cfg_runtime['obstacle_clearance_topk']} "
        f"obs_clearance_time_weights={cfg_runtime['obstacle_clearance_time_weights']} "
        f"lambda_rollout_comfort={cfg_runtime['lambda_rollout_comfort']} "
        f"comfort_thresholds="
        f"lon_acc:[{cfg_runtime['comfort_min_lon_accel']},{cfg_runtime['comfort_acc_threshold']}],"
        f"lat_acc:{cfg_runtime['comfort_lat_accel_threshold']},"
        f"jerk:{cfg_runtime['comfort_jerk_threshold']},"
        f"yaw_rate:{cfg_runtime['comfort_yaw_rate_threshold']},"
        f"yaw_accel:{cfg_runtime['comfort_yaw_accel_threshold']} "
        f"lane_clearance={cfg_runtime['lane_clearance_loss_weight']}@{cfg_runtime['lane_clearance_margin']} "
        f"metric_lane={cfg_runtime['lambda_metric_lane']}@{cfg_runtime['metric_lane_margin']} "
        f"safe_side={cfg_runtime['metric_lane_use_gt_safe_side']} "
        f"time_weights={cfg_runtime['metric_lane_time_weights']} "
        f"hard_max_weight={cfg_runtime['metric_lane_hard_max_weight']} "
        f"disable_pred_lane={cfg_runtime['disable_predicted_lane_losses']}"
    )
    print(
        "[PNN-B2D] "
        f"train_soft_constraint_lambdas={cfg_runtime['train_soft_constraint_lambdas']}"
        f" freeze_control_policy={cfg_runtime['freeze_control_policy']}"
        f" control_update_start={cfg_runtime['control_update_start_epoch']}"
        f" train_raw_collisions={cfg_runtime['train_raw_collisions']}"
        f" require_hipad_plan_2hz={cfg_runtime['require_hipad_plan_2hz']}"
        f" weight_hipad_risk={cfg_runtime['lambda_weight_hipad_risk']}"
        f" weight_pnn_collision={cfg_runtime['lambda_weight_pnn_collision']}"
        f" weight_gate={cfg_runtime['weight_application_gate_mode']}"
        f" weight_dipp_safety={cfg_runtime['lambda_weight_dipp_safety']}"
        f" weight_dipp_lane={cfg_runtime['lambda_weight_dipp_lane']}"
        f" weight_dipp_trust={cfg_runtime['lambda_weight_dipp_trust']}"
        f" dipp_start={cfg_runtime['weight_dipp_start_epoch']}"
        f" dipp_interval={cfg_runtime['weight_dipp_update_interval']}"
        f" planner_iterations={cfg_runtime['planner_max_iterations']}"
        f" dipp_control_distill={cfg_runtime['lambda_dipp_control_distill']}"
        f"@epoch{cfg_runtime['dipp_control_distill_start_epoch']}"
        f" safe_parent_control_distill="
        f"{cfg_runtime['lambda_safe_parent_control_distill']}"
    )
    mp.spawn(train_v10.train, args=(world_size, cfg_runtime), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
