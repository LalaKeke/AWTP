#!/usr/bin/env python3
"""Convert one saved HiP-AD inference chunk into PNN_EV_v12 .pt files.

This is intentionally standalone: it reads HiP-AD `tools/test.py --out` pkl
records and writes the pair consumed by PNN `PairedOldNewDataset`:

  old.pt: ego_state, ped_states, veh_states, lane_points, ped_mask, veh_mask
  new.pt: ego_future_gt, ego_future_gt_valid_mask

By default, `ego_future_gt` is reconstructed from Bench2Drive train infos:
future 2Hz poses are transformed into the current LiDAR frame, accumulated to
absolute relative coordinates, and sampled at 1s/2s/3s.
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.storage

HIPAD_ROOT = str(Path(os.environ.get("HIPAD_PNN_ROOT", Path(__file__).resolve().parents[2])).resolve())
sys.path.insert(0, HIPAD_ROOT)
from pnn_temporal_alignment import (
    ALIGNMENT_VERSION,
    HIPAD_MOTION_DT,
    PNN_ACTOR_TIMES,
    align_hipad_motion_future,
)
from hipad_pnn_adapter import (
    NUM_PEDS,
    NUM_VEHS,
    NUM_LANES,
    LANE_POINTS,
    agent_states_hipad_to_pnn,
    hipad_points_to_pnn,
    hipad_yaw_to_pnn,
    normalize_lane_points,
    navigation_points_to_route_targets,
    pack_agents,
    plan_to_route_targets,
    select_left_right_lane_boundaries,
)


VEH_LABELS = {0, 1, 2, 3}
PED_LABELS = {7}
PNN_GT_INDICES = (1, 3, 5)
GT_ACTOR_FRAMES = 6
MAX_GT_ACTORS = 64


def official_actor_kind(name: Any) -> Optional[str]:
    """Map raw Bench2Drive names to the two occupancy classes used by STP3."""
    name = str(name)
    if name.startswith("walker.pedestrian."):
        return "pedestrian"
    if name.startswith("vehicle.") or name.startswith("/Game/Carla/Static/Car/"):
        return "vehicle"
    return None


def build_coord_meta(args: argparse.Namespace) -> Dict[str, Any]:
    """Small non-tensor contract saved into .pt files.

    Training code reads this to fail fast when a script accidentally mixes
    hipad_xy and pnn_xy tensors. Keep it outside the per-chunk tensor dicts so
    cat_tensor_dicts can remain purely tensor-based.
    """
    return {
        "coord_convention": str(args.coord_convention),
        "route_source": str(args.route_source),
        "gt_source": str(args.gt_source),
        "plan_key": str(args.plan_key),
        "lane_points_semantics": "lane_points[0]=left_boundary,lane_points[1]=right_boundary",
        "gt_reference_line_semantics": (
            "GT-derived path reference [ego_at_t0, GT_1s, GT_2s, GT_3s]; "
            "used only for lane_xy/lane_theta supervision, never as network input"
        ),
        "gt_actor_boxes_2hz_semantics": (
            "sample-aligned future GT actors [T=6,A=64,5=(x,y,yaw,length,width)] "
            "at 0.5s intervals in coord_convention; current nonzero-point vehicle/human actors"
        ),
        "official_collision_semantics": (
            "HiP-AD/STP3 occupancy uses ego length=4.084,width=1.85 and a +0.5m "
            "forward footprint-center offset; fut_valid checks complete ego future only"
        ),
        "state_layout": "ego_state=[x,y,theta,v,xr1,yr1,xr2,yr2,xr3,yr3]",
        "hipad_xy": "x=right/lateral,y=forward,ego_theta=pi/2",
        "pnn_xy": "x=forward,y=left,ego_theta=0",
        "target_forward_offset": float(args.target_forward_offset),
        "reference_forward_offset": float(args.reference_forward_offset),
        "reference_forward_offset_semantics": (
            "PNN ego state reference point is this many meters ahead of the "
            "HiP-AD/GT ego reference point; ego_state xy, route targets, and "
            "ego_future_gt are shifted forward by this amount for training."
        ),
        "actor_alignment_version": ALIGNMENT_VERSION,
        "actor_motion_source_dt": HIPAD_MOTION_DT,
        "actor_target_times": tuple(float(value) for value in PNN_ACTOR_TIMES),
    }


def attach_coord_meta(
    old_data: Dict[str, torch.Tensor],
    new_data: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    meta = build_coord_meta(args)
    old_data["__meta__"] = dict(meta, tensor_role="old_scene_inputs")
    new_data["__meta__"] = dict(meta, tensor_role="new_future_gt")


def apply_forward_offset_to_points(
    points: np.ndarray,
    offset: float,
    coord_convention: str,
) -> np.ndarray:
    offset = float(offset)
    out = np.asarray(points, dtype=np.float32).copy()
    if abs(offset) < 1e-8:
        return out
    if coord_convention == "pnn_xy":
        out[..., 0] += offset
    elif coord_convention == "hipad_xy":
        out[..., 1] += offset
    else:
        raise ValueError(f"Unsupported coord_convention={coord_convention!r}")
    return out.astype(np.float32)


def _load_from_bytes_cpu(buf: bytes):
    return torch.load(io.BytesIO(buf), map_location="cpu")


def safe_pickle_load(path: str):
    original = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = _load_from_bytes_cpu
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = original


def to_numpy(x: Any, dtype=np.float32) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def extract_agents(img_bbox: Mapping[str, Any], score_thr: float = 0.3) -> Tuple[List[dict], List[dict]]:
    boxes = to_numpy(img_bbox["boxes_3d"])
    scores = to_numpy(img_bbox["scores_3d"])
    labels = to_numpy(img_bbox["labels_3d"], dtype=np.int64)
    trajs = to_numpy(img_bbox["trajs_3d"])
    traj_scores = to_numpy(img_bbox["trajs_score"])

    keep = scores > float(score_thr)
    boxes = boxes[keep]
    labels = labels[keep]
    trajs = trajs[keep]
    traj_scores = traj_scores[keep]
    if len(trajs) > 0:
        trajs = trajs[np.arange(len(trajs)), traj_scores.argmax(axis=-1)]

    ped_agents: List[dict] = []
    veh_agents: List[dict] = []
    for box, label, future in zip(boxes, labels, trajs):
        future = np.asarray(future, dtype=np.float32)
        if future.ndim != 2 or future.shape[-1] != 2 or future.shape[0] == 0:
            continue
        yaw = float(box[6]) if box.shape[0] > 6 else np.pi / 2
        velocity_xy = box[7:9] if box.shape[0] >= 9 else None
        aligned_future = align_hipad_motion_future(
            current_xy=(float(box[0]), float(box[1])),
            future=future,
            source_dt=HIPAD_MOTION_DT,
            measured_velocity_xy=velocity_xy,
            max_speed=5.0 if int(label) in PED_LABELS else 20.0,
        )
        if velocity_xy is not None:
            speed = float(np.linalg.norm(velocity_xy))
        elif aligned_future.shape[0] >= 2:
            speed = float(np.linalg.norm(aligned_future[1] - aligned_future[0]) / 0.5)
        else:
            speed = 0.0
        agent = {
            "x": float(box[0]),
            "y": float(box[1]),
            "yaw": yaw,
            "speed": speed,
            "future": aligned_future,
            "goal": aligned_future[-1],
        }
        label = int(label)
        if label in PED_LABELS:
            ped_agents.append(agent)
        elif label in VEH_LABELS:
            veh_agents.append(agent)
    return ped_agents, veh_agents


def extract_lane_points(img_bbox: Mapping[str, Any], reference_plan: Optional[np.ndarray] = None) -> np.ndarray:
    vectors = img_bbox.get("vectors", [])
    scores = img_bbox.get("scores", None)
    labels = img_bbox.get("labels", None)
    if len(vectors) < 2:
        return normalize_lane_points(None)
    return select_left_right_lane_boundaries(
        vectors=vectors,
        scores=scores,
        labels=labels,
        reference_plan=reference_plan,
        num_lanes=NUM_LANES,
        num_points=LANE_POINTS,
    )


def split_group_order(infos: Sequence[Mapping[str, Any]], split_group: int = 5) -> List[Mapping[str, Any]]:
    if split_group <= 0:
        return list(infos)
    ordered: List[Mapping[str, Any]] = []
    for group_idx in range(split_group):
        ordered.extend(infos[group_idx::split_group])
    return ordered


def build_info_index(infos: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    index: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for info in infos:
        if "folder" not in info or "frame_idx" not in info:
            continue
        index[(str(info["folder"]), int(info["frame_idx"]))] = info
    return index


def reconstruct_ego_future_gt(
    info: Mapping[str, Any],
    info_index: Mapping[Tuple[str, int], Mapping[str, Any]],
    future_frames: int = 6,
    frame_interval: int = 5,
) -> Tuple[np.ndarray, bool]:
    folder = str(info["folder"])
    frame_idx = int(info["frame_idx"])
    world2lidar_cur = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float64)

    future_points = np.zeros((future_frames, 2), dtype=np.float32)
    valid_mask = np.zeros((future_frames,), dtype=np.bool_)

    for i in range(future_frames):
        target_frame_idx = frame_idx + (i + 1) * frame_interval
        adj_info = info_index.get((folder, target_frame_idx))
        if adj_info is None:
            continue
        world2lidar_adj = np.asarray(adj_info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float64)
        adj2cur_lidar = world2lidar_cur @ np.linalg.inv(world2lidar_adj)
        future_points[i] = adj2cur_lidar[0:2, 3].astype(np.float32)
        valid_mask[i] = True

    selected = future_points[list(PNN_GT_INDICES)]
    selected_valid = bool(valid_mask[list(PNN_GT_INDICES)].all())
    if not selected_valid:
        selected = np.zeros((len(PNN_GT_INDICES), 2), dtype=np.float32)
    return selected, selected_valid


def reconstruct_gt_actor_boxes(
    info: Mapping[str, Any],
    info_index: Mapping[Tuple[str, int], Mapping[str, Any]],
    coord_convention: str,
    future_frames: int = GT_ACTOR_FRAMES,
    frame_interval: int = 5,
    max_actors: int = MAX_GT_ACTORS,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Rebuild official 2 Hz actor boxes in the current LiDAR frame.

    Returns boxes [T,A,5] as (x,y,yaw,length,width), validity [T,A], and
    the number of eligible current actors omitted by the fixed padded shape.
    The actor selection matches get_ann_info: current boxes with nonzero lidar
    points, restricted to vehicle/human classes used by PlanningMetric.
    """
    folder = str(info["folder"])
    frame_idx = int(info["frame_idx"])
    world2lidar_cur = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float64)
    gt_ids = np.asarray(info["gt_ids"])
    gt_names = np.asarray(info["gt_names"])
    gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float32)
    num_points = np.asarray(info.get("num_points", np.ones(len(gt_ids))), dtype=np.float32)

    eligible = [
        i for i, name in enumerate(gt_names)
        if num_points[i] != 0 and official_actor_kind(name) is not None
    ]
    eligible.sort(key=lambda i: float(np.linalg.norm(gt_boxes[i, :2])))
    truncated = max(0, len(eligible) - int(max_actors))
    eligible = eligible[:max_actors]

    boxes = np.zeros((future_frames, max_actors, 5), dtype=np.float32)
    mask = np.zeros((future_frames, max_actors), dtype=np.bool_)
    for actor_slot, cur_idx in enumerate(eligible):
        actor_id = gt_ids[cur_idx]
        # Raw B2D boxes use (width,length,height) at dimensions 3:6.
        width = float(gt_boxes[cur_idx, 3])
        length = float(gt_boxes[cur_idx, 4])
        for t in range(future_frames):
            adj_info = info_index.get((folder, frame_idx + (t + 1) * frame_interval))
            if adj_info is None:
                continue
            matches = np.flatnonzero(np.asarray(adj_info["gt_ids"]) == actor_id)
            if matches.size != 1:
                continue
            adj_idx = int(matches[0])
            adj2cur_lidar = world2lidar_cur @ np.asarray(adj_info["npc2world"][adj_idx], dtype=np.float64)
            xy = adj2cur_lidar[0:2, 3].astype(np.float32)
            yaw = np.float32(np.arctan2(adj2cur_lidar[1, 0], adj2cur_lidar[0, 0]))
            if coord_convention == "pnn_xy":
                xy = hipad_points_to_pnn(xy)
                yaw = np.asarray(hipad_yaw_to_pnn(yaw), dtype=np.float32).item()
            boxes[t, actor_slot] = (xy[0], xy[1], yaw, length, width)
            mask[t, actor_slot] = True
    return boxes, mask, truncated


def navigation_points_from_info(info: Mapping[str, Any]) -> np.ndarray:
    """Convert Bench2Drive near/far navigation annotations to LiDAR coordinates."""
    theta_to_lidar = -(float(info["ego_yaw"]) - np.pi / 2)
    rotation_matrix = np.array(
        [
            [np.cos(theta_to_lidar), -np.sin(theta_to_lidar)],
            [np.sin(theta_to_lidar), np.cos(theta_to_lidar)],
        ],
        dtype=np.float32,
    )
    ego_xy = np.asarray(info["ego_translation"], dtype=np.float32)[:2]
    near_xy = np.asarray(info["command_near_xy"], dtype=np.float32) - ego_xy
    far_xy = np.asarray(info["command_far_xy"], dtype=np.float32) - ego_xy
    return np.stack(
        [rotation_matrix @ near_xy, rotation_matrix @ far_xy],
        axis=0,
    ).astype(np.float32)


def make_sample(
    record: Mapping[str, Any],
    info: Optional[Mapping[str, Any]],
    plan_key: str,
    score_thr: float,
    gt_source: str,
    route_source: str,
    navigation_min_speed: float,
    navigation_max_speed: float,
    navigation_distance_scale: Any,
    navigation_interpolation: str,
    coord_convention: str,
    target_forward_offset: float,
    reference_forward_offset: float,
    info_index: Optional[Mapping[Tuple[str, int], Mapping[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    img_bbox = record["img_bbox"]
    plan = None
    if plan_key in img_bbox:
        plan = to_numpy(img_bbox[plan_key])
        if plan.ndim != 2 or plan.shape != (6, 2):
            plan = None
    if route_source == "hipad_plan" and plan is None:
        return None

    ped_agents, veh_agents = extract_agents(img_bbox, score_thr=score_thr)
    ped_states, ped_mask = pack_agents(ped_agents, NUM_PEDS)
    veh_states, veh_mask = pack_agents(veh_agents, NUM_VEHS)
    lane_points = extract_lane_points(img_bbox, reference_plan=plan)

    ego_speed = 0.0
    if info is not None and "ego_vel" in info:
        ego_speed = float(np.linalg.norm(np.asarray(info["ego_vel"], dtype=np.float32)[:2]))

    if route_source == "navigation":
        if info is None:
            raise ValueError("route_source=navigation requires Bench2Drive info annotations")
        navigation_points = navigation_points_from_info(info)
        route = navigation_points_to_route_targets(
            navigation_points,
            ego_speed=ego_speed,
            min_speed=navigation_min_speed,
            max_speed=navigation_max_speed,
            distance_scale=navigation_distance_scale,
            interpolation=navigation_interpolation,
        )
    elif route_source == "hipad_plan":
        route = plan_to_route_targets(plan)
    else:
        raise ValueError(f"Unsupported route_source={route_source!r}")

    if coord_convention == "pnn_xy":
        route = hipad_points_to_pnn(route.reshape(3, 2)).reshape(-1)
        ped_states = agent_states_hipad_to_pnn(ped_states)
        veh_states = agent_states_hipad_to_pnn(veh_states)
        lane_points = hipad_points_to_pnn(lane_points)
        ego_xy = (0.0, 0.0)
        ego_yaw = 0.0
    elif coord_convention == "hipad_xy":
        ego_xy = (0.0, 0.0)
        ego_yaw = np.pi / 2
    else:
        raise ValueError(f"Unsupported coord_convention={coord_convention!r}")

    reference_forward_offset = float(reference_forward_offset)
    if abs(reference_forward_offset) > 1e-8:
        ego_xy_arr = np.asarray(ego_xy, dtype=np.float32)
        ego_xy = tuple(apply_forward_offset_to_points(
            ego_xy_arr,
            reference_forward_offset,
            coord_convention,
        ).reshape(2).tolist())
        route = apply_forward_offset_to_points(
            route.reshape(3, 2),
            reference_forward_offset,
            coord_convention,
        ).reshape(-1)

    target_forward_offset = float(target_forward_offset)
    if abs(target_forward_offset) > 1e-8:
        route = apply_forward_offset_to_points(
            route.reshape(3, 2),
            target_forward_offset,
            coord_convention,
        ).reshape(-1)

    ego_state = np.array([ego_xy[0], ego_xy[1], ego_yaw, ego_speed, *route], dtype=np.float32)
    if gt_source == "hipad_plan":
        if plan is None:
            return None
        ego_future_gt = route.reshape(3, 2).astype(np.float32)
        valid = bool(record.get("metric_results", {}).get("fut_valid_flag", True))
    else:
        if info is None or info_index is None:
            raise ValueError("gt_source=true_gt requires info and full info index")
        ego_future_gt, valid = reconstruct_ego_future_gt(info, info_index)
        if coord_convention == "pnn_xy":
            ego_future_gt = hipad_points_to_pnn(ego_future_gt)
        if abs(reference_forward_offset) > 1e-8:
            ego_future_gt = apply_forward_offset_to_points(
                ego_future_gt,
                reference_forward_offset,
                coord_convention,
            )
        if abs(target_forward_offset) > 1e-8:
            ego_future_gt = apply_forward_offset_to_points(
                ego_future_gt,
                target_forward_offset,
                coord_convention,
            )

    if info is None or info_index is None:
        gt_actor_boxes = np.zeros((GT_ACTOR_FRAMES, MAX_GT_ACTORS, 5), dtype=np.float32)
        gt_actor_mask = np.zeros((GT_ACTOR_FRAMES, MAX_GT_ACTORS), dtype=np.bool_)
        gt_actor_truncated = 0
    else:
        gt_actor_boxes, gt_actor_mask, gt_actor_truncated = reconstruct_gt_actor_boxes(
            info,
            info_index,
            coord_convention=coord_convention,
        )

    metric_results = record.get("metric_results", {})
    official_hipad_obj_box_col = np.asarray(
        [metric_results.get(f"plan_obj_box_col_{i}s", 0.0) for i in (1, 2, 3)],
        dtype=np.float32,
    )
    return {
        "ego_state": ego_state,
        "ped_states": ped_states,
        "veh_states": veh_states,
        "lane_points": lane_points,
        "ped_mask": ped_mask,
        "veh_mask": veh_mask,
        "ego_future_gt": ego_future_gt,
        "ego_future_gt_valid_mask": np.asarray(valid, dtype=np.bool_),
        # The GT ego path is the reliable lane-following supervision available
        # in the B2D infos. It is kept separate from predicted map boundaries:
        # predicted lanes remain network input and hard-boundary constraints.
        "gt_reference_line": np.concatenate(
            [np.asarray(ego_xy, dtype=np.float32)[None], ego_future_gt], axis=0
        ).astype(np.float32),
        "gt_reference_line_valid_mask": np.asarray(valid, dtype=np.bool_),
        "gt_actor_boxes_2hz": gt_actor_boxes,
        "gt_actor_mask_2hz": gt_actor_mask,
        "gt_actor_truncated": np.asarray(gt_actor_truncated, dtype=np.int16),
        "official_hipad_obj_box_col": official_hipad_obj_box_col,
        "official_fut_valid_mask": np.asarray(
            metric_results.get("fut_valid_flag", valid), dtype=np.bool_
        ),
    }


def convert_records(
    records: Sequence[Mapping[str, Any]],
    infos: Sequence[Optional[Mapping[str, Any]]],
    plan_key: str,
    score_thr: float,
    gt_source: str,
    route_source: str,
    navigation_min_speed: float,
    navigation_max_speed: float,
    navigation_distance_scale: Any,
    navigation_interpolation: str,
    coord_convention: str,
    target_forward_offset: float,
    reference_forward_offset: float,
    info_index: Optional[Mapping[Tuple[str, int], Mapping[str, Any]]] = None,
    limit: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    limit = len(records) if limit <= 0 else min(limit, len(records))

    samples = []
    for idx in range(limit):
        sample = make_sample(
            records[idx],
            infos[idx] if idx < len(infos) else None,
            plan_key,
            score_thr,
            gt_source,
            route_source,
            navigation_min_speed,
            navigation_max_speed,
            navigation_distance_scale,
            navigation_interpolation,
            coord_convention,
            target_forward_offset,
            reference_forward_offset,
            info_index,
        )
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise RuntimeError("No valid samples converted.")

    old_data = {
        "ego_state": torch.as_tensor(np.stack([s["ego_state"] for s in samples]), dtype=torch.float32),
        "ped_states": torch.as_tensor(np.stack([s["ped_states"] for s in samples]), dtype=torch.float32),
        "veh_states": torch.as_tensor(np.stack([s["veh_states"] for s in samples]), dtype=torch.float32),
        "lane_points": torch.as_tensor(np.stack([s["lane_points"] for s in samples]), dtype=torch.float32),
        "ped_mask": torch.as_tensor(np.stack([s["ped_mask"] for s in samples]), dtype=torch.bool),
        "veh_mask": torch.as_tensor(np.stack([s["veh_mask"] for s in samples]), dtype=torch.bool),
    }
    new_data = {
        "ego_future_gt": torch.as_tensor(np.stack([s["ego_future_gt"] for s in samples]), dtype=torch.float32),
        "ego_future_gt_valid_mask": torch.as_tensor(np.stack([s["ego_future_gt_valid_mask"] for s in samples]), dtype=torch.bool),
        "gt_reference_line": torch.as_tensor(np.stack([s["gt_reference_line"] for s in samples]), dtype=torch.float32),
        "gt_reference_line_valid_mask": torch.as_tensor(np.stack([s["gt_reference_line_valid_mask"] for s in samples]), dtype=torch.bool),
        "gt_actor_boxes_2hz": torch.as_tensor(np.stack([s["gt_actor_boxes_2hz"] for s in samples]), dtype=torch.float32),
        "gt_actor_mask_2hz": torch.as_tensor(np.stack([s["gt_actor_mask_2hz"] for s in samples]), dtype=torch.bool),
        "gt_actor_truncated": torch.as_tensor(np.stack([s["gt_actor_truncated"] for s in samples]), dtype=torch.int16),
        "official_hipad_obj_box_col": torch.as_tensor(np.stack([s["official_hipad_obj_box_col"] for s in samples]), dtype=torch.float32),
        "official_fut_valid_mask": torch.as_tensor(np.stack([s["official_fut_valid_mask"] for s in samples]), dtype=torch.bool),
    }
    return old_data, new_data


def convert(args: argparse.Namespace) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    records = safe_pickle_load(args.chunk_pkl)
    raw_infos = safe_pickle_load(args.info_pkl) if args.info_pkl else [None] * len(records)
    infos = split_group_order(raw_infos, args.split_group) if args.info_pkl else raw_infos
    info_index = None
    if args.gt_source == "true_gt":
        all_infos = safe_pickle_load(args.all_info_pkl)
        info_index = build_info_index(all_infos)
    old_data, new_data = convert_records(
        records=records,
        infos=infos,
        plan_key=args.plan_key,
        score_thr=args.score_thr,
        gt_source=args.gt_source,
        route_source=args.route_source,
        navigation_min_speed=args.navigation_min_speed,
        navigation_max_speed=args.navigation_max_speed,
        navigation_distance_scale=args.navigation_distance_scale,
        navigation_interpolation=args.navigation_interpolation,
        coord_convention=args.coord_convention,
        target_forward_offset=args.target_forward_offset,
        reference_forward_offset=args.reference_forward_offset,
        info_index=info_index,
        limit=args.limit,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    old_path = os.path.join(args.output_dir, args.old_name)
    new_path = os.path.join(args.output_dir, args.new_name)
    attach_coord_meta(old_data, new_data, args)
    torch.save(old_data, old_path)
    torch.save(new_data, new_path)
    print(f"saved old_data: {old_path}")
    print(f"saved new_data: {new_path}")
    print(f"num_samples={old_data['ego_state'].shape[0]}")
    for key, value in old_data.items():
        if not torch.is_tensor(value):
            print(f"old.{key}: {value}")
            continue
        print(f"old.{key}: {tuple(value.shape)} {value.dtype}")
    for key, value in new_data.items():
        if not torch.is_tensor(value):
            print(f"new.{key}: {value}")
            continue
        print(f"new.{key}: {tuple(value.shape)} {value.dtype}")
    return old_data, new_data


def cat_tensor_dicts(parts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = parts[0].keys()
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}


def convert_all_chunks(args: argparse.Namespace) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    all_infos = safe_pickle_load(args.all_info_pkl)
    info_index = build_info_index(all_infos) if args.gt_source == "true_gt" else None
    old_parts: List[Dict[str, torch.Tensor]] = []
    new_parts: List[Dict[str, torch.Tensor]] = []

    for chunk_id in range(args.num_chunks):
        chunk_pkl = os.path.join(args.chunk_dir, f"hipad_stage2_b2d_train_chunk_{chunk_id:03d}.pkl")
        info_pkl = os.path.join(args.info_dir, f"b2d_infos_train_chunk_{chunk_id:03d}.pkl")
        if not os.path.exists(chunk_pkl):
            raise FileNotFoundError(chunk_pkl)
        if not os.path.exists(info_pkl):
            raise FileNotFoundError(info_pkl)

        records = safe_pickle_load(chunk_pkl)
        infos = split_group_order(safe_pickle_load(info_pkl), args.split_group)
        old_data, new_data = convert_records(
            records=records,
            infos=infos,
            plan_key=args.plan_key,
            score_thr=args.score_thr,
            gt_source=args.gt_source,
            route_source=args.route_source,
            navigation_min_speed=args.navigation_min_speed,
            navigation_max_speed=args.navigation_max_speed,
            navigation_distance_scale=args.navigation_distance_scale,
            navigation_interpolation=args.navigation_interpolation,
            coord_convention=args.coord_convention,
            target_forward_offset=args.target_forward_offset,
            reference_forward_offset=args.reference_forward_offset,
            info_index=info_index,
            limit=0,
        )
        old_parts.append(old_data)
        new_parts.append(new_data)
        valid = int(new_data["ego_future_gt_valid_mask"].sum().item())
        total = int(new_data["ego_future_gt_valid_mask"].numel())
        print(f"chunk {chunk_id:03d}: converted {total} samples, valid_gt={valid}", flush=True)

    old_all = cat_tensor_dicts(old_parts)
    new_all = cat_tensor_dicts(new_parts)
    attach_coord_meta(old_all, new_all, args)

    os.makedirs(args.output_dir, exist_ok=True)
    old_path = os.path.join(args.output_dir, args.old_name)
    new_path = os.path.join(args.output_dir, args.new_name)
    torch.save(old_all, old_path)
    torch.save(new_all, new_path)
    print(f"saved old_data: {old_path}", flush=True)
    print(f"saved new_data: {new_path}", flush=True)
    print(f"num_samples={old_all['ego_state'].shape[0]}", flush=True)
    print(f"valid_gt={int(new_all['ego_future_gt_valid_mask'].sum().item())}", flush=True)
    for key, value in old_all.items():
        if not torch.is_tensor(value):
            print(f"old.{key}: {value}", flush=True)
            continue
        print(f"old.{key}: {tuple(value.shape)} {value.dtype}", flush=True)
    for key, value in new_all.items():
        if not torch.is_tensor(value):
            print(f"new.{key}: {value}", flush=True)
            continue
        print(f"new.{key}: {tuple(value.shape)} {value.dtype}", flush=True)
    return old_all, new_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-pkl", default=os.path.join(HIPAD_ROOT, "outputs/inference_chunks/hipad_stage2_b2d_train_chunk_000.pkl"))
    parser.add_argument("--info-pkl", default=os.path.join(HIPAD_ROOT, "data/infos/chunks/b2d_infos_train_chunk_000.pkl"))
    parser.add_argument("--all-info-pkl", default=os.path.join(HIPAD_ROOT, "data/infos/b2d_infos_train.pkl"))
    parser.add_argument("--chunk-dir", default=os.path.join(HIPAD_ROOT, "outputs/inference_chunks"))
    parser.add_argument("--info-dir", default=os.path.join(HIPAD_ROOT, "data/infos/chunks"))
    parser.add_argument("--output-dir", default=os.path.join(HIPAD_ROOT, "data/pnn/navigation"))
    parser.add_argument("--old-name", default="hipad_stage2_b2d_train_pnn_navigation_old.pt")
    parser.add_argument("--new-name", default="hipad_stage2_b2d_train_pnn_navigation_true_gt.pt")
    parser.add_argument("--plan-key", default="plan_temp_2hz")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--split-group", type=int, default=5)
    parser.add_argument("--gt-source", choices=["true_gt", "hipad_plan"], default="true_gt")
    parser.add_argument("--route-source", choices=["navigation", "hipad_plan"], default="navigation")
    parser.add_argument(
        "--coord-convention",
        choices=["hipad_xy", "pnn_xy"],
        default=os.environ.get("PNN_COORD_CONVENTION", "hipad_xy"),
        help="coordinate convention used inside PNN training tensors",
    )
    parser.add_argument(
        "--target-forward-offset",
        type=float,
        default=float(os.environ.get("PNN_TARGET_FORWARD_OFFSET", "0.0")),
        help=(
            "meters; shift route targets and true GT targets along the ego "
            "forward axis in the selected coordinate convention. Negative "
            "values move targets backward."
        ),
    )
    parser.add_argument(
        "--reference-forward-offset",
        type=float,
        default=float(os.environ.get("PNN_REFERENCE_FORWARD_OFFSET", "0.0")),
        help=(
            "meters; PNN ego state reference point forward offset relative to "
            "HiP-AD/GT ego reference. Positive means PNN state point is ahead. "
            "Shifts ego_state xy, route targets, and ego_future_gt forward in "
            "training tensors. At inference, use the same value through "
            "PNN_REFERENCE_FORWARD_OFFSET to shift PNN outputs back."
        ),
    )
    parser.add_argument("--navigation-min-speed", type=float, default=1.0)
    parser.add_argument("--navigation-max-speed", type=float, default=15.0)
    parser.add_argument(
        "--navigation-distance-scale",
        default="1.0",
        help="scalar or comma-separated per-horizon scales, e.g. 0.96,1.03,1.12",
    )
    parser.add_argument(
        "--navigation-interpolation",
        choices=["spline", "polyline"],
        default="spline",
        help="route geometry interpolation before arc-length sampling",
    )
    parser.add_argument("--all-chunks", action="store_true")
    parser.add_argument("--num-chunks", type=int, default=32)
    args = parser.parse_args()
    if args.all_chunks:
        convert_all_chunks(args)
    else:
        convert(args)


if __name__ == "__main__":
    main()
