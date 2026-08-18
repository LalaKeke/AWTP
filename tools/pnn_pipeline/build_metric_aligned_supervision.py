#!/usr/bin/env python3
"""Build index-aligned metric supervision for PNN Stage-1 training.

The existing converted PNN tensors remain unchanged.  This sidecar stores
training-only GT geometry constructed from the same Bench2Drive fields and
coordinate conventions as the corrected open-loop metrics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import multiprocessing as mp
import os
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
HIPAD_ROOT = SCRIPT_DIR.parents[1]
MAP_ROOT = HIPAD_ROOT.parent
sys.path[:0] = [str(MAP_ROOT), str(HIPAD_ROOT), str(SCRIPT_DIR)]

from hipad_pnn_adapter import hipad_points_to_pnn, hipad_yaw_to_pnn
from convert_hipad_chunk_to_pnn_pt import safe_pickle_load, split_group_order


_DATASET = None
_KEY_TO_INDEX = None
_CFG = None
_METRIC = None
_VECTORIZER = None


def sample_key(info: Mapping[str, Any]) -> Tuple[str, int]:
    return str(info["folder"]), int(info["frame_idx"])


def key_hash(info: Mapping[str, Any]) -> np.int64:
    raw = f"{info['folder']}#{int(info['frame_idx'])}".encode("utf-8")
    return np.frombuffer(hashlib.blake2b(raw, digest_size=8).digest(), dtype=np.int64)[0]


def as_tensor(value, dtype=None) -> torch.Tensor:
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def metric_actor_boxes(data: Mapping[str, Any], max_actors: int):
    """Build boxes from exactly the current-frame tensors used by STP3 occupancy."""
    gt_boxes = as_tensor(data["gt_bboxes_3d"], torch.float32)
    attrs = as_tensor(data["gt_attr_labels"], torch.float32)
    count = min(gt_boxes.shape[0], attrs.shape[0])
    gt_boxes, attrs = gt_boxes[:count], attrs[:count]

    centers = attrs[:, :12].reshape(count, 6, 2).cumsum(dim=1)
    centers = centers + gt_boxes[:, None, :2]
    future_mask = attrs[:, 12:18].bool()
    labels = attrs[:, 27].long()
    # PlanningMetric's two internal yaw conversions cancel the wrapper-side
    # conversion, so its final actor yaw is the original HiPAD yaw plus future
    # deltas. Convert that final yaw once into pnn_xy. The previous expression
    # converted an already intermediate yaw and rotated actors by ~90 degrees.
    future_yaw = gt_boxes[:, 6, None] + attrs[:, 28:34].cumsum(dim=1)
    keep = torch.isin(labels, torch.tensor([0, 1, 2, 3, 7]))
    eligible_all = torch.nonzero(keep, as_tuple=False).flatten()

    # Preserve all official classes, preferring actors closest to ego if padding is needed.
    if eligible_all.numel() > max_actors:
        distance = gt_boxes[eligible_all, :2].norm(dim=-1)
        eligible = eligible_all[torch.argsort(distance)[:max_actors]]
    else:
        eligible = eligible_all

    boxes = torch.zeros((6, max_actors, 5), dtype=torch.float32)
    mask = torch.zeros((6, max_actors), dtype=torch.bool)
    actor_type = torch.zeros((6, max_actors), dtype=torch.int8)
    if eligible.numel():
        centers_pnn = np.asarray(hipad_points_to_pnn(centers[eligible].numpy()), dtype=np.float32)
        yaw_pnn = np.asarray(hipad_yaw_to_pnn(future_yaw[eligible].numpy()), dtype=np.float32)
        n = int(eligible.numel())
        boxes[:, :n, :2] = torch.from_numpy(centers_pnn).transpose(0, 1)
        boxes[:, :n, 2] = torch.from_numpy(yaw_pnn).transpose(0, 1)
        boxes[:, :n, 3] = gt_boxes[eligible, 3].unsqueeze(0)
        boxes[:, :n, 4] = gt_boxes[eligible, 4].unsqueeze(0)
        mask[:, :n] = future_mask[eligible].transpose(0, 1)
        type_now = torch.where(labels[eligible] == 7, 2, 1).to(torch.int8)
        actor_type[:, :n] = type_now.unsqueeze(0).expand(6, -1)
    return boxes, mask, actor_type, max(0, int(eligible_all.numel()) - max_actors)


def sample_line(line, count: int) -> np.ndarray:
    distances = np.linspace(0.0, float(line.length), count, dtype=np.float64)
    return np.asarray([line.interpolate(float(d)).coords[0][:2] for d in distances], dtype=np.float32)


def solid_lane_points(data: Mapping[str, Any], gt_plan_hipad: torch.Tensor, max_lines: int, num_pts: int):
    candidates = []
    reference = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), gt_plan_hipad[:6].numpy()], axis=0
    )
    for label in (1, 2):
        for line in data.get("map_geoms", {}).get(label, []):
            if float(line.length) <= 1e-3:
                continue
            points = sample_line(line, num_pts)
            min_distance = np.linalg.norm(points[:, None] - reference[None], axis=-1).min()
            candidates.append((float(min_distance), points))
    candidates.sort(key=lambda pair: pair[0])
    truncated = max(0, len(candidates) - max_lines)
    candidates = candidates[:max_lines]

    points = np.zeros((max_lines, num_pts, 2), dtype=np.float32)
    mask = np.zeros((max_lines,), dtype=np.bool_)
    if candidates:
        hipad = np.stack([entry[1] for entry in candidates])
        points[: len(candidates)] = hipad_points_to_pnn(hipad)
        mask[: len(candidates)] = True
    return points, mask, truncated


def official_collision_masks(data: Mapping[str, Any], gt_plan_hipad: torch.Tensor):
    gt_boxes = as_tensor(data["gt_bboxes_3d"], torch.float32).clone()
    if gt_boxes.numel():
        length = gt_boxes[:, 3].clone()
        gt_boxes[:, 3] = gt_boxes[:, 4]
        gt_boxes[:, 4] = length
        gt_boxes[:, 6] = -gt_boxes[:, 6] - np.pi / 2
    attrs = as_tensor(data["gt_attr_labels"])
    if attrs.ndim == 2:
        attrs = attrs.unsqueeze(0)
    vehicle_occ, pedestrian_occ = _METRIC.get_label(gt_boxes, attrs)
    occupancy = torch.logical_or(vehicle_occ, pedestrian_occ)
    if occupancy.ndim == 4:
        occupancy = occupancy[0]
    obj_mask = _METRIC.evaluate_single_coll(gt_plan_hipad[:6], occupancy, input_gt=True)

    mapped = _VECTORIZER({"map_geoms": data.get("map_geoms", {})})
    map_pts = torch.as_tensor(mapped.get("gt_map_pts", np.zeros((0, 38, 20, 2))), dtype=torch.float32)
    map_labels = torch.as_tensor(mapped.get("gt_map_labels", np.zeros((0,))), dtype=torch.long)
    lane_mask = _METRIC.evaluate_lane_edge_coll(gt_plan_hipad[:6], map_pts, map_labels)
    return obj_mask.bool(), lane_mask.bool()


def build_sample(info: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    key = sample_key(info)
    if key not in _KEY_TO_INDEX:
        raise KeyError(f"sample missing from configured train dataset: {key}")
    data = _DATASET.get_data_info(_KEY_TO_INDEX[key])
    if sample_key(data) != key:
        raise RuntimeError(f"dataset key mismatch: requested={key}, got={sample_key(data)}")

    boxes, actor_mask, actor_type, actor_truncated = metric_actor_boxes(data, _CFG.max_actors)
    gt_delta = as_tensor(data["gt_ego_fut_trajs_2hz"], torch.float32)
    gt_plan = gt_delta.cumsum(dim=-2)[:6]
    lane, lane_mask, lane_truncated = solid_lane_points(
        data, gt_plan, _CFG.max_solid_lines, _CFG.lane_points
    )
    gt_obj_collision, gt_lane_collision = official_collision_masks(data, gt_plan)
    return {
        "metric_gt_actor_boxes_2hz": boxes.to(torch.float16),
        "metric_gt_actor_mask_2hz": actor_mask,
        "metric_gt_actor_type_2hz": actor_type,
        "metric_gt_actor_truncated": torch.tensor(actor_truncated, dtype=torch.int16),
        "gt_solid_lane_points": torch.from_numpy(lane).to(torch.float16),
        "gt_solid_lane_mask": torch.from_numpy(lane_mask),
        "gt_solid_lane_truncated": torch.tensor(lane_truncated, dtype=torch.int16),
        "gt_obj_collision_mask_2hz": gt_obj_collision,
        "gt_lane_collision_mask_2hz": gt_lane_collision,
        "sample_key_hash": torch.tensor(int(key_hash(info)), dtype=torch.int64),
    }


def part_path(chunk_id: int) -> Path:
    return Path(_CFG.part_dir) / f"metric_supervision_chunk_{chunk_id:03d}.pt"


def atomic_save(data, path: Path):
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(data, tmp)
    os.replace(tmp, path)


def convert_chunk(chunk_id: int):
    torch.set_num_threads(1)
    output = part_path(chunk_id)
    if _CFG.resume and output.exists():
        cached = torch.load(output, map_location="cpu")
        required = {
            "metric_gt_actor_boxes_2hz",
            "metric_gt_actor_mask_2hz",
            "metric_gt_actor_type_2hz",
            "gt_solid_lane_points",
            "gt_solid_lane_mask",
            "gt_obj_collision_mask_2hz",
            "gt_lane_collision_mask_2hz",
            "sample_key_hash",
        }
        if required.issubset(cached):
            return chunk_id, int(cached["sample_key_hash"].shape[0]), "cached"

    info_path = Path(_CFG.info_dir) / f"b2d_infos_train_chunk_{chunk_id:03d}.pkl"
    infos = split_group_order(safe_pickle_load(str(info_path)), _CFG.split_group)
    if _CFG.limit_per_chunk:
        infos = infos[: _CFG.limit_per_chunk]
    rows = [build_sample(info) for info in infos]
    merged = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    atomic_save(merged, output)
    return chunk_id, len(rows), "converted"


def merge_parts(args):
    paths = [Path(args.part_dir) / f"metric_supervision_chunk_{i:03d}.pt" for i in range(args.num_chunks)]
    parts = [torch.load(path, map_location="cpu") for path in paths]
    keys = parts[0].keys()
    merged = {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
    merged["__meta__"] = {
        "version": 2,
        "coord_convention": "pnn_xy",
        "source": "Bench2Drive official occupancy and GT solid lane metrics",
        "actor_semantics": "metric-aligned [6,A,5=(x,y,yaw,length,width)]",
        "actor_yaw_semantics": "pnn_xy yaw converted once from final HiPAD future yaw",
        "lane_semantics": "nearest labels 1=Solid and 2=SolidSolid, sampled in pnn_xy",
        "max_actors": args.max_actors,
        "max_solid_lines": args.max_solid_lines,
        "lane_points": args.lane_points,
    }
    atomic_save(merged, Path(args.output))
    return int(merged["sample_key_hash"].shape[0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HIPAD_ROOT / "projects/configs/hipad_b2d_stage2.py"))
    parser.add_argument("--info-dir", default=str(HIPAD_ROOT / "data/infos/chunks"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--part-dir", default=None)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PNN_CONVERT_WORKERS", "4")))
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--split-group", type=int, default=5)
    parser.add_argument("--max-actors", type=int, default=64)
    parser.add_argument("--max-solid-lines", type=int, default=16)
    parser.add_argument("--lane-points", type=int, default=20)
    parser.add_argument("--limit-per-chunk", type=int, default=0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--keep-parts", action="store_true")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, args.num_chunks))
    args.part_dir = args.part_dir or str(Path(args.output).with_suffix(".parts"))
    return args


def main():
    global _DATASET, _KEY_TO_INDEX, _CFG, _METRIC, _VECTORIZER
    args = parse_args()
    os.chdir(HIPAD_ROOT)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.part_dir).mkdir(parents=True, exist_ok=True)

    from mmcv import Config
    from mmdet.datasets import build_dataset
    import projects.mmdet3d_plugin  # noqa: F401
    from projects.mmdet3d_plugin.datasets.evaluation.planning.metric_stp3 import PlanningMetric
    from projects.mmdet3d_plugin.datasets.pipelines.vectorize import VectorizePloyLine

    cfg = Config.fromfile(args.config)
    _DATASET = build_dataset(cfg.data.train)
    _KEY_TO_INDEX = {sample_key(info): i for i, info in enumerate(_DATASET.data_infos)}
    if len(_KEY_TO_INDEX) != len(_DATASET.data_infos):
        raise RuntimeError("folder/frame_idx keys are not unique in train dataset")
    _METRIC = PlanningMetric()
    _VECTORIZER = VectorizePloyLine(
        roi_size=tuple(cfg.get("map_roi_size", (30, 60))),
        simplify=False,
        normalize=False,
        sample_num=int(cfg.get("map_num_pts", 20)),
        permute=True,
    )
    _CFG = SimpleNamespace(**vars(args))

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for done, (chunk_id, count, status) in enumerate(
            pool.imap_unordered(convert_chunk, range(args.num_chunks)), start=1
        ):
            print(f"[metric-supervision] {done:02d}/{args.num_chunks} chunk={chunk_id:03d} n={count} {status}", flush=True)

    count = merge_parts(args)
    print(f"[metric-supervision] saved {args.output}; samples={count}", flush=True)
    if not args.keep_parts:
        shutil.rmtree(args.part_dir)
    del _DATASET
    gc.collect()


if __name__ == "__main__":
    main()
