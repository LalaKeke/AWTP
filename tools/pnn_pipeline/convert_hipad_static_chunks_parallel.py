#!/usr/bin/env python3
"""Parallel, resumable converter for HiP-AD chunks to paired PNN tensors."""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIPAD_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MAP_ROOT = os.path.dirname(HIPAD_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import convert_hipad_chunk_to_pnn_static_pt as base


_INFO_INDEX = None
_CFG = None


def _part_paths(part_dir: str, chunk_id: int) -> Tuple[str, str]:
    stem = f"chunk_{chunk_id:03d}"
    return os.path.join(part_dir, f"{stem}_old.pt"), os.path.join(part_dir, f"{stem}_new.pt")


def _atomic_torch_save(data: Dict[str, torch.Tensor], path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(data, tmp)
    os.replace(tmp, path)


def _convert_chunk(chunk_id: int):
    torch.set_num_threads(1)
    old_path, new_path = _part_paths(_CFG.part_dir, chunk_id)
    if _CFG.resume and os.path.exists(old_path) and os.path.exists(new_path):
        old = torch.load(old_path, map_location="cpu")
        new = torch.load(new_path, map_location="cpu")
        required_new = {
            "gt_actor_boxes_2hz",
            "gt_actor_mask_2hz",
            "official_hipad_obj_box_col",
        }
        required_old = {"static_states", "static_mask"}
        if required_new.issubset(new) and required_old.issubset(old):
            return chunk_id, int(old["ego_state"].shape[0]), int(new["ego_future_gt_valid_mask"].sum()), "cached"

    chunk_pkl = os.path.join(_CFG.chunk_dir, f"hipad_stage2_b2d_train_chunk_{chunk_id:03d}.pkl")
    info_pkl = os.path.join(_CFG.info_dir, f"b2d_infos_train_chunk_{chunk_id:03d}.pkl")
    records = base.safe_pickle_load(chunk_pkl)
    infos = base.split_group_order(base.safe_pickle_load(info_pkl), _CFG.split_group)
    old, new = base.convert_records(
        records=records,
        infos=infos,
        plan_key=_CFG.plan_key,
        score_thr=_CFG.score_thr,
        gt_source=_CFG.gt_source,
        route_source=_CFG.route_source,
        navigation_min_speed=1.0,
        navigation_max_speed=15.0,
        navigation_distance_scale=(1.0, 1.0, 1.0),
        navigation_interpolation="spline",
        coord_convention=_CFG.coord_convention,
        target_forward_offset=_CFG.target_forward_offset,
        reference_forward_offset=_CFG.reference_forward_offset,
        actor_motion_source_dt=_CFG.actor_motion_source_dt,
        info_index=_INFO_INDEX,
        limit=0,
    )
    _atomic_torch_save(old, old_path)
    _atomic_torch_save(new, new_path)
    return chunk_id, int(old["ego_state"].shape[0]), int(new["ego_future_gt_valid_mask"].sum()), "converted"


def _merge_parts(args, role: str, output_path: str) -> Dict[str, torch.Tensor]:
    paths = [_part_paths(args.part_dir, i)[0 if role == "old" else 1] for i in range(args.num_chunks)]
    lengths = []
    template = None
    for path in paths:
        part = torch.load(path, map_location="cpu")
        if template is None:
            template = part
        first = next(value for value in part.values() if torch.is_tensor(value))
        lengths.append(int(first.shape[0]))
        if part is not template:
            del part
    total = sum(lengths)
    merged = {
        key: torch.empty((total, *value.shape[1:]), dtype=value.dtype)
        for key, value in template.items()
        if torch.is_tensor(value)
    }
    del template
    gc.collect()

    offset = 0
    for chunk_id, (path, length) in enumerate(zip(paths, lengths)):
        part = torch.load(path, map_location="cpu")
        for key, target in merged.items():
            target[offset:offset + length].copy_(part[key])
        offset += length
        del part
        print(f"[merge-{role}] {chunk_id + 1:02d}/{args.num_chunks} samples={offset}/{total}", flush=True)
    base.attach_coord_meta(merged, merged, args)
    merged["__meta__"]["tensor_role"] = "old_scene_inputs" if role == "old" else "new_future_gt"
    _atomic_torch_save(merged, output_path)
    return merged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-info-pkl", default=os.path.join(HIPAD_ROOT, "data/infos/b2d_infos_train.pkl"))
    parser.add_argument("--chunk-dir", default=os.path.join(HIPAD_ROOT, "outputs/inference_chunks"))
    parser.add_argument("--info-dir", default=os.path.join(HIPAD_ROOT, "data/infos/chunks"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-name", default="hipad_stage2_b2d_train_pnn_static_old.pt")
    parser.add_argument("--new-name", default="hipad_stage2_b2d_train_pnn_static_true_gt.pt")
    parser.add_argument("--part-dir", default=None)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PNN_CONVERT_WORKERS", "4")))
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--plan-key", default="plan_temp_2hz")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--actor-motion-source-dt", type=float, default=0.1)
    parser.add_argument("--split-group", type=int, default=5)
    parser.add_argument("--gt-source", choices=["true_gt", "hipad_plan"], default="true_gt")
    parser.add_argument("--route-source", choices=["navigation", "hipad_plan"], default="hipad_plan")
    parser.add_argument("--coord-convention", choices=["hipad_xy", "pnn_xy"], default="pnn_xy")
    parser.add_argument("--target-forward-offset", type=float, default=0.0)
    parser.add_argument("--reference-forward-offset", type=float, default=0.0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--keep-parts", action="store_true")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, args.num_chunks))
    args.part_dir = args.part_dir or os.path.join(args.output_dir, ".parallel_parts")
    return args


def main() -> None:
    global _INFO_INDEX, _CFG
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.part_dir, exist_ok=True)
    for chunk_id in range(args.num_chunks):
        for path in (
            os.path.join(args.chunk_dir, f"hipad_stage2_b2d_train_chunk_{chunk_id:03d}.pkl"),
            os.path.join(args.info_dir, f"b2d_infos_train_chunk_{chunk_id:03d}.pkl"),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(path)

    print(f"[parallel-convert] loading shared infos: {args.all_info_pkl}", flush=True)
    started = time.time()
    all_infos = base.safe_pickle_load(args.all_info_pkl)
    _INFO_INDEX = base.build_info_index(all_infos) if args.gt_source == "true_gt" else None
    _CFG = SimpleNamespace(**vars(args))
    print(f"[parallel-convert] infos ready in {time.time() - started:.1f}s; workers={args.workers}", flush=True)

    ctx = mp.get_context("fork")
    completed = 0
    with ctx.Pool(processes=args.workers) as pool:
        for chunk_id, count, valid, status in pool.imap_unordered(_convert_chunk, range(args.num_chunks)):
            completed += 1
            print(
                f"[parallel-convert] {completed:02d}/{args.num_chunks} chunk={chunk_id:03d} "
                f"samples={count} valid_gt={valid} status={status}",
                flush=True,
            )

    old_path = os.path.join(args.output_dir, args.old_name)
    new_path = os.path.join(args.output_dir, args.new_name)
    old = _merge_parts(args, "old", old_path)
    new = _merge_parts(args, "new", new_path)
    print(f"[parallel-convert] saved old: {old_path}", flush=True)
    print(f"[parallel-convert] saved new: {new_path}", flush=True)
    print(f"[parallel-convert] samples={old['ego_state'].shape[0]} valid_gt={int(new['ego_future_gt_valid_mask'].sum())}", flush=True)
    if not args.keep_parts:
        shutil.rmtree(args.part_dir)
        print(f"[parallel-convert] removed temporary parts: {args.part_dir}", flush=True)


if __name__ == "__main__":
    main()
