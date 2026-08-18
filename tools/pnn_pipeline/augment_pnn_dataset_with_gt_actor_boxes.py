#!/usr/bin/env python3
"""Add sample-aligned official GT actor boxes to an existing PNN dataset."""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIPAD_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
import convert_hipad_chunk_to_pnn_pt as conv


_INFO_INDEX = None
_CFG = None


def part_path(chunk_id):
    return os.path.join(_CFG.part_dir, f"chunk_{chunk_id:03d}_gt_actor.pt")


def atomic_save(value, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(value, tmp)
    os.replace(tmp, path)


def convert_chunk(chunk_id):
    torch.set_num_threads(1)
    output = part_path(chunk_id)
    if _CFG.resume and os.path.exists(output):
        part = torch.load(output, map_location="cpu")
        required = {
            "gt_actor_boxes_2hz",
            "gt_actor_mask_2hz",
            "hipad_plan_2hz",
            "official_hipad_obj_box_col",
        }
        if required.issubset(part):
            return chunk_id, len(part["official_fut_valid_mask"]), int(part["gt_actor_truncated"].sum()), "cached"

    records = conv.safe_pickle_load(os.path.join(_CFG.chunk_dir, f"hipad_stage2_b2d_train_chunk_{chunk_id:03d}.pkl"))
    raw_infos = conv.safe_pickle_load(os.path.join(_CFG.info_dir, f"b2d_infos_train_chunk_{chunk_id:03d}.pkl"))
    infos = conv.split_group_order(raw_infos, _CFG.split_group)
    boxes_all, masks_all, plans_all, truncated_all, official_all, valid_all = [], [], [], [], [], []
    for record, info in zip(records, infos):
        plan = record.get("img_bbox", {}).get(_CFG.plan_key)
        if plan is None or conv.to_numpy(plan).shape != (6, 2):
            continue
        boxes, mask, truncated = conv.reconstruct_gt_actor_boxes(
            info,
            _INFO_INDEX,
            coord_convention=_CFG.coord_convention,
        )
        metrics = record.get("metric_results", {})
        boxes_all.append(boxes)
        masks_all.append(mask)
        plan_np = conv.to_numpy(plan)
        if _CFG.coord_convention == "pnn_xy":
            plan_np = conv.hipad_points_to_pnn(plan_np)
        plans_all.append(plan_np)
        truncated_all.append(truncated)
        official_all.append([metrics.get(f"plan_obj_box_col_{s}s", 0.0) for s in (1, 2, 3)])
        valid_all.append(bool(metrics.get("fut_valid_flag", False)))

    part = {
        "gt_actor_boxes_2hz": torch.as_tensor(np.stack(boxes_all), dtype=torch.float32),
        "gt_actor_mask_2hz": torch.as_tensor(np.stack(masks_all), dtype=torch.bool),
        "hipad_plan_2hz": torch.as_tensor(np.stack(plans_all), dtype=torch.float32),
        "gt_actor_truncated": torch.as_tensor(np.asarray(truncated_all), dtype=torch.int16),
        "official_hipad_obj_box_col": torch.as_tensor(np.asarray(official_all), dtype=torch.float32),
        "official_fut_valid_mask": torch.as_tensor(np.asarray(valid_all), dtype=torch.bool),
    }
    atomic_save(part, output)
    return chunk_id, len(valid_all), int(sum(truncated_all)), "converted"


def merge(args):
    paths = [os.path.join(args.part_dir, f"chunk_{i:03d}_gt_actor.pt") for i in range(args.num_chunks)]
    lengths = []
    for path in paths:
        part = torch.load(path, map_location="cpu")
        lengths.append(len(part["official_fut_valid_mask"]))
        del part
    total = sum(lengths)
    base_data = torch.load(args.base_new_data, map_location="cpu")
    if len(base_data["ego_future_gt_valid_mask"]) != total:
        raise RuntimeError(
            f"sample count mismatch: base={len(base_data['ego_future_gt_valid_mask'])} augmentation={total}"
        )

    first = torch.load(paths[0], map_location="cpu")
    merged = {
        key: torch.empty((total, *value.shape[1:]), dtype=value.dtype)
        for key, value in first.items()
    }
    del first
    offset = 0
    for chunk_id, (path, length) in enumerate(zip(paths, lengths)):
        part = torch.load(path, map_location="cpu")
        for key in merged:
            merged[key][offset:offset + length].copy_(part[key])
        offset += length
        del part
        print(f"[merge] {chunk_id + 1:02d}/{args.num_chunks} samples={offset}/{total}", flush=True)

    base_data.update(merged)
    meta = base_data.get("__meta__", {})
    if isinstance(meta, dict):
        meta.update(conv.build_coord_meta(args))
        meta["tensor_role"] = "new_future_gt_with_official_actor_boxes"
        meta["hipad_plan_2hz_semantics"] = (
            "full HiPAD plan at 0.5s intervals in coord_convention; training-only risk label, "
            "never a WeightNet input"
        )
        base_data["__meta__"] = meta
    atomic_save(base_data, args.output)
    print(f"saved: {args.output}", flush=True)
    print(f"samples={total} valid={int(merged['official_fut_valid_mask'].sum())}", flush=True)
    print(f"truncated_actors={int(merged['gt_actor_truncated'].sum())}", flush=True)
    del base_data, merged
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-info-pkl", default=os.path.join(HIPAD_ROOT, "data/infos/b2d_infos_train.pkl"))
    parser.add_argument("--chunk-dir", default=os.path.join(HIPAD_ROOT, "outputs/inference_chunks"))
    parser.add_argument("--info-dir", default=os.path.join(HIPAD_ROOT, "data/infos/chunks"))
    parser.add_argument("--base-new-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--part-dir", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--split-group", type=int, default=5)
    parser.add_argument("--plan-key", default="plan_temp_2hz")
    parser.add_argument("--coord-convention", choices=["pnn_xy", "hipad_xy"], default="pnn_xy")
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--parts-only", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    # Metadata compatibility with build_coord_meta.
    parser.add_argument("--route-source", default="hipad_plan")
    parser.add_argument("--gt-source", default="true_gt")
    parser.add_argument("--target-forward-offset", type=float, default=0.0)
    parser.add_argument("--reference-forward-offset", type=float, default=0.0)
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, args.num_chunks))
    args.part_dir = args.part_dir or os.path.join(os.path.dirname(args.output), ".gt_actor_parts")
    return args


def main():
    global _INFO_INDEX, _CFG
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.part_dir, exist_ok=True)
    print(f"loading shared infos: {args.all_info_pkl}", flush=True)
    all_infos = conv.safe_pickle_load(args.all_info_pkl)
    _INFO_INDEX = conv.build_info_index(all_infos)
    _CFG = SimpleNamespace(**vars(args))
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        done = 0
        for chunk_id, count, truncated, status in pool.imap_unordered(convert_chunk, range(args.num_chunks)):
            done += 1
            print(
                f"[augment] {done:02d}/{args.num_chunks} chunk={chunk_id:03d} "
                f"samples={count} truncated={truncated} status={status}",
                flush=True,
            )
    if args.parts_only:
        print(f"parts-only complete: {args.part_dir}", flush=True)
        return
    merge(args)
    if not args.keep_parts:
        shutil.rmtree(args.part_dir)


if __name__ == "__main__":
    main()
