#!/usr/bin/env python3
"""Add aligned HiPAD 2 Hz plans to an existing static PNN GT tensor."""

from __future__ import annotations

import argparse
import gc
import io
import multiprocessing as mp
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch


def load_from_bytes_cpu(buffer: bytes):
    return torch.load(io.BytesIO(buffer), map_location="cpu")


def safe_pickle_load(path: Path):
    original = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = load_from_bytes_cpu
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    finally:
        torch.storage._load_from_bytes = original


def atomic_save(value, path: Path) -> None:
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(value, temp)
    os.replace(temp, path)


def hipad_to_pnn(points: np.ndarray) -> np.ndarray:
    return np.stack([points[..., 1], -points[..., 0]], axis=-1).astype(
        np.float32
    )


def extract_chunk(task):
    chunk_id, chunk_dir, part_dir, plan_key = task
    torch.set_num_threads(1)
    part_path = Path(part_dir) / f"chunk_{chunk_id:03d}_hipad_plan.pt"
    if part_path.exists():
        plans = torch.load(part_path, map_location="cpu")
        if torch.is_tensor(plans) and plans.ndim == 3 and plans.shape[1:] == (6, 2):
            return chunk_id, int(plans.shape[0]), "cached"

    chunk_path = (
        Path(chunk_dir)
        / f"hipad_stage2_b2d_train_chunk_{chunk_id:03d}.pkl"
    )
    records = safe_pickle_load(chunk_path)

    plans = []
    for record in records:
        plan = record.get("img_bbox", {}).get(plan_key)
        if plan is None:
            continue
        if torch.is_tensor(plan):
            plan = plan.detach().cpu().numpy()
        plan = np.asarray(plan, dtype=np.float32)
        if plan.shape != (6, 2):
            continue
        plans.append(hipad_to_pnn(plan))
    if not plans:
        raise RuntimeError(f"no valid plans in {chunk_path}")

    tensor = torch.from_numpy(np.stack(plans))
    atomic_save(tensor, part_path)
    return chunk_id, int(tensor.shape[0]), "converted"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-new-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-dir", required=True)
    parser.add_argument("--part-dir", default=None)
    parser.add_argument("--plan-key", default="plan_temp_2hz")
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part_dir = Path(
        args.part_dir
        or output.parent / ".hipad_plan_parts"
    )
    part_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (i, args.chunk_dir, str(part_dir), args.plan_key)
        for i in range(args.num_chunks)
    ]
    workers = max(1, min(args.workers, args.num_chunks))
    with mp.get_context("fork").Pool(workers) as pool:
        done = 0
        for chunk_id, count, status in pool.imap_unordered(
            extract_chunk, tasks
        ):
            done += 1
            print(
                f"[hipad-plan] {done:02d}/{args.num_chunks} "
                f"chunk={chunk_id:03d} samples={count} status={status}",
                flush=True,
            )

    parts = [
        torch.load(
            part_dir / f"chunk_{i:03d}_hipad_plan.pt",
            map_location="cpu",
        )
        for i in range(args.num_chunks)
    ]
    plans = torch.cat(parts, dim=0)
    del parts
    data = torch.load(args.base_new_data, map_location="cpu")
    expected = int(data["ego_future_gt_valid_mask"].shape[0])
    if plans.shape != (expected, 6, 2):
        raise RuntimeError(
            f"alignment mismatch: plans={tuple(plans.shape)} "
            f"expected=({expected}, 6, 2)"
        )
    if not torch.isfinite(plans).all():
        raise RuntimeError("hipad_plan_2hz contains non-finite values")

    data["hipad_plan_2hz"] = plans
    meta = data.get("__meta__")
    if isinstance(meta, dict):
        meta["hipad_plan_2hz_semantics"] = (
            "HiPAD plan at 0.5 s intervals in pnn_xy; supervision only"
        )
    atomic_save(data, output)
    print(f"[hipad-plan] saved: {output}", flush=True)
    print(f"[hipad-plan] samples={expected}", flush=True)
    del data, plans
    gc.collect()
    if not args.keep_parts:
        shutil.rmtree(part_dir)


if __name__ == "__main__":
    main()
