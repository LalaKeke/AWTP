#!/usr/bin/env python3
"""Fast actor-only rebuild for the corrected HiP-AD -> PNN time contract."""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import shutil
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIPAD_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MAP_ROOT = os.path.dirname(HIPAD_ROOT)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, MAP_ROOT)

import convert_hipad_chunk_to_pnn_static_pt as base
from hipad_pnn_adapter import agent_states_hipad_to_pnn, pack_agents
from pnn_temporal_alignment import ALIGNMENT_VERSION, HIPAD_MOTION_DT, PNN_ACTOR_TIMES


_CFG = None


def _part_path(part_dir: str, chunk_id: int) -> str:
    return os.path.join(part_dir, f"chunk_{chunk_id:03d}_actors.pt")


def _atomic_save(data: Dict, path: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(data, temporary)
    os.replace(temporary, path)


def _convert_chunk(chunk_id: int) -> Tuple[int, int, str]:
    torch.set_num_threads(1)
    output = _part_path(_CFG.part_dir, chunk_id)
    if _CFG.resume and os.path.isfile(output):
        cached = torch.load(output, map_location="cpu")
        if cached.get("alignment_version") == ALIGNMENT_VERSION:
            return chunk_id, int(cached["ped_states"].shape[0]), "cached"

    path = os.path.join(
        _CFG.chunk_dir, f"hipad_stage2_b2d_train_chunk_{chunk_id:03d}.pkl"
    )
    records = base.safe_pickle_load(path)
    ped_states = []
    veh_states = []
    ped_masks = []
    veh_masks = []
    for record in records:
        img_bbox = record["img_bbox"]
        plan = base.to_numpy(img_bbox.get(_CFG.plan_key, []))
        if plan.shape != (6, 2):
            continue
        pedestrians, vehicles, _static = base.extract_agents(
            img_bbox,
            score_thr=_CFG.score_thr,
            actor_motion_source_dt=_CFG.actor_motion_source_dt,
        )
        ped, ped_mask = pack_agents(pedestrians, base.NUM_PEDS)
        veh, veh_mask = pack_agents(vehicles, base.NUM_VEHS)
        if _CFG.coord_convention == "pnn_xy":
            ped = agent_states_hipad_to_pnn(ped)
            veh = agent_states_hipad_to_pnn(veh)
        ped_states.append(ped)
        veh_states.append(veh)
        ped_masks.append(ped_mask)
        veh_masks.append(veh_mask)

    part = {
        "ped_states": torch.as_tensor(np.stack(ped_states), dtype=torch.float32),
        "veh_states": torch.as_tensor(np.stack(veh_states), dtype=torch.float32),
        "ped_mask": torch.as_tensor(np.stack(ped_masks), dtype=torch.bool),
        "veh_mask": torch.as_tensor(np.stack(veh_masks), dtype=torch.bool),
        "alignment_version": ALIGNMENT_VERSION,
    }
    _atomic_save(part, output)
    return chunk_id, len(ped_states), "converted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-old", required=True)
    parser.add_argument("--output-old", required=True)
    parser.add_argument("--chunk-dir", default=os.path.join(HIPAD_ROOT, "outputs/inference_chunks"))
    parser.add_argument("--part-dir", default=None)
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PNN_CONVERT_WORKERS", "4")))
    parser.add_argument("--plan-key", default="plan_temp_2hz")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--actor-motion-source-dt", type=float, default=HIPAD_MOTION_DT)
    parser.add_argument("--coord-convention", choices=["hipad_xy", "pnn_xy"], default="pnn_xy")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--keep-parts", action="store_true")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, args.num_chunks))
    args.part_dir = args.part_dir or os.path.join(os.path.dirname(args.output_old), ".actor_alignment_parts")
    return args


def main() -> None:
    global _CFG
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_old), exist_ok=True)
    os.makedirs(args.part_dir, exist_ok=True)
    _CFG = args

    start = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        completed = 0
        for chunk_id, count, status in pool.imap_unordered(_convert_chunk, range(args.num_chunks)):
            completed += 1
            print(
                f"[actor-align] {completed:02d}/{args.num_chunks} chunk={chunk_id:03d} "
                f"samples={count} status={status}",
                flush=True,
            )

    print(f"[actor-align] loading base tensors: {args.base_old}", flush=True)
    old = torch.load(args.base_old, map_location="cpu")
    offset = 0
    max_position_yaw_error = 0.0
    speed_changed = 0
    valid_actor_count = 0
    for chunk_id in range(args.num_chunks):
        part = torch.load(_part_path(args.part_dir, chunk_id), map_location="cpu")
        length = int(part["ped_states"].shape[0])
        end = offset + length
        for state_key, mask_key in (("ped_states", "ped_mask"), ("veh_states", "veh_mask")):
            old_mask = old[mask_key][offset:end]
            new_mask = part[mask_key]
            if not torch.equal(old_mask, new_mask):
                raise RuntimeError(f"{mask_key} changed in chunk {chunk_id:03d}")
            valid = new_mask
            if valid.any():
                error = (old[state_key][offset:end, :, :3] - part[state_key][:, :, :3]).abs()[valid]
                max_position_yaw_error = max(max_position_yaw_error, float(error.max()))
                speed_changed += int(
                    ((old[state_key][offset:end, :, 3] - part[state_key][:, :, 3]).abs() > 1e-4)[valid].sum()
                )
                valid_actor_count += int(valid.sum())
            old[state_key][offset:end].copy_(part[state_key])
        offset = end
        del part
    if offset != int(old["ego_state"].shape[0]):
        raise RuntimeError(f"sample count mismatch: rebuilt={offset}, base={old['ego_state'].shape[0]}")
    if max_position_yaw_error > 1e-4:
        raise RuntimeError(f"actor identity/order mismatch: max xyz/yaw error={max_position_yaw_error}")

    metadata = dict(old.get("__meta__", {}))
    metadata.update(
        {
            "actor_motion_alignment": ALIGNMENT_VERSION,
            "hipad_actor_motion_dt": float(args.actor_motion_source_dt),
            "pnn_actor_target_times": [float(x) for x in PNN_ACTOR_TIMES],
        }
    )
    old["__meta__"] = metadata
    _atomic_save(old, args.output_old)
    print(
        f"[actor-align] saved={args.output_old} samples={offset} actors={valid_actor_count} "
        f"speed_changed={speed_changed} elapsed={time.time() - start:.1f}s",
        flush=True,
    )
    del old
    gc.collect()
    if not args.keep_parts:
        shutil.rmtree(args.part_dir)


if __name__ == "__main__":
    main()
