#!/usr/bin/env python3
"""Build the minimal GT-solid lane sidecar required by Static-v3.1.

Unlike build_metric_aligned_supervision.py, this script does not rasterize
actors or evaluate occupancy. Static training already has corrected actor
boxes, so avoiding that duplicate work makes lane preparation substantially
faster.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
HIPAD_ROOT = SCRIPT_DIR.parents[1]
MAP_ROOT = HIPAD_ROOT.parent
sys.path[:0] = [str(MAP_ROOT), str(HIPAD_ROOT), str(SCRIPT_DIR)]

from build_metric_aligned_supervision import (  # noqa: E402
    as_tensor,
    key_hash,
    sample_key,
    solid_lane_points,
)
from convert_hipad_chunk_to_pnn_pt import (  # noqa: E402
    safe_pickle_load,
    split_group_order,
)


_DATASET = None
_KEY_TO_INDEX = None
_CFG = None
_CANDIDATE_MASK = None
_SEED_SUPERVISION = None
_BASE_CHUNK_COUNT = None


def atomic_save(data, path: Path):
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(data, tmp)
    os.replace(tmp, path)


def part_path(chunk_id: int) -> Path:
    return Path(_CFG.part_dir) / f"solid_lane_chunk_{chunk_id:03d}.pt"


def get_solid_map_geoms(index: int):
    """Run HiP-AD's exact map path; filtering happens after map construction."""
    return _DATASET.get_map_info(index)


def build_sample(info, candidate_index: int):
    key = sample_key(info)
    if key not in _KEY_TO_INDEX:
        raise KeyError(f"sample missing from configured train dataset: {key}")
    index = _KEY_TO_INDEX[key]
    indexed_info = _DATASET.data_infos[index]
    if sample_key(indexed_info) != key:
        raise RuntimeError(
            f"dataset key mismatch: requested={key}, got={sample_key(indexed_info)}"
        )
    if (
        _SEED_SUPERVISION is not None
        and bool(
            _SEED_SUPERVISION["gt_solid_lane_mask"][candidate_index]
            .bool()
            .any()
        )
    ):
        return {
            key: _SEED_SUPERVISION[key][candidate_index]
            for key in (
                "gt_solid_lane_points",
                "gt_solid_lane_mask",
                "gt_solid_lane_truncated",
                "sample_key_hash",
            )
        }
    if (
        _CANDIDATE_MASK is not None
        and not bool(_CANDIDATE_MASK[candidate_index])
    ):
        return {
            "gt_solid_lane_points": torch.zeros(
                _CFG.max_solid_lines,
                _CFG.lane_points,
                2,
                dtype=torch.float16,
            ),
            "gt_solid_lane_mask": torch.zeros(
                _CFG.max_solid_lines,
                dtype=torch.bool,
            ),
            "gt_solid_lane_truncated": torch.tensor(0, dtype=torch.int16),
            "sample_key_hash": torch.tensor(
                int(key_hash(info)),
                dtype=torch.int64,
            ),
        }

    # get_data_info() also builds detection boxes, all actor trajectories,
    # attributes, and map annotations. None are needed for this sidecar.
    # Calling only these two dataset methods preserves the exact planning/map
    # definitions while making the one-time conversion substantially cheaper.
    plan_data = _DATASET.get_plan_info(index)
    map_geoms = get_solid_map_geoms(index)
    gt_delta = as_tensor(plan_data["gt_ego_fut_trajs_2hz"], torch.float32)
    gt_plan = gt_delta.cumsum(dim=-2)[:6]
    lane, lane_mask, lane_truncated = solid_lane_points(
        {"map_geoms": map_geoms},
        gt_plan,
        _CFG.max_solid_lines,
        _CFG.lane_points,
    )
    return {
        "gt_solid_lane_points": torch.from_numpy(lane).to(torch.float16),
        "gt_solid_lane_mask": torch.from_numpy(lane_mask),
        "gt_solid_lane_truncated": torch.tensor(lane_truncated, dtype=torch.int16),
        "sample_key_hash": torch.tensor(int(key_hash(info)), dtype=torch.int64),
    }


def convert_chunk(chunk_id: int):
    torch.set_num_threads(1)
    output = part_path(chunk_id)
    required = {
        "gt_solid_lane_points",
        "gt_solid_lane_mask",
        "sample_key_hash",
    }
    if _CFG.resume and output.exists():
        cached = torch.load(output, map_location="cpu")
        if required.issubset(cached):
            return chunk_id, int(cached["sample_key_hash"].shape[0]), "cached"

    info_path = Path(_CFG.info_dir) / f"b2d_infos_train_chunk_{chunk_id:03d}.pkl"
    infos = split_group_order(safe_pickle_load(str(info_path)), _CFG.split_group)
    if _CFG.limit_per_chunk:
        infos = infos[: _CFG.limit_per_chunk]
    offset = chunk_id * _BASE_CHUNK_COUNT
    rows = [
        build_sample(info, offset + local_index)
        for local_index, info in enumerate(infos)
    ]
    merged = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    atomic_save(merged, output)
    return chunk_id, len(rows), "converted"


def merge_parts(args):
    paths = [
        Path(args.part_dir) / f"solid_lane_chunk_{chunk_id:03d}.pt"
        for chunk_id in range(args.num_chunks)
    ]
    parts = [torch.load(path, map_location="cpu") for path in paths]
    merged = {
        key: torch.cat([part[key] for part in parts], dim=0)
        for key in parts[0]
    }
    merged["__meta__"] = {
        "version": 1,
        "coord_convention": "pnn_xy",
        "source": "Bench2Drive map_geoms labels 1=Solid and 2=SolidSolid",
        "lane_semantics": "nearest GT solid lines sampled in pnn_xy",
        "max_solid_lines": args.max_solid_lines,
        "lane_points": args.lane_points,
        "candidate_mask": args.candidate_mask,
    }
    atomic_save(merged, Path(args.output))
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(HIPAD_ROOT / "projects/configs/hipad_b2d_stage2.py"),
    )
    parser.add_argument("--info-dir", default=str(HIPAD_ROOT / "data/infos/chunks"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--part-dir", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PNN_CONVERT_WORKERS", "12")),
    )
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--split-group", type=int, default=5)
    parser.add_argument("--max-solid-lines", type=int, default=16)
    parser.add_argument("--lane-points", type=int, default=20)
    parser.add_argument("--limit-per-chunk", type=int, default=0)
    parser.add_argument("--candidate-mask", default=None)
    parser.add_argument("--seed-output", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--keep-parts", action="store_true")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, args.num_chunks))
    args.part_dir = args.part_dir or str(Path(args.output).with_suffix(".parts"))
    return args


def main():
    global _DATASET, _KEY_TO_INDEX, _CFG
    global _CANDIDATE_MASK, _SEED_SUPERVISION, _BASE_CHUNK_COUNT
    args = parse_args()
    os.chdir(HIPAD_ROOT)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.part_dir).mkdir(parents=True, exist_ok=True)

    from mmcv import Config
    from mmdet.datasets import build_dataset
    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(args.config)
    _DATASET = build_dataset(cfg.data.train)
    _KEY_TO_INDEX = {
        sample_key(info): index
        for index, info in enumerate(_DATASET.data_infos)
    }
    if len(_KEY_TO_INDEX) != len(_DATASET.data_infos):
        raise RuntimeError("folder/frame_idx keys are not unique in train dataset")
    _CFG = SimpleNamespace(**vars(args))
    first_info_path = Path(args.info_dir) / "b2d_infos_train_chunk_000.pkl"
    _BASE_CHUNK_COUNT = len(
        split_group_order(
            safe_pickle_load(str(first_info_path)),
            args.split_group,
        )
    )
    if args.candidate_mask:
        candidate_data = torch.load(args.candidate_mask, map_location="cpu")
        _CANDIDATE_MASK = candidate_data["candidate_mask"].bool()
        if int(_CANDIDATE_MASK.shape[0]) != len(_DATASET.data_infos):
            raise ValueError(
                "candidate-mask/dataset sample count mismatch: "
                f"{int(_CANDIDATE_MASK.shape[0])} vs {len(_DATASET.data_infos)}"
            )
        print(
            "[solid-lane] candidate filtering enabled: "
            f"{int(_CANDIDATE_MASK.sum())}/{len(_CANDIDATE_MASK)} samples",
            flush=True,
        )
    if args.seed_output:
        _SEED_SUPERVISION = torch.load(args.seed_output, map_location="cpu")
        if int(_SEED_SUPERVISION["sample_key_hash"].shape[0]) != len(
            _DATASET.data_infos
        ):
            raise ValueError("seed-output/dataset sample count mismatch")
        print(
            "[solid-lane] seed reuse enabled: "
            f"{int(_SEED_SUPERVISION['gt_solid_lane_mask'].bool().any(dim=1).sum())} "
            "samples already have lines",
            flush=True,
        )

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for done, (chunk_id, count, status) in enumerate(
            pool.imap_unordered(convert_chunk, range(args.num_chunks)),
            start=1,
        ):
            print(
                f"[solid-lane] {done:02d}/{args.num_chunks} "
                f"chunk={chunk_id:03d} n={count} {status}",
                flush=True,
            )

    merged = merge_parts(args)
    count = int(merged["sample_key_hash"].shape[0])
    mask = merged["gt_solid_lane_mask"].bool()
    if not torch.isfinite(merged["gt_solid_lane_points"]).all():
        raise RuntimeError("generated solid-lane points contain NaN/Inf")
    print(
        f"[solid-lane] saved {args.output}; samples={count} "
        f"valid_lines={int(mask.sum())} "
        f"samples_with_lines={int(mask.any(dim=1).sum())}",
        flush=True,
    )
    if not args.keep_parts:
        shutil.rmtree(args.part_dir)
    del _DATASET
    gc.collect()


if __name__ == "__main__":
    main()
