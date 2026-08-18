#!/usr/bin/env python3
"""Select samples whose perceived static boxes are near the HiP-AD plan."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data", required=True)
    parser.add_argument("--new-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--distance", type=float, default=10.0)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()

    old = torch.load(args.old_data, map_location="cpu")
    new = torch.load(args.new_data, map_location="cpu")
    missing = {"static_states", "static_mask"}.difference(old)
    if missing:
        raise KeyError(f"old data missing keys: {sorted(missing)}")
    if "hipad_plan_2hz" not in new:
        raise KeyError("new data missing key: hipad_plan_2hz")

    count = int(old["static_states"].shape[0])
    if int(new["hipad_plan_2hz"].shape[0]) != count:
        raise ValueError("old/new sample count mismatch")
    candidate = torch.zeros(count, dtype=torch.bool)
    min_distance = torch.full((count,), float("inf"), dtype=torch.float32)
    for start in range(0, count, args.chunk_size):
        end = min(start + args.chunk_size, count)
        static_xy = old["static_states"][start:end, :, :2].float()
        static_mask = old["static_mask"][start:end].bool()
        hipad_plan = new["hipad_plan_2hz"][start:end].float()
        distance = torch.linalg.vector_norm(
            static_xy[:, :, None, :] - hipad_plan[:, None, :, :],
            dim=-1,
        )
        distance = distance.masked_fill(~static_mask[:, :, None], float("inf"))
        nearest = distance.amin(dim=(1, 2))
        min_distance[start:end] = nearest
        candidate[start:end] = nearest < args.distance

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate_mask": candidate,
            "min_static_to_hipad_plan_distance": min_distance.to(torch.float16),
            "__meta__": {
                "version": 1,
                "distance_threshold_m": args.distance,
                "semantics": (
                    "perceived static box center within threshold of any "
                    "HiP-AD 2 Hz plan point"
                ),
            },
        },
        output,
    )
    print(
        f"[static-lane-candidate] saved {output}; "
        f"selected={int(candidate.sum())}/{count} "
        f"({100.0 * candidate.float().mean():.2f}%) "
        f"distance={args.distance:.2f}m"
    )


if __name__ == "__main__":
    main()
