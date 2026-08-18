#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path

import mmcv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(PROJECT_ROOT / "data/infos/b2d_infos_train.pkl"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data/infos/chunks"))
    parser.add_argument("--num-chunks", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    infos = mmcv.load(args.src)
    total = len(infos)
    chunk_size = int(math.ceil(total / args.num_chunks))
    manifest = []

    for chunk_id in range(args.num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, total)
        if start >= end:
            break
        path = os.path.join(args.out_dir, f"b2d_infos_train_chunk_{chunk_id:03d}.pkl")
        mmcv.dump(infos[start:end], path)
        manifest.append({"chunk_id": chunk_id, "start": start, "end": end, "path": path})
        print(f"chunk {chunk_id:03d}: [{start}, {end}) -> {path}")

    mmcv.dump(manifest, os.path.join(args.out_dir, "manifest.pkl"))
    print(f"wrote {len(manifest)} chunks for {total} samples")


if __name__ == "__main__":
    main()
