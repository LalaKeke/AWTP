#!/usr/bin/env python3
import argparse
import io
import os
import pickle
from pathlib import Path

import torch
import torch.storage


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


def _payload_without_proto_and_stop(obj) -> bytes:
    payload = pickle.dumps(obj, protocol=2)
    if not payload.startswith(b"\x80\x02") or not payload.endswith(b"."):
        raise RuntimeError("unexpected pickle payload format")
    return payload[2:-1]


def stream_dump_list(chunk_paths, out_path: str) -> int:
    total = 0
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(b"\x80\x02](")
        for chunk_id, path in chunk_paths:
            part = safe_pickle_load(path)
            print(f"chunk {chunk_id:03d}: {len(part)} samples", flush=True)
            for item in part:
                f.write(_payload_without_proto_and_stop(item))
            total += len(part)
            del part
        f.write(b"e.")
    os.replace(tmp_path, out_path)
    return total


def main():
    project_root = Path(
        os.environ.get("HIPAD_PNN_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", default=str(project_root / "outputs" / "inference_chunks"))
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--out", default=str(project_root / "outputs" / "hipad_train_outputs.pkl"))
    args = parser.parse_args()

    chunk_paths = []
    missing = []
    for chunk_id in range(args.num_chunks):
        path = os.path.join(args.chunk_dir, f"hipad_stage2_b2d_train_chunk_{chunk_id:03d}.pkl")
        if not os.path.exists(path):
            missing.append(path)
            continue
        chunk_paths.append((chunk_id, path))

    if missing:
        raise FileNotFoundError("missing chunk outputs:\n" + "\n".join(missing))

    total = stream_dump_list(chunk_paths, args.out)
    print(f"merged {total} samples -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
