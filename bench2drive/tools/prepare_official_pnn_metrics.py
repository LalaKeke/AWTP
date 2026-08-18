#!/usr/bin/env python3
import argparse
import collections
import copy
import glob
import json
import os
import re
import shutil
from pathlib import Path


SPLIT_RE = re.compile(r".+_(\d+)\.json$")


def split_index(path):
    match = SPLIT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    split_files = []
    for path in result_dir.glob("*.json"):
        index = split_index(path)
        if index is not None and 0 <= index <= 15:
            split_files.append((index, path))
    split_files.sort()

    indices = [index for index, _ in split_files]
    if indices != list(range(16)):
        raise RuntimeError(f"Expected split JSON 0..15, got {indices}")

    total_records = 0
    for index, path in split_files:
        with path.open() as file:
            checkpoint = json.load(file)["_checkpoint"]
        records = checkpoint["records"]
        progress = checkpoint["progress"]
        if progress[0] != progress[1] or progress[0] != len(records):
            raise RuntimeError(
                f"Split {index} is incomplete: records={len(records)} progress={progress}"
            )
        total_records += len(records)
    if total_records != 220:
        raise RuntimeError(f"Expected 220 records, got {total_records}")

    staging = result_dir / "official_metric_json"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for _, source in split_files:
        os.symlink(source, staging / source.name)

    tools_dir = Path(__file__).resolve().parent
    import sys
    sys.path.insert(0, str(tools_dir))
    from merge_route_json import merge_route_json

    merge_route_json(str(staging))
    merged_source = staging / "merged.json"
    merged_path = result_dir / "merged.json"
    shutil.copy2(merged_source, merged_path)

    with merged_path.open() as file:
        merged = json.load(file)
    records = merged["_checkpoint"]["records"]

    metric_links = result_dir / "official_metric_links"
    if metric_links.exists():
        shutil.rmtree(metric_links)
    metric_links.mkdir()

    metric_records = []
    missing_metric = []
    for record in records:
        save_name = record.get("save_name")
        matches = (
            glob.glob(str(result_dir / f"*_{save_name}"))
            if save_name
            else []
        )
        matches = [
            Path(path) for path in matches
            if (Path(path) / "metric_info.json").is_file()
        ]
        if len(matches) == 1:
            os.symlink(matches[0], metric_links / save_name)
            metric_records.append(record)
        else:
            missing_metric.append((record["route_id"], record["status"]))

    metric_merged = copy.deepcopy(merged)
    metric_merged["_checkpoint"]["records"] = metric_records
    metric_merged_path = result_dir / "merged_metric_valid.json"
    with metric_merged_path.open("w") as file:
        json.dump(metric_merged, file, indent=2)

    statuses = collections.Counter(record["status"] for record in records)
    print(f"records={len(records)}")
    print(f"driving_score={merged['driving score']:.6f}")
    print(f"success_rate={merged['success rate'] * 100:.6f}%")
    print(f"statuses={dict(statuses)}")
    print(
        f"metric_info_coverage={len(metric_records)}/{len(records)}; "
        f"missing={len(missing_metric)}"
    )
    print(f"merged={merged_path}")
    print(f"metric_merged={metric_merged_path}")
    print(f"metric_links={metric_links}")


if __name__ == "__main__":
    main()
