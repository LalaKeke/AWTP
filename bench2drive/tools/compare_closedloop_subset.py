#!/usr/bin/env python3
import argparse
import collections
import csv
import glob
import json
from pathlib import Path


def load_records(root, splits):
    root = Path(root)
    records = {}
    for split in splits:
        paths = sorted(root.glob(f"*_{split}.json"))
        if len(paths) != 1:
            raise RuntimeError(f"expected one split {split} JSON in {root}, got {paths}")
        checkpoint = json.load(paths[0].open())["_checkpoint"]
        progress = checkpoint["progress"]
        if progress[0] != progress[1] or progress[0] != len(checkpoint["records"]):
            raise RuntimeError(f"incomplete split {split}: {progress}")
        for record in checkpoint["records"]:
            records[record["route_id"]] = record
    return records


def is_success(record):
    if record["status"] not in ("Completed", "Perfect"):
        return False
    return all(not value or key == "min_speed_infractions" for key, value in record["infractions"].items())


def summarize(records):
    values = list(records.values())
    n = len(values)
    infractions = collections.Counter()
    for record in values:
        for key, value in record["infractions"].items():
            infractions[key] += len(value)
    return {
        "routes": n,
        "driving_score": sum(float(r["scores"]["score_composed"]) for r in values) / n,
        "success_rate": 100.0 * sum(is_success(r) for r in values) / n,
        "route_score": sum(float(r["scores"]["score_route"]) for r in values) / n,
        "collisions_vehicle": infractions["collisions_vehicle"],
        "collisions_layout": infractions["collisions_layout"],
        "outside_route_lanes": infractions["outside_route_lanes"],
        "vehicle_blocked": infractions["vehicle_blocked"],
        "tick_runtime": sum(r["status"] == "Failed - TickRuntime" for r in values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--splits", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-superiority", action="store_true")
    parser.add_argument("--require-comprehensive", action="store_true")
    args = parser.parse_args()

    candidate_records = load_records(args.candidate, args.splits)
    baseline_records = load_records(args.baseline, args.splits)
    if set(candidate_records) != set(baseline_records):
        raise RuntimeError("candidate and baseline route IDs do not match")

    candidate = summarize(candidate_records)
    baseline = summarize(baseline_records)
    fixed = sum(
        not is_success(baseline_records[key]) and is_success(candidate_records[key])
        for key in candidate_records
    )
    regressed = sum(
        is_success(baseline_records[key]) and not is_success(candidate_records[key])
        for key in candidate_records
    )
    passed = (
        candidate["driving_score"] > baseline["driving_score"]
        and candidate["success_rate"] > baseline["success_rate"]
        and candidate["collisions_vehicle"] <= baseline["collisions_vehicle"]
        and candidate["collisions_layout"] <= baseline["collisions_layout"]
    )
    comprehensive_passed = (
        passed
        and candidate["outside_route_lanes"] <= baseline["outside_route_lanes"]
        and candidate["vehicle_blocked"] <= baseline["vehicle_blocked"]
        and candidate["tick_runtime"] <= baseline["tick_runtime"]
        and candidate["route_score"] >= baseline["route_score"]
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "candidate", "hipad", "delta"])
        for key in candidate:
            writer.writerow([key, candidate[key], baseline[key], candidate[key] - baseline[key]])
        writer.writerow(["paired_fixed_success", fixed, "", ""])
        writer.writerow(["paired_regressed_success", regressed, "", ""])
        writer.writerow(["superiority_pass", int(passed), "", ""])
        writer.writerow(["comprehensive_pass", int(comprehensive_passed), "", ""])

    print(f"candidate={candidate}")
    print(f"hipad={baseline}")
    print(f"paired fixed/regressed={fixed}/{regressed}")
    print(f"superiority_pass={passed}")
    print(f"comprehensive_pass={comprehensive_passed}")
    print(f"report={output}")
    if args.require_superiority and not passed:
        raise SystemExit(3)
    if args.require_comprehensive and not comprehensive_passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
