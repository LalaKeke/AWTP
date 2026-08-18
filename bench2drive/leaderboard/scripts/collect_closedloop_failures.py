#!/usr/bin/env python3
"""Export closed-loop failures and select only infrastructure failures for rerun.

This deliberately keeps policy failures (collisions, red lights, blocking, etc.)
out of the rerun list.  They remain in the CSV as training/debugging evidence,
while TickRuntime/watchdog/agent-start failures can be rerun reproducibly.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


RERUN_STATUSES = {
    "Failed - TickRuntime",
    "Failed - Watchdog skipped",
    "Failed - Agent couldn't be set up",
    "Failed - Agent crashed",
}


def is_success(record):
    if record.get("status") not in ("Completed", "Perfect"):
        return False
    return all(
        not value or key == "min_speed_infractions"
        for key, value in record.get("infractions", {}).items()
    )


def failure_reasons(record):
    reasons = []
    if record.get("status") not in ("Completed", "Perfect"):
        reasons.append(record.get("status", "missing status"))
    for name, values in record.get("infractions", {}).items():
        if values and name != "min_speed_infractions":
            reasons.append(f"{name}:{len(values)}")
    return "; ".join(reasons) or "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--checkpoint-prefix", required=True,
                        help="Original result JSON prefix, without _<split>.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", type=int,
                        default=list(range(16)))
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    rerun_events = []
    status_counts = Counter()
    infraction_counts = Counter()

    for split in args.splits:
        path = result_dir / f"{args.checkpoint_prefix}_{split}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = json.loads(path.read_text(encoding="utf-8"))["_checkpoint"]
        progress = checkpoint.get("progress", [0, 0])
        records = checkpoint.get("records", [])
        if progress[0] != progress[1] or progress[0] != len(records):
            raise RuntimeError(f"incomplete split {split}: progress={progress}, records={len(records)}")
        for record_index, record in enumerate(records):
            if is_success(record):
                continue
            status = record.get("status", "missing status")
            infractions = record.get("infractions", {})
            for name, values in infractions.items():
                infraction_counts[name] += len(values)
            status_counts[status] += 1
            eligible = status in RERUN_STATUSES
            row = {
                "route_id": record.get("route_id"),
                "split": split,
                "record_index": record_index,
                "status": status,
                "rerun_eligible": int(eligible),
                "driving_score": record.get("scores", {}).get("score_composed"),
                "route_score": record.get("scores", {}).get("score_route"),
                "failure_reasons": failure_reasons(record),
                "collisions_vehicle": len(infractions.get("collisions_vehicle", [])),
                "collisions_layout": len(infractions.get("collisions_layout", [])),
                "outside_route_lanes": len(infractions.get("outside_route_lanes", [])),
                "vehicle_blocked": len(infractions.get("vehicle_blocked", [])),
                "red_light": len(infractions.get("red_light", [])),
                "stop_infraction": len(infractions.get("stop_infraction", [])),
            }
            failures.append(row)
            if eligible:
                rerun_events.append({
                    "route_id": record["route_id"],
                    "split": split,
                    "route_index": record_index,
                    "checkpoint": str(path),
                    "source_status": status,
                    "reason": "infrastructure/runtime failure selected for isolated rerun",
                })

    failures.sort(key=lambda row: (row["split"], row["record_index"]))
    rerun_events.sort(key=lambda row: (row["split"], row["route_index"]))
    fields = list(failures[0]) if failures else ["route_id", "split", "record_index", "status", "rerun_eligible"]
    with (output_dir / "failure_routes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    with (output_dir / "rerun_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for event in rerun_events:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    summary = {
        "result_dir": str(result_dir),
        "splits": args.splits,
        "failed_or_infraction_routes": len(failures),
        "rerun_count": len(rerun_events),
        "rerun_statuses": sorted(RERUN_STATUSES),
        "status_counts": dict(sorted(status_counts.items())),
        "infraction_counts": dict(sorted(infraction_counts.items())),
    }
    (output_dir / "failure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(f"failure_csv={output_dir / 'failure_routes.csv'}")
    print(f"rerun_candidates={output_dir / 'rerun_candidates.jsonl'}")


if __name__ == "__main__":
    main()
