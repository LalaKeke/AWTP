#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from compare_closedloop_subset import load_records, summarize


METRICS = (
    "routes",
    "driving_score",
    "success_rate",
    "route_score",
    "collisions_vehicle",
    "collisions_layout",
    "outside_route_lanes",
    "vehicle_blocked",
    "tick_runtime",
)


def rank_key(summary):
    # Closed-loop objective only: completion first, then driving quality and
    # safety. Open-loop L2/ACR/CCR are deliberately absent.
    return (
        summary["success_rate"],
        summary["driving_score"],
        summary["route_score"],
        -summary["collisions_vehicle"],
        -summary["collisions_layout"],
        -summary["outside_route_lanes"],
        -summary["vehicle_blocked"],
        -summary["tick_runtime"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", nargs=3,
                        metavar=("NAME", "CHECKPOINT", "RESULT"), required=True)
    parser.add_argument("--hipad", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--splits", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection-out", required=True)
    args = parser.parse_args()

    entries = [(name, checkpoint, result) for name, checkpoint, result in args.candidate]
    roots = {name: result for name, _, result in entries}
    roots["parent_epoch19"] = args.parent
    roots["hipad"] = args.hipad
    records = {name: load_records(root, args.splits) for name, root in roots.items()}
    route_ids = set(records["hipad"])
    for name, value in records.items():
        if set(value) != route_ids:
            raise RuntimeError(f"{name} route IDs do not match HiP-AD")
    summaries = {name: summarize(value) for name, value in records.items()}

    hipad = summaries["hipad"]
    parent = summaries["parent_epoch19"]
    beats_ds_sr = lambda value, base: (
        value["success_rate"] > base["success_rate"]
        and value["driving_score"] > base["driving_score"]
    )
    hipad_eligible = [
        item for item in entries if beats_ds_sr(summaries[item[0]], hipad)
    ]
    parent_eligible = [
        item for item in entries if beats_ds_sr(summaries[item[0]], parent)
    ]
    winner_name, winner_checkpoint, _ = max(
        hipad_eligible or parent_eligible or entries,
        key=lambda item: rank_key(summaries[item[0]]),
    )
    winner = summaries[winner_name]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", *METRICS])
        for name in [*(item[0] for item in entries), "parent_epoch19", "hipad"]:
            writer.writerow([name, *(summaries[name][metric] for metric in METRICS)])

    beats_parent = rank_key(winner) > rank_key(parent)
    beats_hipad = beats_ds_sr(winner, hipad)
    selection = Path(args.selection_out)
    with selection.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "selected_model", "checkpoint", "beats_parent", "beats_hipad",
            "success_rate", "driving_score",
        ])
        writer.writerow([
            winner_name, winner_checkpoint, int(beats_parent), int(beats_hipad),
            winner["success_rate"], winner["driving_score"],
        ])

    for name, summary in summaries.items():
        print(f"{name}={summary}")
    print(f"selected={winner_name} checkpoint={winner_checkpoint}")
    print(f"beats_parent={beats_parent} beats_hipad={beats_hipad}")
    print(f"summary={output}")
    print(f"selection={selection}")


if __name__ == "__main__":
    main()
