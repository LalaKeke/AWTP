#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


INFRACTION_KEYS = [
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
]


def make_skipped_record(route_id, target_idx, reason, event_time):
    return {
        "index": target_idx,
        "route_id": route_id,
        "scenario_name": None,
        "weather_id": None,
        "save_name": None,
        "status": "Failed - Watchdog skipped",
        "num_infractions": 0,
        "infractions": {key: [] for key in INFRACTION_KEYS},
        "scores": {
            "score_route": 0,
            "score_penalty": 1.0,
            "score_composed": 0,
        },
        "meta": {
            "route_length": 0,
            "duration_game": 0,
            "duration_system": 0,
            "watchdog_skip_reason": reason,
            "watchdog_skip_time": event_time,
        },
        "town_name": None,
    }


def load_events(path):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"skip malformed jsonl line {line_no}: {exc}")
    return events


def repair_checkpoint(checkpoint_path, events):
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    checkpoint = data.setdefault("_checkpoint", {})
    progress = checkpoint.setdefault("progress", [0, 0])
    records = checkpoint.setdefault("records", [])

    changed = False
    for event in events:
        route_id = event["route_id"]
        target_idx = None
        for i, record in enumerate(records):
            if record.get("route_id") == route_id:
                target_idx = i
                break

        if target_idx is None:
            if "route_index" not in event:
                print(f"{checkpoint_path}: route {route_id} not found and event has no route_index")
                continue
            target_idx = int(event["route_index"])
            if target_idx > len(records):
                print(
                    f"{checkpoint_path}: route {route_id} index {target_idx} is beyond "
                    f"records length {len(records)}; cannot safely insert"
                )
                continue

            reason = event.get("reason", "watchdog skipped repeated CARLA crash")
            record = make_skipped_record(route_id, target_idx, reason, event.get("time", "unknown"))
            records.insert(target_idx, record)
            for i, item in enumerate(records):
                item["index"] = i
            changed = True
        else:
            record = records[target_idx]

        reason = event.get("reason", "watchdog skipped repeated CARLA crash")
        if record.get("status") != "Failed - Watchdog skipped":
            record["status"] = "Failed - Watchdog skipped"
            changed = True

        record["index"] = target_idx
        record["num_infractions"] = record.get("num_infractions", 0)
        record.setdefault("scores", {})
        if record["scores"].get("score_route") != 0:
            record["scores"]["score_route"] = 0
            changed = True
        record["scores"]["score_penalty"] = record["scores"].get("score_penalty", 1.0)
        if record["scores"].get("score_composed") != 0:
            record["scores"]["score_composed"] = 0
            changed = True

        record.setdefault("meta", {})
        if record["meta"].get("watchdog_skip_reason") != reason:
            record["meta"]["watchdog_skip_reason"] = reason
            changed = True
        record["meta"].setdefault("watchdog_skip_time", event.get("time", "unknown"))

        if len(progress) < 2:
            progress[:] = [target_idx + 1, max(target_idx + 1, len(records))]
            changed = True
        elif int(progress[0]) < target_idx + 1:
            progress[0] = target_idx + 1
            changed = True

    if data.get("entry_status") == "Crashed":
        data["entry_status"] = "Started"
        changed = True
    if data.get("eligible", True):
        data["eligible"] = False
        changed = True

    if changed:
        tmp_path = str(checkpoint_path) + ".repair_tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, checkpoint_path)

    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jsonl",
        nargs="?",
        default="evaluation/hipad_b2d_stage2_pnn/watchdog_skipped_routes.jsonl",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    events = load_events(jsonl_path)
    grouped = {}
    for event in events:
        checkpoint = Path(event["checkpoint"])
        grouped.setdefault(checkpoint, {})
        grouped[checkpoint][event["route_id"]] = event

    changed_count = 0
    for checkpoint, route_events in sorted(grouped.items()):
        changed = repair_checkpoint(checkpoint, list(route_events.values()))
        print(f"{checkpoint}: {'repaired' if changed else 'already ok'}")
        changed_count += int(changed)

    print(f"checked {len(grouped)} checkpoint files, repaired {changed_count}")


if __name__ == "__main__":
    main()
