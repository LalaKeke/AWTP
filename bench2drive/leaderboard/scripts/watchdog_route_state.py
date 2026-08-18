#!/usr/bin/env python3
"""Small, atomic checkpoint operations used by watchdog_hipad_eval.sh."""

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET


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


def load_json(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_routes(path):
    root = ET.parse(path).getroot()
    return list(root.iter("route"))


def route_info(route, repetition=0):
    scenario = route.find("./scenarios/scenario")
    weathers = route.findall("./weathers/weather")
    weather_id = None
    if weathers:
        weather_id = weathers[0].get("id") or weathers[0].get("route_percentage") or "0"
    route_number = route.get("id")
    return {
        "route_id": f"RouteScenario_{route_number}_rep{repetition}",
        "scenario_name": scenario.get("name") if scenario is not None else None,
        "weather_id": weather_id,
        "town_name": route.get("town"),
    }


def checkpoint_state(checkpoint_path, routes_path):
    routes = load_routes(routes_path)
    data = load_json(checkpoint_path)
    checkpoint = data.get("_checkpoint", {})
    progress = checkpoint.get("progress") or [0, len(routes)]
    current = int(progress[0]) if progress else 0
    total = int(progress[1]) if len(progress) > 1 and progress[1] else len(routes)
    status = checkpoint.get("global_record", {}).get("status") or "none"
    current_route = route_info(routes[current])["route_id"] if current < len(routes) else "none"
    return current, total, status, current_route


def make_skipped_record(info, index, reason, event_time):
    return {
        "index": index,
        "route_id": info["route_id"],
        "scenario_name": info["scenario_name"],
        "weather_id": info["weather_id"],
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
        "town_name": info["town_name"],
    }


def skip_current_route(checkpoint_path, routes_path, skipped_log, split, gpu, reason):
    routes = load_routes(routes_path)
    data = load_json(checkpoint_path)
    checkpoint = data.setdefault("_checkpoint", {})
    records = checkpoint.setdefault("records", [])
    progress = checkpoint.setdefault("progress", [len(records), len(routes)])

    if len(progress) < 2:
        progress[:] = [len(records), len(routes)]
    current = int(progress[0])
    progress[1] = len(routes)
    if current >= len(routes):
        raise RuntimeError(f"split {split} is already complete ({current}/{len(routes)})")

    info = route_info(routes[current])
    expected_route_id = info["route_id"]
    target_idx = None
    for index, record in enumerate(records):
        if record.get("route_id") == expected_route_id:
            target_idx = index
            break

    event_time = time.strftime("%F %T")
    if target_idx is None:
        if current > len(records):
            raise RuntimeError(
                f"checkpoint progress {current} is beyond records length {len(records)}"
            )
        records.insert(current, make_skipped_record(info, current, reason, event_time))
        target_idx = current
    else:
        record = records[target_idx]
        record.update(make_skipped_record(info, target_idx, reason, event_time))

    for index, record in enumerate(records):
        record["index"] = index

    progress[0] = max(current + 1, target_idx + 1)
    checkpoint["global_record"] = {}
    data["entry_status"] = "Started"
    data["eligible"] = False
    data.setdefault("sensors", [])
    data.setdefault("values", [])
    data.setdefault("labels", [])

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    tmp_path = checkpoint_path + f".watchdog_tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, checkpoint_path)

    event = {
        "time": event_time,
        "split": int(split),
        "gpu": int(gpu),
        "route_id": expected_route_id,
        "route_index": target_idx,
        "checkpoint": checkpoint_path,
        "routes": routes_path,
        "reason": reason,
    }
    os.makedirs(os.path.dirname(os.path.abspath(skipped_log)), exist_ok=True)
    with open(skipped_log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(expected_route_id, progress[0], progress[1])


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    state_parser = subparsers.add_parser("state")
    state_parser.add_argument("checkpoint")
    state_parser.add_argument("routes")

    skip_parser = subparsers.add_parser("skip")
    skip_parser.add_argument("checkpoint")
    skip_parser.add_argument("routes")
    skip_parser.add_argument("skipped_log")
    skip_parser.add_argument("split")
    skip_parser.add_argument("gpu")
    skip_parser.add_argument("reason")

    args = parser.parse_args()
    if args.command == "state":
        print(*checkpoint_state(args.checkpoint, args.routes))
    else:
        skip_current_route(
            args.checkpoint,
            args.routes,
            args.skipped_log,
            args.split,
            args.gpu,
            args.reason,
        )


if __name__ == "__main__":
    main()
