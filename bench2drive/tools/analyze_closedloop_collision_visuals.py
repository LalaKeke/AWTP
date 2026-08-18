#!/usr/bin/env python3
"""Index closed-loop infraction locations and their nearest visualization frames."""

import argparse
import csv
import html
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


LOCATION_RE = re.compile(
    r"at \(x=(?P<x>-?\d+(?:\.\d+)?), y=(?P<y>-?\d+(?:\.\d+)?), z=(?P<z>-?\d+(?:\.\d+)?)\)"
)
ANALYZED_INFRACTIONS = {
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "outside_route_lanes",
    "red_light",
    "route_dev",
    "stop_infraction",
    "vehicle_blocked",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_route_dir(result_dir, save_name):
    if not save_name:
        return None
    matches = [
        path.parent
        for path in result_dir.glob(f"*{save_name}/metric_info.json")
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def nearest_metric_step(metric_info, target_xy):
    best = None
    for step, item in metric_info.items():
        location = item.get("location")
        if not location or len(location) < 2:
            continue
        distance = math.hypot(float(location[0]) - target_xy[0], float(location[1]) - target_xy[1])
        if best is None or distance < best[0]:
            best = (distance, int(step), location)
    return best


def nearest_numbered_file(folder, target_step, suffix):
    if folder is None or not folder.exists():
        return None
    candidates = []
    for path in folder.glob(f"*{suffix}"):
        try:
            candidates.append((abs(int(path.stem) - target_step), path))
        except ValueError:
            continue
    return min(candidates, default=(None, None))[1]


def min_actor_clearance(meta):
    plan = np.asarray(meta.get("pnn_plan", []), dtype=np.float32)
    if plan.ndim != 2 or plan.shape[-1] != 2:
        return None
    distances = []
    for key in ("pnn_veh_agents", "pnn_ped_agents"):
        for actor in meta.get(key, []):
            future = np.asarray(actor.get("future", []), dtype=np.float32)
            if future.ndim != 2 or future.shape[-1] != 2:
                continue
            count = min(len(plan), len(future))
            if count:
                distances.append(float(np.linalg.norm(plan[:count] - future[:count], axis=-1).min()))
    return min(distances) if distances else None


def min_lane_distance(meta):
    plan = np.asarray(meta.get("pnn_plan", []), dtype=np.float32)
    lanes = np.asarray(meta.get("pnn_lane_points", []), dtype=np.float32)
    if (
        plan.ndim != 2
        or plan.shape[-1] != 2
        or lanes.ndim != 3
        or lanes.shape[0] < 2
        or lanes.shape[-1] != 2
    ):
        return None
    distances = np.linalg.norm(
        plan[:, None, None, :] - lanes[None, :2, :, :],
        axis=-1,
    )
    return float(distances.min())


def likely_cause(infraction_type, meta):
    if not meta:
        return "no visualization metadata near the infraction"
    speed = float(meta.get("speed", 0.0))
    brake = float(meta.get("brake", 0.0))
    vehicles = int(meta.get("pnn_num_veh_agents", 0))
    pedestrians = int(meta.get("pnn_num_ped_agents", 0))
    if infraction_type == "collisions_layout":
        return "static layout is not represented by the PNN actor list; inspect selected lane/map vectors"
    if infraction_type == "collisions_vehicle" and vehicles == 0:
        return "no vehicle actor entered the PNN input near collision; likely perception/input omission"
    if infraction_type == "collisions_pedestrian" and pedestrians == 0:
        return "no pedestrian actor entered the PNN input near collision; likely perception/input omission"
    if infraction_type in {"red_light", "stop_infraction"}:
        return "PNN has no explicit traffic-light/stop-sign safety input; it relies on the HiP-AD reference"
    if infraction_type in {"outside_route_lanes", "route_dev"}:
        return "inspect 3s PNN trajectory against yellow/orange selected boundaries and gray map candidates"
    if infraction_type == "vehicle_blocked" or (speed < 0.2 and brake > 0.5):
        return "ego remained nearly stopped; inspect whether conservative braking or a missed escape path caused blockage"
    return "actor was present; inspect prediction error, clearance, and braking timing in the nearest frame"


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def analyze(args):
    result_dir = Path(args.result_dir).resolve()
    merged = load_json(args.merged_json)
    out_dir = Path(args.out_dir).resolve()
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    route_cache = {}
    per_type = defaultdict(int)
    for record in merged["_checkpoint"]["records"]:
        save_name = record.get("save_name")
        route_dir = find_route_dir(result_dir, save_name)
        if route_dir not in route_cache:
            metric_path = route_dir / "metric_info.json" if route_dir else None
            route_cache[route_dir] = load_json(metric_path) if metric_path and metric_path.exists() else {}
        metric_info = route_cache[route_dir]

        for infraction_type, messages in record.get("infractions", {}).items():
            if infraction_type not in ANALYZED_INFRACTIONS:
                continue
            for event_index, message in enumerate(messages):
                if per_type[infraction_type] >= args.topk_per_type:
                    continue
                location_match = LOCATION_RE.search(message)
                target_xy = None
                if location_match:
                    target_xy = (
                        float(location_match.group("x")),
                        float(location_match.group("y")),
                    )
                nearest = nearest_metric_step(metric_info, target_xy) if target_xy else None
                distance, step, ego_location = nearest if nearest else (None, None, None)
                image_path = (
                    nearest_numbered_file(route_dir / "images", step, ".jpg")
                    if route_dir and step is not None
                    else None
                )
                meta_path = (
                    nearest_numbered_file(route_dir / "metas", step, ".json")
                    if route_dir and step is not None
                    else None
                )
                meta = load_json(meta_path) if meta_path else {}

                linked_image = None
                if image_path:
                    linked_image = image_dir / (
                        f"{infraction_type}_{per_type[infraction_type]:03d}_"
                        f"{record['route_id']}_{image_path.name}"
                    )
                    if not linked_image.exists():
                        os.symlink(image_path, linked_image)

                rows.append(
                    {
                        "route_id": record["route_id"],
                        "scenario_name": record.get("scenario_name"),
                        "status": record.get("status"),
                        "infraction_type": infraction_type,
                        "event_index": event_index,
                        "message": message,
                        "collision_x": target_xy[0] if target_xy else None,
                        "collision_y": target_xy[1] if target_xy else None,
                        "nearest_step": step,
                        "location_match_distance_m": distance,
                        "ego_x": ego_location[0] if ego_location else None,
                        "ego_y": ego_location[1] if ego_location else None,
                        "speed": meta.get("speed"),
                        "steer": meta.get("steer"),
                        "throttle": meta.get("throttle"),
                        "brake": meta.get("brake"),
                        "pnn_num_veh_agents": meta.get("pnn_num_veh_agents"),
                        "pnn_num_ped_agents": meta.get("pnn_num_ped_agents"),
                        "min_plan_actor_center_distance_m": min_actor_clearance(meta),
                        "min_plan_selected_lane_point_distance_m": min_lane_distance(meta),
                        "likely_cause": likely_cause(infraction_type, meta),
                        "route_dir": str(route_dir) if route_dir else "",
                        "meta_path": str(meta_path) if meta_path else "",
                        "image_path": str(image_path) if image_path else "",
                        "report_image": str(linked_image.relative_to(out_dir)) if linked_image else "",
                    }
                )
                per_type[infraction_type] += 1

    csv_path = out_dir / "collision_frame_index.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["route_id"])
        writer.writeheader()
        writer.writerows({key: fmt(value) for key, value in row.items()} for row in rows)

    html_path = out_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write("<!doctype html><meta charset='utf-8'><title>PNN closed-loop infractions</title>")
        handle.write("<style>body{font-family:sans-serif;margin:24px}article{border-top:1px solid #bbb;padding:16px 0}img{max-width:1200px;width:100%}code{white-space:pre-wrap}</style>")
        handle.write("<h1>PNN closed-loop infraction frames</h1>")
        handle.write(f"<p>Rows: {len(rows)}. Colors: HiP-AD trajectory red, PNN trajectory magenta, selected left/right boundaries yellow/orange, other map candidates gray.</p>")
        for row in rows:
            handle.write("<article>")
            handle.write(
                f"<h2>{html.escape(str(row['infraction_type']))}: "
                f"{html.escape(str(row['route_id']))}</h2>"
            )
            handle.write(f"<p><b>Scenario:</b> {html.escape(str(row['scenario_name']))}</p>")
            handle.write(f"<p><b>Evidence:</b> {html.escape(str(row['message']))}</p>")
            handle.write(f"<p><b>Initial diagnosis:</b> {html.escape(str(row['likely_cause']))}</p>")
            handle.write(
                f"<p>speed={fmt(row['speed'])}, brake={fmt(row['brake'])}, "
                f"vehicles={fmt(row['pnn_num_veh_agents'])}, pedestrians={fmt(row['pnn_num_ped_agents'])}, "
                f"actor_center_clearance={fmt(row['min_plan_actor_center_distance_m'])} m</p>"
            )
            if row["report_image"]:
                handle.write(
                    f"<a href='{html.escape(row['report_image'])}'>"
                    f"<img src='{html.escape(row['report_image'])}' loading='lazy'></a>"
                )
            else:
                handle.write("<p>No visualization frame available.</p>")
            handle.write("</article>")

    print(f"rows={len(rows)}")
    print(f"counts={dict(sorted(per_type.items()))}")
    print(f"csv={csv_path}")
    print(f"html={html_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--merged-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk-per-type", type=int, default=40)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
