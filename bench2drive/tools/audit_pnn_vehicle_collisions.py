#!/usr/bin/env python3
"""Audit PNN perception, risk, policy, and actuation before vehicle collisions."""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter
from pathlib import Path


LOCATION_RE = re.compile(
    r"at \(x=(?P<x>-?\d+(?:\.\d+)?), y=(?P<y>-?\d+(?:\.\d+)?), z=(?P<z>-?\d+(?:\.\d+)?)\)"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def find_route_dir(result_dir, save_name):
    candidates = [
        Path(path)
        for path in glob.glob(str(result_dir / ("*" + save_name)))
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "metric_info.json"))
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def nearest_step(metric_info, target_xy):
    best = None
    for step, item in metric_info.items():
        location = item.get("location")
        if not location or len(location) < 2:
            continue
        distance = math.hypot(float(location[0]) - target_xy[0], float(location[1]) - target_xy[1])
        if best is None or distance < best[0]:
            best = (distance, int(step))
    return best


def debug_value(item, key, default=None):
    return item.get("pnn_control_debug", {}).get(key, default)


def first_step(frames, predicate):
    for step, item in frames:
        if predicate(item):
            return step
    return None


def step_lead_seconds(collision_step, step, fps):
    return None if step is None else max(0.0, (collision_step - step) / fps)


def value_at_or_before(frames, collision_step, seconds, fps, key):
    target = collision_step - round(seconds * fps)
    candidates = [(abs(step - target), item) for step, item in frames if step <= collision_step]
    if not candidates:
        return None
    return debug_value(min(candidates, key=lambda pair: pair[0])[1], key)


def classify(actor_seen, risk_step, policy_step, final_brake_step, collision_step, fps):
    if not actor_seen:
        return (
            "A_OR_B_NO_ACTOR_EVIDENCE",
            "旧日志未发现有限的动态障碍距离：可能输入未包含碰撞车辆，也可能车辆未进入风险距离计算范围。",
        )
    if risk_step is None:
        return "B_RISK_NOT_TRIGGERED", "PNN运行时已看到车辆，但动态风险门在碰撞前3秒内没有触发。"
    if policy_step is None:
        return "C_POLICY_DID_NOT_BRAKE", "风险门已触发，但ControlNet融合后的加速度命令没有请求制动。"
    if final_brake_step is None:
        return "D_BRAKE_NOT_EXECUTED", "PNN已请求减速，但最终CARLA brake命令没有落实。"
    lead = step_lead_seconds(collision_step, final_brake_step, fps)
    if lead is not None and lead < 0.5:
        return "E_BRAKE_TOO_LATE", "PNN与执行层均制动，但首次有效制动距离碰撞不足0.5秒。"
    return "F_BRAKE_INSUFFICIENT_OR_GEOMETRY", "碰撞前已较早制动，需检查减速度幅值、预测误差或侧向几何。"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--merged-json", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--policy-brake-accel", type=float, default=-0.1)
    parser.add_argument("--final-brake-threshold", type=float, default=0.08)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    merged_path = Path(args.merged_json).resolve() if args.merged_json else result_dir / "merged.json"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_json(merged_path)["_checkpoint"]["records"]

    event_rows = []
    frame_rows = []
    missing_metrics = 0
    for record in records:
        messages = record.get("infractions", {}).get("collisions_vehicle", [])
        if not messages:
            continue
        route_dir = find_route_dir(result_dir, record.get("save_name", ""))
        metric_path = route_dir / "metric_info.json" if route_dir else None
        if not metric_path or not metric_path.exists():
            missing_metrics += len(messages)
            continue
        metric_info = load_json(metric_path)
        for event_index, message in enumerate(messages):
            match = LOCATION_RE.search(message)
            if not match:
                continue
            target_xy = (float(match.group("x")), float(match.group("y")))
            nearest = nearest_step(metric_info, target_xy)
            if nearest is None:
                continue
            location_error, collision_step = nearest
            window_start = collision_step - round(args.window_seconds * args.fps)
            frames = sorted(
                (int(step), item)
                for step, item in metric_info.items()
                if window_start <= int(step) <= collision_step
            )

            actor_seen = any(
                finite(debug_value(item, "pnn_dynamic_min_distance"))
                or int(debug_value(item, "pnn_num_veh_agents", 0) or 0) > 0
                for _, item in frames
            )
            risk_step = first_step(
                frames,
                lambda item: float(debug_value(item, "pnn_dynamic_brake_floor", 0.0) or 0.0) > 0.05
                or bool(debug_value(item, "pnn_dynamic_long_risk", False)),
            )
            policy_step = first_step(
                frames,
                lambda item: float(debug_value(item, "pnn_accel_command", 0.0) or 0.0)
                < args.policy_brake_accel,
            )
            final_brake_step = first_step(
                frames,
                lambda item: float(debug_value(item, "brake", 0.0) or 0.0)
                >= args.final_brake_threshold,
            )
            category, explanation = classify(
                actor_seen, risk_step, policy_step, final_brake_step, collision_step, args.fps
            )

            finite_distances = [
                float(debug_value(item, "pnn_dynamic_min_distance"))
                for _, item in frames
                if finite(debug_value(item, "pnn_dynamic_min_distance"))
            ]
            row = {
                "route_id": record.get("route_id"),
                "scenario_name": record.get("scenario_name"),
                "status": record.get("status"),
                "event_index": event_index,
                "collision_step": collision_step,
                "collision_x": target_xy[0],
                "collision_y": target_xy[1],
                "location_match_error_m": location_error,
                "actor_seen_by_runtime_guard": actor_seen,
                "min_dynamic_actor_distance_m": min(finite_distances) if finite_distances else None,
                "risk_lead_s": step_lead_seconds(collision_step, risk_step, args.fps),
                "policy_brake_lead_s": step_lead_seconds(collision_step, policy_step, args.fps),
                "final_brake_lead_s": step_lead_seconds(collision_step, final_brake_step, args.fps),
                "accel_command_tminus2s": value_at_or_before(frames, collision_step, 2.0, args.fps, "pnn_accel_command"),
                "accel_command_tminus1s": value_at_or_before(frames, collision_step, 1.0, args.fps, "pnn_accel_command"),
                "accel_command_tminus0_5s": value_at_or_before(frames, collision_step, 0.5, args.fps, "pnn_accel_command"),
                "brake_at_collision": value_at_or_before(frames, collision_step, 0.0, args.fps, "brake"),
                "throttle_at_collision": value_at_or_before(frames, collision_step, 0.0, args.fps, "throttle"),
                "category": category,
                "explanation": explanation,
                "route_dir": str(route_dir),
                "message": message,
            }
            event_rows.append(row)

            event_id = f"{record.get('route_id')}#{event_index}"
            for step, item in frames:
                debug = item.get("pnn_control_debug", {})
                frame_rows.append(
                    {
                        "event_id": event_id,
                        "step": step,
                        "time_to_collision_s": (step - collision_step) / args.fps,
                        "speed_mps": math.hypot(*[float(x) for x in item.get("velocity", [0, 0])[:2]]) if item.get("velocity") else None,
                        "num_veh_agents": debug.get("pnn_num_veh_agents"),
                        "dynamic_min_distance_m": debug.get("pnn_dynamic_min_distance"),
                        "dynamic_risk_time_s": debug.get("pnn_dynamic_risk_time"),
                        "dynamic_brake_floor": debug.get("pnn_dynamic_brake_floor"),
                        "accel_first": debug.get("pnn_accel_first"),
                        "accel_command": debug.get("pnn_accel_command"),
                        "throttle": debug.get("throttle"),
                        "brake": debug.get("brake"),
                        "steer": debug.get("steer"),
                    }
                )

    event_path = out_dir / "vehicle_collision_audit.csv"
    frame_path = out_dir / "vehicle_collision_timeline.csv"
    if event_rows:
        with open(event_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
            writer.writeheader()
            writer.writerows(event_rows)
    if frame_rows:
        with open(frame_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
            writer.writeheader()
            writer.writerows(frame_rows)

    counts = Counter(row["category"] for row in event_rows)
    summary_path = out_dir / "vehicle_collision_audit_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"result_dir: {result_dir}\n")
        handle.write(f"vehicle_collision_events: {len(event_rows)}\n")
        handle.write(f"missing_metric_events: {missing_metrics}\n")
        for name, count in sorted(counts.items()):
            handle.write(f"{name}: {count} ({100.0 * count / max(len(event_rows), 1):.2f}%)\n")
        handle.write("\n说明：旧日志未保存actor数量时，A_OR_B不能仅凭min_distance=inf强行拆分；新日志已补齐该字段。\n")
    print(f"events={len(event_rows)} missing_metrics={missing_metrics}")
    print(f"categories={dict(sorted(counts.items()))}")
    print(f"summary={summary_path}")
    print(f"events_csv={event_path}")
    print(f"timeline_csv={frame_path}")


if __name__ == "__main__":
    main()
