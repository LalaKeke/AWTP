#!/usr/bin/env python3
"""Collect, prepare, validate, and merge selected route reruns."""

import argparse
import copy
import datetime
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


SKIPPED_STATUS = "Failed - Watchdog skipped"


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path, data):
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def load_events(path):
    events = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            route_id = event.get("route_id")
            if not route_id:
                raise RuntimeError(f"{path}:{line_no} has no route_id")
            events[route_id] = event
    return list(events.values())


def route_id_from_xml(route):
    return f"RouteScenario_{route.get('id')}_rep0"


def source_routes(base_routes, split_count):
    routes = {}
    for split in range(split_count):
        xml_path = Path(f"{base_routes}_{split}.xml")
        for route in ET.parse(xml_path).getroot().findall("route"):
            route_id = route_id_from_xml(route)
            if route_id in routes:
                raise RuntimeError(f"duplicate route id in source XML: {route_id}")
            routes[route_id] = route
    return routes


def original_record(event):
    checkpoint_path = Path(event["checkpoint"])
    data = load_json(checkpoint_path)
    for record in data["_checkpoint"]["records"]:
        if record.get("route_id") == event["route_id"]:
            return checkpoint_path, record
    raise RuntimeError(f"{event['route_id']} is absent from {checkpoint_path}")


def selected_statuses(args):
    return set(args.source_status or [SKIPPED_STATUS])


def collect(args):
    statuses = selected_statuses(args)
    events = []
    checkpoint_base = Path(args.source_checkpoint_base)
    for split in range(args.split_count):
        checkpoint_path = Path(f"{checkpoint_base}_{split}.json").resolve()
        data = load_json(checkpoint_path)
        for record_index, record in enumerate(data["_checkpoint"]["records"]):
            status = record.get("status")
            if status not in statuses:
                continue
            events.append(
                {
                    "route_id": record["route_id"],
                    "split": split,
                    "route_index": record_index,
                    "checkpoint": str(checkpoint_path),
                    "source_status": status,
                    "reason": f"selected for rerun from status: {status}",
                }
            )

    events.sort(key=lambda item: (item["split"], item["route_index"]))
    if len(events) != args.expected_count:
        counts = {}
        for event in events:
            status = event["source_status"]
            counts[status] = counts.get(status, 0) + 1
        raise RuntimeError(
            f"expected {args.expected_count} selected routes, found {len(events)}; "
            f"status_counts={counts}"
        )

    output_path = Path(args.skipped_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, output_path)

    counts = {}
    for event in events:
        status = event["source_status"]
        counts[status] = counts.get(status, 0) + 1
    print(f"collected={len(events)} status_counts={counts}")
    print(f"selection_log={output_path}")


def prepare(args):
    events = sorted(load_events(args.skipped_log), key=lambda item: (item["split"], item["route_index"]))
    if len(events) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} unique skipped routes, found {len(events)}"
        )

    allowed_statuses = selected_statuses(args)
    routes_by_id = source_routes(args.base_routes, args.split_count)
    route_dir = Path(args.rerun_dir) / "routes"
    checkpoint_dir = Path(args.rerun_dir) / "checkpoints"
    route_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    shard_roots = [ET.Element("routes") for _ in range(args.shards)]
    manifest = []
    for index, event in enumerate(events):
        checkpoint_path, record = original_record(event)
        source_status = event.get("source_status", record.get("status"))
        if record.get("status") != source_status or source_status not in allowed_statuses:
            raise RuntimeError(
                f"{event['route_id']} source status mismatch: "
                f"event={source_status}, current={record.get('status')}, "
                f"allowed={sorted(allowed_statuses)}"
            )
        route = routes_by_id.get(event["route_id"])
        if route is None:
            raise RuntimeError(f"{event['route_id']} is absent from source route XML files")
        shard = index % args.shards
        shard_roots[shard].append(copy.deepcopy(route))
        manifest.append(
            {
                "route_id": event["route_id"],
                "source_split": int(event["split"]),
                "source_checkpoint": str(checkpoint_path),
                "source_record_index": int(record["index"]),
                "source_status": source_status,
                "rerun_shard": shard,
                "skip_reason": event.get("reason"),
            }
        )

    base_name = Path(args.rerun_dir) / "routes" / args.route_name
    for shard, root in enumerate(shard_roots):
        xml_path = Path(f"{base_name}_{shard}.xml")
        ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)

    manifest_path = Path(args.rerun_dir) / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"prepared={len(manifest)} shards={args.shards}")
    print(f"manifest={manifest_path}")
    print(f"base_routes={base_name}")


def load_rerun_records(args):
    records = {}
    progress = []
    checkpoint_base = Path(args.rerun_dir) / "checkpoints" / args.route_name
    for shard in range(args.shards):
        path = Path(f"{checkpoint_base}_{shard}.json")
        if not path.exists():
            raise RuntimeError(f"rerun checkpoint is missing: {path}")
        data = load_json(path)
        checkpoint = data.get("_checkpoint", {})
        shard_progress = checkpoint.get("progress") or [0, 0]
        progress.append((path, int(shard_progress[0]), int(shard_progress[1])))
        for record in checkpoint.get("records", []):
            route_id = record.get("route_id")
            if route_id in records:
                raise RuntimeError(f"duplicate rerun result: {route_id}")
            records[route_id] = record
    return records, progress


def status(args):
    manifest = load_json(Path(args.rerun_dir) / "manifest.json")
    expected = {item["route_id"] for item in manifest}
    try:
        records, progress = load_rerun_records(args)
    except RuntimeError as exc:
        print(f"ready=no reason={exc}")
        return 1

    for path, current, total in progress:
        print(f"{path}: {current}/{total}")
    missing = sorted(expected - records.keys())
    unexpected = sorted(records.keys() - expected)
    still_skipped = sorted(
        route_id for route_id, record in records.items()
        if record.get("status") == SKIPPED_STATUS
    )
    print(
        f"expected={len(expected)} records={len(records)} "
        f"missing={len(missing)} unexpected={len(unexpected)} still_skipped={len(still_skipped)}"
    )
    if missing:
        print("missing:", " ".join(missing))
    if unexpected:
        print("unexpected:", " ".join(unexpected))
    if still_skipped:
        print("still_skipped:", " ".join(still_skipped))
    ready = (
        len(expected) == args.expected_count
        and not missing
        and not unexpected
        and not still_skipped
        and all(current == total for _, current, total in progress)
    )
    print(f"ready={'yes' if ready else 'no'}")
    return 0 if ready else 1


def merge(args):
    if status(args) != 0:
        raise RuntimeError("rerun results are incomplete; refusing to modify original checkpoints")

    manifest = load_json(Path(args.rerun_dir) / "manifest.json")
    rerun_records, _ = load_rerun_records(args)
    by_checkpoint = {}
    for item in manifest:
        by_checkpoint.setdefault(item["source_checkpoint"], []).append(item)

    backup_root = (
        Path(args.result_dir)
        / args.backup_name
        / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    backup_root.mkdir(parents=True, exist_ok=False)

    replaced = 0
    for checkpoint_name, items in sorted(by_checkpoint.items()):
        checkpoint_path = Path(checkpoint_name)
        backup_path = backup_root / checkpoint_path.name
        shutil.copy2(checkpoint_path, backup_path)
        data = load_json(checkpoint_path)
        records = data["_checkpoint"]["records"]
        index_by_route = {record["route_id"]: index for index, record in enumerate(records)}
        for item in items:
            route_id = item["route_id"]
            target_index = index_by_route.get(route_id)
            if target_index is None:
                raise RuntimeError(f"{route_id} is absent from {checkpoint_path}")
            source_status = item.get("source_status", SKIPPED_STATUS)
            if records[target_index].get("status") != source_status:
                raise RuntimeError(
                    f"{route_id} changed before merge: expected={source_status}, "
                    f"current={records[target_index].get('status')}"
                )
            replacement = copy.deepcopy(rerun_records[route_id])
            replacement["index"] = target_index
            records[target_index] = replacement
            replaced += 1
        for index, record in enumerate(records):
            record["index"] = index
        write_json_atomic(checkpoint_path, data)

    for artifact in ("merged.json", "merged_metric_valid.json"):
        artifact_path = Path(args.result_dir) / artifact
        if artifact_path.exists():
            artifact_path.unlink()

    print(f"replaced={replaced}")
    print(f"backup={backup_root}")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--result-dir", required=True)
    common.add_argument("--rerun-dir", required=True)
    common.add_argument("--skipped-log", required=True)
    common.add_argument(
        "--base-routes",
        default="bench2drive/leaderboard/data/splits16/bench2drive220",
    )
    common.add_argument("--route-name", default="watchdog_skipped25")
    common.add_argument("--split-count", type=int, default=16)
    common.add_argument("--shards", type=int, default=4)
    common.add_argument("--expected-count", type=int, default=25)
    common.add_argument(
        "--source-status",
        action="append",
        default=[],
        help="Original status selected for rerun; repeat for multiple statuses.",
    )
    common.add_argument(
        "--source-checkpoint-base",
        help="Checkpoint prefix used by collect, without _<split>.json.",
    )
    common.add_argument("--backup-name", default="watchdog_skipped_backup")
    subparsers.add_parser("collect", parents=[common])
    subparsers.add_parser("prepare", parents=[common])
    subparsers.add_parser("status", parents=[common])
    subparsers.add_parser("merge", parents=[common])
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "collect":
        if not args.source_checkpoint_base:
            raise RuntimeError("--source-checkpoint-base is required by collect")
        collect(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "status":
        raise SystemExit(status(args))
    else:
        merge(args)


if __name__ == "__main__":
    main()
