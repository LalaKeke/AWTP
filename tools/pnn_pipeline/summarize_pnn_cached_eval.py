#!/usr/bin/env python3
"""Summarize cached PNN open-loop result files into one CSV."""

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np


def epoch_from_path(path):
    matches = re.findall(r"epoch[_-]?(\d+)", str(path))
    return int(matches[-1]) if matches else -1


def mean_metric(rows, key):
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output_rows = []
    for result_path in sorted(args.results, key=epoch_from_path):
        with open(result_path, "rb") as handle:
            outputs = pickle.load(handle)
        rows = [
            item["metric_results"]
            for item in outputs
            if item["metric_results"].get("fut_valid_flag", False)
        ]
        summary = {
            "epoch": epoch_from_path(result_path),
            "valid_samples": len(rows),
            "result_path": str(Path(result_path).resolve()),
        }
        for horizon in (1, 2, 3):
            keys = {
                f"l2_{horizon}s": f"plan_L2_{horizon}s",
                f"pnn_obj_raw_{horizon}s_pct": f"plan_obj_box_col_{horizon}s",
                f"hipad_obj_raw_{horizon}s_pct": f"hipad_plan_obj_box_col_{horizon}s",
                f"pnn_lane_raw_{horizon}s_pct": f"plan_lane_edge_col_{horizon}s",
                f"hipad_lane_raw_{horizon}s_pct": f"hipad_plan_lane_edge_col_{horizon}s",
                f"pnn_obj_masked_{horizon}s_pct": f"plan_recomputed_masked_obj_box_col_{horizon}s",
                f"hipad_obj_masked_{horizon}s_pct": f"hipad_plan_recomputed_masked_obj_box_col_{horizon}s",
                f"pnn_lane_masked_{horizon}s_pct": f"plan_recomputed_masked_lane_edge_col_{horizon}s",
                f"hipad_lane_masked_{horizon}s_pct": f"hipad_plan_recomputed_masked_lane_edge_col_{horizon}s",
                f"comfort_{horizon}s_pct": f"plan_comfort_score_{horizon}s",
            }
            for output_key, metric_key in keys.items():
                value = mean_metric(rows, metric_key)
                if not np.isfinite(value):
                    raise ValueError(
                        f"non-finite metric {metric_key} in {result_path}; "
                        "the evaluation result is invalid"
                    )
                summary[output_key] = value if output_key.startswith("l2_") else 100.0 * value
            static_keys = {
                f"pnn_static_raw_{horizon}s_pct": f"plan_static_box_col_{horizon}s",
                f"hipad_static_raw_{horizon}s_pct": f"hipad_plan_static_box_col_{horizon}s",
            }
            if all(
                metric_key in row
                for row in rows
                for metric_key in static_keys.values()
            ):
                for output_key, metric_key in static_keys.items():
                    value = mean_metric(rows, metric_key)
                    if not np.isfinite(value):
                        raise ValueError(
                            f"non-finite metric {metric_key} in {result_path}"
                        )
                    summary[output_key] = 100.0 * value
        if all(
            key in row
            for row in rows
            for key in (
                "plan_static_box_col_3s",
                "hipad_plan_static_box_col_3s",
            )
        ):
            pnn_static_any = np.asarray(
                [
                    float(row["plan_static_box_col_3s"]) > 0.0
                    for row in rows
                ],
                dtype=bool,
            )
            hipad_static_any = np.asarray(
                [
                    float(row["hipad_plan_static_box_col_3s"]) > 0.0
                    for row in rows
                ],
                dtype=bool,
            )
            summary["pnn_static_any_3s_count"] = int(
                pnn_static_any.sum()
            )
            summary["hipad_static_any_3s_count"] = int(
                hipad_static_any.sum()
            )
            summary["pnn_only_static_any_3s_count"] = int(
                (pnn_static_any & ~hipad_static_any).sum()
            )
            summary["hipad_only_static_any_3s_count"] = int(
                (~pnn_static_any & hipad_static_any).sum()
            )
        output_rows.append(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[eval-summary] wrote {len(output_rows)} epochs to {out}")


if __name__ == "__main__":
    main()
