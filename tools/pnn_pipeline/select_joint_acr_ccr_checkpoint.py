#!/usr/bin/env python3
"""Select a checkpoint using separate 1/2/3 s mean ACR and mean CCR."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--parent-epoch", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-l2-regression", type=float, default=0.03)
    parser.add_argument("--min-comfort", type=float, default=84.0)
    parser.add_argument(
        "--include-parent",
        action="store_true",
        help="allow the parent checkpoint to remain selected",
    )
    parser.add_argument(
        "--min-acr-margin",
        type=float,
        default=0.0,
        help="required HiPAD-minus-PNN ACR_mean margin in percentage points",
    )
    parser.add_argument(
        "--min-ccr-margin",
        type=float,
        default=0.0,
        help="required HiPAD-minus-PNN CCR_mean margin in percentage points",
    )
    args = parser.parse_args()

    with open(args.summary, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty summary: {args.summary}")

    parent = next(
        (row for row in rows if int(row["epoch"]) == args.parent_epoch), None
    )
    if parent is None:
        raise ValueError(f"parent epoch {args.parent_epoch} is absent")

    parent_l2 = float(parent["l2_3s"])
    candidates = []
    for row in rows:
        if int(row["epoch"]) == args.parent_epoch and not args.include_parent:
            continue
        pnn_acr_mean = sum(
            float(row[f"pnn_obj_raw_{horizon}s_pct"])
            for horizon in (1, 2, 3)
        ) / 3.0
        hipad_acr_mean = sum(
            float(row[f"hipad_obj_raw_{horizon}s_pct"])
            for horizon in (1, 2, 3)
        ) / 3.0
        pnn_ccr_mean = sum(
            float(row[f"pnn_lane_raw_{horizon}s_pct"])
            for horizon in (1, 2, 3)
        ) / 3.0
        hipad_ccr_mean = sum(
            float(row[f"hipad_lane_raw_{horizon}s_pct"])
            for horizon in (1, 2, 3)
        ) / 3.0
        acr_margin = hipad_acr_mean - pnn_acr_mean
        ccr_margin = hipad_ccr_mean - pnn_ccr_mean
        passes = (
            acr_margin >= args.min_acr_margin
            and ccr_margin >= args.min_ccr_margin
            and float(row["l2_3s"]) <= parent_l2 + args.max_l2_regression
            and float(row["comfort_3s_pct"]) >= args.min_comfort
        )
        # Prefer a true joint pass. If none passes, preserve a nonnegative mean
        # CCR margin and move mean ACR as close to HiPAD as possible.
        rank = (
            int(passes),
            int(ccr_margin >= args.min_ccr_margin),
            int(acr_margin >= args.min_acr_margin),
            acr_margin,
            ccr_margin,
            -float(row["l2_3s"]),
        )
        candidates.append(
            (
                rank,
                row,
                pnn_acr_mean,
                hipad_acr_mean,
                pnn_ccr_mean,
                hipad_ccr_mean,
                acr_margin,
                ccr_margin,
                passes,
            )
        )

    if not candidates:
        raise ValueError("summary contains no trained checkpoint")
    (
        _,
        selected,
        pnn_acr_mean,
        hipad_acr_mean,
        pnn_ccr_mean,
        hipad_ccr_mean,
        acr_margin,
        ccr_margin,
        passes,
    ) = max(
        candidates, key=lambda item: item[0]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(selected) + [
        "joint_mean_pass",
        "pnn_acr_mean_pct",
        "hipad_acr_mean_pct",
        "acr_mean_margin_vs_hipad_pct",
        "pnn_ccr_mean_pct",
        "hipad_ccr_mean_pct",
        "ccr_mean_margin_vs_hipad_pct",
    ]
    selected = dict(selected)
    selected.update(
        {
            "joint_mean_pass": int(passes),
            "pnn_acr_mean_pct": pnn_acr_mean,
            "hipad_acr_mean_pct": hipad_acr_mean,
            "acr_mean_margin_vs_hipad_pct": acr_margin,
            "pnn_ccr_mean_pct": pnn_ccr_mean,
            "hipad_ccr_mean_pct": hipad_ccr_mean,
            "ccr_mean_margin_vs_hipad_pct": ccr_margin,
        }
    )
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(selected)

    status = "PASS" if passes else "CLOSEST_ONLY"
    print(
        f"[joint-select] {status} epoch={int(selected['epoch']):04d} "
        f"ACR_mean={pnn_acr_mean:.4f}%/{hipad_acr_mean:.4f}% "
        f"margin={acr_margin:+.4f}pp "
        f"CCR_mean={pnn_ccr_mean:.4f}%/{hipad_ccr_mean:.4f}% "
        f"margin={ccr_margin:+.4f}pp"
    )
    print(f"[joint-select] wrote: {out}")


if __name__ == "__main__":
    main()
