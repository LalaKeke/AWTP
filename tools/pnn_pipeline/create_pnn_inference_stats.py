#!/usr/bin/env python3
import argparse

import torch


def feature_minmax(values, q_low, q_high):
    values = values.float()
    lo = torch.quantile(values, q_low, dim=0)
    hi = torch.quantile(values, q_high, dim=0)
    fallback_lo = values.min(0).values
    fallback_hi = values.max(0).values
    bad = (hi - lo).abs() < 1e-6
    return torch.where(bad, fallback_lo, lo), torch.where(bad, fallback_hi, hi)


def masked_agent_minmax(states, mask, q_low, q_high):
    states = states.reshape(-1, 6)
    if mask is not None:
        valid = mask.reshape(-1).bool()
        if valid.any():
            states = states[valid]
    return feature_minmax(states, q_low, q_high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--q-low", type=float, default=0.005)
    parser.add_argument("--q-high", type=float, default=0.995)
    args = parser.parse_args()

    data = torch.load(args.input, map_location="cpu")
    min_ego, max_ego = feature_minmax(
        data["ego_state"], args.q_low, args.q_high
    )
    min_ped, max_ped = masked_agent_minmax(
        data["ped_states"], data.get("ped_mask"), args.q_low, args.q_high
    )
    min_veh, max_veh = masked_agent_minmax(
        data["veh_states"], data.get("veh_mask"), args.q_low, args.q_high
    )
    lane_xy = data["lane_points"][:, 0:2].reshape(-1, 2)
    min_lane, max_lane = feature_minmax(lane_xy, args.q_low, args.q_high)

    output = {
        "format": "pnn_precomputed_normalization_v1",
        "q_low": args.q_low,
        "q_high": args.q_high,
        "min_ego": min_ego,
        "max_ego": max_ego,
        "min_ped": min_ped,
        "max_ped": max_ped,
        "min_veh": min_veh,
        "max_veh": max_veh,
        "min_lane": min_lane,
        "max_lane": max_lane,
    }
    torch.save(output, args.output)
    print(f"wrote {args.output}")
    for key, value in output.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.tolist()}")


if __name__ == "__main__":
    main()
