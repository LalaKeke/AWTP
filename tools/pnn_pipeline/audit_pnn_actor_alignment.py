#!/usr/bin/env python3
import argparse

import torch


def audit(states, mask, name):
    valid = states[mask.bool()].float()
    if valid.numel() == 0:
        raise RuntimeError(f"{name}: no valid actors")
    speed = valid[:, 3].clamp_min(0.0)
    displacement = torch.linalg.norm(valid[:, 4:6] - valid[:, 0:2], dim=-1)
    implied_speed = displacement / 3.0
    low = speed < 0.75
    bad_low = low & (displacement > 4.0)
    ratio = implied_speed / speed.clamp_min(0.2)
    print(
        f"[{name}] actors={len(valid)} low_speed={int(low.sum())} "
        f"low_disp_p99={torch.quantile(displacement[low], 0.99).item():.3f}m "
        f"implied_speed_p995={torch.quantile(implied_speed, 0.995).item():.3f}m/s "
        f"ratio_p995={torch.quantile(ratio, 0.995).item():.3f} "
        f"bad_low={int(bad_low.sum())}"
    )
    if bad_low.any():
        examples = valid[bad_low][:5]
        raise RuntimeError(
            f"{name}: {int(bad_low.sum())} low-speed actors move over 4m at 3s; "
            f"examples={examples.tolist()}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    data = torch.load(args.input, map_location="cpu")
    version = data.get("__meta__", {}).get("actor_motion_alignment")
    if version != args.expected_version:
        raise RuntimeError(
            f"alignment version mismatch: file={version!r}, expected={args.expected_version!r}"
        )
    audit(data["ped_states"], data["ped_mask"], "ped")
    audit(data["veh_states"], data["veh_mask"], "veh")
    print(f"[actor-alignment-audit] PASS version={version}")


if __name__ == "__main__":
    main()
