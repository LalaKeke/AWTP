from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation" / "pnn_epoch0019_pid_hybrid_physical_epic_0811"
OUT = ROOT / "outputs" / "pnn_closed_loop_cases_latest_0811_surround_bev"

CASES = {
    "01_route1792_collision_avoidance": ("1792", ["0250", "0300", "0350"]),
    "02_route2143_priority_interaction": ("2143", ["0200", "0250", "0300"]),
    "03_route2144_lane_recovery": ("2144", ["0400", "0450", "0500"]),
}


def find_route(route_id: str) -> Path:
    routes = sorted(EVAL.glob(f"*RouteScenario_{route_id}_*"))
    if not routes:
        raise FileNotFoundError(route_id)
    return routes[-1]


def remove_spatial_trajectory(rgb: np.ndarray) -> np.ndarray:
    # The spatial-planning overlay is cyan in the saved RGB images. Restricting
    # the mask to high-saturation cyan preserves blue detection boxes and the
    # red/magenta HiP-AD/PNN trajectories.
    r, g, b = cv2.split(rgb)
    mask = ((r < 115) & (g > 100) & (b > 125) & ((b.astype(int) - r) > 55)).astype(np.uint8) * 255
    # In BEV only, also remove the yellow/orange selected lane-boundary
    # overlays so the sole planning outputs are HiP-AD and PNN.
    bev = np.zeros(mask.shape, dtype=bool)
    bev[:, 1440:] = True
    lane_overlay = bev & (r > 145) & (g > 90) & (b < 105) & ((r.astype(int) - b) > 80)
    mask[lane_overlay] = 255
    # Remove other-agent motion trails from BEV as well. Detection boxes remain;
    # only the two ego planning trajectories are retained as colored lines.
    actor_trails = bev & (g > 140) & (r < 75) & (b < 85)
    mask[actor_trails] = 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cleaned = cv2.inpaint(bgr, mask, 4, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)


def add_legend(rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    # Replace the original three-item BEV legend with a compact two-item one.
    cv2.rectangle(out, (1450, 8), (1670, 74), (25, 31, 40), -1)
    cv2.line(out, (1466, 29), (1522, 29), (255, 35, 35), 5)
    cv2.putText(out, "HiP-AD", (1532, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(out, (1466, 56), (1522, 56), (255, 0, 255), 5)
    cv2.putText(out, "PNN", (1532, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for case_name, (route_id, frames) in CASES.items():
        route = find_route(route_id)
        case_out = OUT / case_name
        case_out.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            src = route / "images" / f"{frame}.jpg"
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(src)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cleaned = remove_spatial_trajectory(rgb)
            cleaned = add_legend(cleaned)
            cv2.imwrite(str(case_out / f"{frame}.png"), cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
