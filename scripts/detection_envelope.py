"""Part B: where can the robot actually see the ArUco marker?

Measures the detection rate of the marker over three geometric variables:
  range       base_link to marker centre
  view angle  angle between the marker's face normal and the line to the robot
              (0 = straight in front of the marker, large = looking at it side-on)
  bearing     where the marker sits in the image (0 = image centre)

Every pose is rendered several times WITH the configured camera noise, so a
pose that only sometimes detects shows up as a rate between 0 and 1.

The numbers feed other parts of the project:
  docking (Part D)     closest reliable range = how near the robot can get
                       to the dock and still see the marker
  EKF (Part C)         furthest reliable range vs ekf.max_marker_range
  planning (Part D)    how far off the image centre the marker may drift

    python scripts/detection_envelope.py
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from _common import load_config, results_dir
from characterise_noise import sightline_blocked
from src.frames.se2 import wrap
from src.sensing.aruco import ArucoDetector
from src.sim.turtlebot_sim import TurtleBotSim

RELIABLE = 0.95          # a pose counts as "reliable" if at least 95% of frames detect


def pose_for(cfg, rng, view, bearing):
    """Robot pose at `rng` from the marker, `view` off its normal, marker at `bearing` in the image."""
    mx, my, myaw = cfg["marker"]["pose_world"]
    x = mx + rng * np.cos(myaw + view)
    y = my + rng * np.sin(myaw + view)
    facing = np.arctan2(my - y, mx - x)                 # heading that centres the marker
    return np.array([x, y, wrap(facing - bearing)])     # turn so the marker sits at `bearing`


def in_room(cfg, pose, margin=0.15):
    x, y = pose[:2]
    if not (margin < x < 5 - margin and margin < y < 4 - margin):
        return False
    mx, my, _ = cfg["marker"]["pose_world"]
    return not sightline_blocked((x, y), (mx, my), cfg)


def _edges(c):
    """Cell edges for pcolormesh from (possibly unevenly spaced) cell centres."""
    c = np.asarray(c, dtype=float)
    mid = (c[1:] + c[:-1]) / 2
    return np.r_[c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])]


def detection_rate(sim, det, pose, frames):
    sim.reset(pose)
    return np.mean([det.detect(sim.render_camera()) is not None for _ in range(frames)])


def reliable_span(values, rates):
    """Smallest and largest value of the contiguous run of reliable detections."""
    ok = np.asarray(rates) >= RELIABLE
    if not ok.any():
        return None, None
    idx = np.flatnonzero(ok)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    best = max(runs, key=len)                            # the main reliable band
    return round(float(values[best[0]]), 2), round(float(values[best[-1]]), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--frames", type=int, default=5, help="noisy frames per pose")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sim = TurtleBotSim(cfg)
    det = ArucoDetector(cfg["marker"], cfg["camera"])
    report = {"config": args.config, "frames_per_pose": args.frames, "reliable_if_rate_at_least": RELIABLE}

    # 1. Range x view angle, marker centred in the image
    # fine steps close in (docking), coarser further out
    ranges = np.round(np.r_[np.arange(0.15, 1.0, 0.05), np.arange(1.0, 4.41, 0.1)], 2)
    views = np.radians(np.arange(0, 71, 10))
    grid = np.full((len(views), len(ranges)), np.nan)    # nan = pose outside the room or blocked
    for i, v in enumerate(views):
        for j, r in enumerate(ranges):
            pose = pose_for(cfg, r, v, 0.0)
            if in_room(cfg, pose):
                grid[i, j] = detection_rate(sim, det, pose, args.frames)

    lo, hi = reliable_span(ranges, grid[0])
    report["head_on"] = {"min_reliable_range_m": lo, "max_reliable_range_m": hi}
    report["max_reliable_view_angle_deg"] = {}
    for r in (0.5, 1.0, 2.0, 3.0):
        j = int(np.argmin(np.abs(ranges - r)))
        col = grid[:, j]
        ok = np.flatnonzero(np.nan_to_num(col) >= RELIABLE)
        tested = round(float(np.degrees(views[np.isfinite(col)].max())), 1)
        best = round(float(np.degrees(views[ok].max())), 1) if len(ok) else None
        report["max_reliable_view_angle_deg"][f"{r:.1f}m"] = {"angle": best, "largest_tested": tested}

    # 2. Bearing: how far off the image centre can the marker be?
    bearings = np.radians(np.arange(0, 41, 1))
    report["max_reliable_bearing_deg"] = {}
    bearing_rates = {}
    for r in (0.3, 0.5, 1.0, 2.0):
        rates = [detection_rate(sim, det, pose_for(cfg, r, 0.0, b), args.frames) for b in bearings]
        bearing_rates[r] = rates
        _, b_hi = reliable_span(np.degrees(bearings), rates)
        report["max_reliable_bearing_deg"][f"{r:.1f}m"] = b_hi
    sim.close()

    half_fov = np.degrees(np.arctan(np.tan(np.radians(cfg["camera"]["fovy_deg"]) / 2)
                                    * cfg["camera"]["width"] / cfg["camera"]["height"]))
    report["camera_horizontal_half_fov_deg"] = round(float(half_fov), 1)

    # ------------------------------------------------------------ outputs
    out = results_dir()
    with open(out / "detection_envelope.yaml", "w") as f:
        yaml.safe_dump(report, f, sort_keys=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    im = a1.pcolormesh(_edges(ranges), _edges(np.degrees(views)), np.ma.masked_invalid(grid),
                       cmap="viridis", vmin=0, vmax=1)
    a1.set(title="Detection rate, marker centred in image (white: outside room / blocked)",
           xlabel="range base_link to marker (m)", ylabel="view angle off marker normal (deg)")
    max_range = cfg.get("ekf", {}).get("max_marker_range")
    if max_range:
        a1.axvline(max_range, color="w", ls="--", lw=1)
        a1.text(max_range + 0.03, 65, "EKF max range", color="w", fontsize=8)
    fig.colorbar(im, ax=a1, label="detection rate")
    for r, rates in bearing_rates.items():
        a2.plot(np.degrees(bearings), rates, label=f"{r:.1f} m")
    a2.axvline(half_fov, color="0.4", ls=":", lw=1)
    a2.text(half_fov - 1, 0.5, "image edge", rotation=90, ha="right", fontsize=8, color="0.4")
    a2.set(title="Detection rate vs marker position in image", xlabel="bearing (deg)",
           ylabel="detection rate", ylim=(-0.05, 1.05))
    a2.legend(title="range", fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "detection_envelope.png", dpi=130)

    print(yaml.safe_dump(report, sort_keys=False))
    print(f"Saved {out / 'detection_envelope.yaml'} and {out / 'detection_envelope.png'}")


if __name__ == "__main__":
    main()