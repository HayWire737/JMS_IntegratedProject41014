"""Part B: calibrate out the ArUco range bias.

The marker always reads too far away, by ~1 mm at 1 m but ~10 cm at 4 m. It is
not an intrinsics problem (calibrate_camera.py recovers the true K); the detected
outer corners of the marker sit slightly inside the true edge, so the marker
looks smaller and further away. OpenCV's corner refinement methods all leave a
bias (sub-pixel is the best of them).

This script measures the bias against surveyed positions and fits

    bias(d) = a2 d^2 + a3 d^3          d = camera-to-marker distance

then checks the fit on DIFFERENT, randomly placed poses so the correction is not
judged on its own training data. The coefficients go in configs/nominal.yaml
under marker.range_correction, and ArucoDetector applies them.

Ground truth is used offline here, like surveying a calibration target; the
estimator never sees it.

    python scripts/calibrate_marker_range.py
"""
import argparse
import copy

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from _common import load_config, results_dir
from characterise_noise import camera_poses, sightline_blocked
from evaluation.ground_truth import true_pose          # offline calibration reference
from src.frames.se2 import wrap
from src.sensing.aruco import ArucoDetector
from src.sim.turtlebot_sim import TurtleBotSim

BANDS = [(0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.5)]


def collect(cfg, poses, frames):
    """Uncorrected camera-to-marker distance vs truth, for every detection."""
    cfg = copy.deepcopy(cfg)
    cfg["marker"].pop("range_correction", None)          # measure the RAW bias
    sim = TurtleBotSim(cfg)
    det = ArucoDetector(cfg["marker"], cfg["camera"])
    mx, my, _ = cfg["marker"]["pose_world"]
    cam_x = cfg["camera"]["x"]
    rows = []
    for x, y, yaw in poses:
        sim.reset([x, y, yaw])
        p = true_pose(sim)
        # camera and marker centre are at the same height, so the planar distance is the 3D one
        d_true = np.hypot(mx - (p[0] + cam_x * np.cos(p[2])), my - (p[1] + cam_x * np.sin(p[2])))
        for _ in range(frames):
            m = det.detect(sim.render_camera())
            if m is not None:
                rows.append((d_true, np.linalg.norm(m.t_cam)))
    sim.close()
    return np.array(rows)


def random_poses(cfg, n, rng):
    """Random poses inside the detection envelope, for testing the fit."""
    mx, my, myaw = cfg["marker"]["pose_world"]
    poses = []
    while len(poses) < n:
        r, view, bearing = rng.uniform(0.4, 4.2), np.radians(rng.uniform(-45, 45)), np.radians(rng.uniform(-20, 20))
        x, y = mx + r * np.cos(myaw + view), my + r * np.sin(myaw + view)
        if not (0.2 < x < 4.8 and 0.2 < y < 3.8) or sightline_blocked((x, y), (mx, my), cfg):
            continue
        poses.append((x, y, wrap(np.arctan2(my - y, mx - x) - bearing)))
    return poses


def by_band(d, err):
    out = {}
    for lo, hi in BANDS:
        s = (d >= lo) & (d < hi)
        if s.any():
            out[f"{lo:.1f}-{hi:.1f}m"] = {"n": int(s.sum()), "bias_mm": round(1000 * float(err[s].mean()), 1),
                                         "rms_mm": round(1000 * float(np.sqrt(np.mean(err[s] ** 2))), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--frames", type=int, default=4, help="noisy frames per pose")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # 1. Fit on the structured grid of poses (the same grid as characterise_noise.py)
    train = collect(cfg, [(x, y, yaw) for x, y, yaw, _ in camera_poses(cfg)], args.frames)
    d_true, d_meas = train[:, 0], train[:, 1]
    X = np.column_stack([d_meas ** 2, d_meas ** 3])
    (a2, a3), *_ = np.linalg.lstsq(X, d_meas - d_true, rcond=None)

    # 2. Test on different, random poses with different noise
    test_cfg = copy.deepcopy(cfg)
    test_cfg["sim"]["seed"] = cfg["sim"]["seed"] + 1000
    test = collect(test_cfg, random_poses(cfg, 120, np.random.default_rng(1000)), args.frames)
    t_true, t_meas = test[:, 0], test[:, 1]
    t_corr = t_meas - (a2 * t_meas ** 2 + a3 * t_meas ** 3)

    report = {
        "model": "bias(d) = a2 d^2 + a3 d^3, d = camera-to-marker distance (m)",
        "range_correction": [float(a2), float(a3)],
        "train_detections": int(len(train)),
        "test_detections": int(len(test)),
        "test_before": by_band(t_true, t_meas - t_true),
        "test_after": by_band(t_true, t_corr - t_true),
    }
    out = results_dir()
    with open(out / "marker_range_calibration.yaml", "w") as f:
        yaml.safe_dump(report, f, sort_keys=False)

    fig, (a1, a2_) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    dd = np.linspace(0, d_meas.max(), 100)
    a1.plot(d_true, 1000 * (d_meas - d_true), ".", ms=3, alpha=0.5, label="training detections")
    a1.plot(dd, 1000 * (a2 * dd ** 2 + a3 * dd ** 3), "k-", lw=1.5, label="fit a2 d² + a3 d³")
    a1.set(title="Raw range error (training poses)", xlabel="true distance (m)", ylabel="range error (mm)")
    a2_.plot(t_true, 1000 * (t_meas - t_true), ".", ms=3, alpha=0.4, color="#d62728", label="before")
    a2_.plot(t_true, 1000 * (t_corr - t_true), ".", ms=3, alpha=0.6, color="#2ca02c", label="after correction")
    a2_.set(title="Held-out random poses", xlabel="true distance (m)")
    for a in (a1, a2_):
        a.axhline(0, color="0.5", lw=0.8)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "marker_range_calibration.png", dpi=130)

    print(yaml.safe_dump(report, sort_keys=False))
    print("Paste into configs/nominal.yaml under marker:")
    print(f"  range_correction: [{a2:.5f}, {a3:.5f}]   # from scripts/calibrate_marker_range.py")
    print(f"Saved {out / 'marker_range_calibration.yaml'} and {out / 'marker_range_calibration.png'}")


if __name__ == "__main__":
    main()