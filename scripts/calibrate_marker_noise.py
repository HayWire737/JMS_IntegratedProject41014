"""Calibrate the marker measurement noise R (Part C, with Part B crossover).

Places the robot at a grid of known poses facing the marker, captures repeated
noisy camera frames, and compares each marker measurement with the value
predicted from the known pose. Ground truth is used here as a surveyed
reference for OFFLINE sensor calibration only; it never reaches the filter.

For each measurement component it fits  sigma(r) = a + b r^2  to the RMS error
(bias + scatter, because the filter cannot correct either) and prints the
coefficients to paste into configs/nominal.yaml under ekf.marker_sigma.

    python scripts/calibrate_marker_noise.py
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import load_config, results_dir
from evaluation.ground_truth import true_pose          # offline calibration reference
from src.frames.se2 import wrap
from src.model.diff_drive import range_bearing
from src.sensing.aruco import ArucoDetector
from src.sim.turtlebot_sim import TurtleBotSim


def expected_pose_meas(x, L):
    dx, dy = L[0] - x[0], L[1] - x[1]
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(L[2] - x[2])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--frames", type=int, default=25, help="frames per pose")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sim = TurtleBotSim(cfg, seed=123)
    det = ArucoDetector(cfg["marker"], cfg["camera"])
    L = np.array(cfg["marker"]["pose_world"])

    rows = []   # range, err_x, err_y, err_yaw, err_range, err_bearing, outlier
    for dist in np.arange(0.6, 3.01, 0.3):
        for side in (-0.4, -0.2, 0.0, 0.2, 0.4):             # lateral offset, as a fraction of distance
            for head in (-0.25, 0.0, 0.25):                  # heading offset from facing the marker (rad)
                px, py = L[0] - dist, L[1] + side * dist
                if not (0.3 < py < 3.7):
                    continue
                yaw = np.arctan2(L[1] - py, L[0] - px) + head
                sim.reset([px, py, yaw])
                x = true_pose(sim)
                zp, zrb = expected_pose_meas(x, L), range_bearing(x, L)
                for _ in range(args.frames):
                    m = det.detect(sim.render_camera())
                    if m is None:
                        continue
                    e = m.pose - zp
                    e[2] = wrap(e[2])
                    erb = np.array([m.range, m.bearing]) - zrb
                    erb[1] = wrap(erb[1])
                    rows.append([zrb[0], *e, *erb])
    sim.close()
    data = np.array(rows)
    r = data[:, 0]

    names = ["x", "y", "yaw", "range", "bearing"]
    units = ["m", "m", "rad", "m", "rad"]
    bins = np.arange(0.5, 3.4, 0.3)
    print(f"{len(data)} detections. Fitted sigma(r) = a + b r^2 (robust RMS per range bin):\n")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    coeffs = {}
    for j, (name, unit) in enumerate(zip(names, units)):
        err = data[:, j + 1]
        centres, rms = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (r >= lo) & (r < hi)
            if sel.sum() < 10:
                continue
            e = err[sel]
            # Robust: drop the worst 2 % (pose flips) so outliers are left to the gate
            e = e[np.abs(e) <= np.quantile(np.abs(e), 0.98)]
            centres.append(r[sel].mean())
            rms.append(np.sqrt(np.mean(e ** 2)))
        centres, rms = np.array(centres), np.array(rms)
        A = np.column_stack([np.ones_like(centres), centres ** 2])
        a, b = np.linalg.lstsq(A, rms, rcond=None)[0]
        a = max(a, 0.2 * rms.min())                   # keep a positive floor
        coeffs[name] = [float(a), float(b)]
        print(f"  {name:<8} a = {a:.5f} {unit}   b = {b:.5f} {unit}/m^2")

        ax = axes[j]
        ax.plot(r, np.abs(err), ".", ms=2, alpha=0.3, label="|error|")
        ax.plot(centres, rms, "o", label="RMS per bin")
        rr = np.linspace(0.5, 3.2, 50)
        ax.plot(rr, a + b * rr ** 2, "-", label="fit a + b r²")
        ax.set_ylim(0, max(3 * rms.max(), 1e-3))
        ax.set_xlabel("range [m]")
        ax.set_title(f"{name} [{unit}]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = results_dir() / "marker_noise_calibration.png"
    fig.savefig(out, dpi=120)

    print("\nPaste into configs/nominal.yaml under ekf:")
    print("  marker_sigma:")
    for k, (a, b) in coeffs.items():
        print(f"    {k}: [{a:.5f}, {b:.5f}]")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
