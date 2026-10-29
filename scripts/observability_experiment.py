"""Part A prediction tested in Part C: does a wrong initial pose get corrected?

Theory (scripts/analyse_model.py): with range+bearing to one marker, rotating the
robot about the marker is unobservable, so an initial error in that direction
can never be corrected. The marker pose model observes it.

Experiment: the robot drives a fixed path in front of the marker (marker in
view and in range throughout). Both filters start from the same wrong initial
estimate with a large initial covariance. Final error is decomposed into the
"rotation about the marker" direction and the rest.

    python scripts/observability_experiment.py
"""
import argparse
import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import load_config, results_dir
from evaluation.ground_truth import true_pose          # evaluation only
from src.estimation.ekf import EKF
from src.frames.se2 import wrap
from src.sensing.aruco import ArucoDetector
from src.sensing.odometry import WheelOdometry
from src.sim.turtlebot_sim import TurtleBotSim

START = [2.6, 1.6, 0.2]
COMMANDS = [(4.0, 0.12, 0.10), (4.0, 0.12, -0.15), (4.0, 0.10, 0.05)]   # (s, v, w), marker stays in view


def rotation_about_marker_error(x_est, x_true, L):
    """Project the error onto the unobservable direction (rotation about the marker)."""
    dx, dy = L[0] - x_true[0], L[1] - x_true[1]
    n = np.array([dy, -dx, 1.0])          # tangent of rotating the pose about L by +1 rad
    e = x_est - x_true
    e[2] = wrap(e[2])
    phi = (e @ n) / (n @ n)               # least-squares rotation angle explaining the error
    return phi, np.linalg.norm(e - phi * n)


def run(cfg, model, offset, seed):
    c = copy.deepcopy(cfg)
    c["ekf"]["measurement_model"] = model
    c["ekf"]["initial_sigma"] = [0.4, 0.4, 0.3]
    c["ekf"]["reset_after_rejections"] = 0
    c["sim"]["camera_every"] = 2
    rob = c["robot"]
    sim = TurtleBotSim(c, seed=seed)
    sim.reset(START)
    odom = WheelOdometry(rob["wheel_radius"], rob["wheel_base"], rob["encoder_ticks_per_rev"])
    det = ArucoDetector(c["marker"], c["camera"])
    f = EKF(np.array(START) + offset, c)
    odom.update(sim.step(0, 0).enc_ticks)
    L = np.array(c["marker"]["pose_world"])
    t, phis, res = [], [], []
    for duration, v, w in COMMANDS:
        for _ in range(round(duration / sim.dt)):
            m = sim.step(v, w)
            f.predict(*odom.update(m.enc_ticks))
            if m.image is not None:
                f.update(det.detect(m.image))
            phi, r = rotation_about_marker_error(f.x, true_pose(sim), L)
            t.append(m.t)
            phis.append(phi)
            res.append(r)
    sim.close()
    return np.array(t), np.array(phis), np.array(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    L = np.array(cfg["marker"]["pose_world"])

    # Initial error = pure rotation of 0.15 rad about the marker, plus a small extra offset
    dx, dy = L[0] - START[0], L[1] - START[1]
    offset = 0.15 * np.array([dy, -dx, 1.0]) + np.array([0.05, -0.05, 0.0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    print("Initial error: rotation about marker 0.15 rad (~{:.2f} m of position), plus 7 cm offset".format(
        0.15 * np.hypot(dx, dy)))
    print(f"{'model':<15} {'seed':>4}  final rotation-about-marker [rad]   final other error")
    for model, col in [("range_bearing", "#d62728"), ("pose", "#1f77b4")]:
        for seed in range(3):
            t, phi, res = run(cfg, model, offset, seed)
            print(f"{model:<15} {seed:>4}  {phi[-1]:+.4f}                            {res[-1]:.4f}")
            lab = model if seed == 0 else None
            axes[0].plot(t, phi, color=col, alpha=0.8, label=lab)
            axes[1].plot(t, res, color=col, alpha=0.8, label=lab)
    axes[0].set_title("Error along the unobservable direction\n(rotation of the robot about the marker)")
    axes[0].set_ylabel("rotation error [rad]")
    axes[1].set_title("Error in the other directions")
    axes[1].set_ylabel("residual error norm")
    for ax in axes:
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = results_dir() / "observability_experiment.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
