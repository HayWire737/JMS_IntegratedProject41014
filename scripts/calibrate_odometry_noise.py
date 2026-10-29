"""Calibrate the odometry process noise Q (Part C).

Model: each wheel's measured travel per step has variance
    var(d_wheel) = q0 + k |d_wheel|
(q0 = encoder quantisation/jitter floor, k = error growing with distance:
slip, radius error). The EKF builds Q for [ds, dth] from this.

Two stages, both offline using the simulator as a reference:
 1. Per-step fit: compare odometry wheel travel with the true wheel travel
    reconstructed from consecutive true poses, fit q0 and k to the squared residuals.
 2. Consistency scaling: per-step residuals miss errors that are correlated over
    time (a constant radius error drifts every step in the same direction), so
    dead-reckoning runs with prediction only are used to find the factor c that
    makes the mean NEES equal to 3 (the state dimension). Final values are c*q0, c*k.

    python scripts/calibrate_odometry_noise.py
"""
import argparse
import copy

import numpy as np

from _common import load_config
from evaluation.ground_truth import true_pose          # offline calibration reference
from src.estimation.ekf import EKF
from src.frames.se2 import wrap
from src.sensing.odometry import WheelOdometry
from src.sim.turtlebot_sim import TurtleBotSim


def true_increment(x0, x1):
    """Distance along the heading and heading change between two true poses."""
    m = x0[2] + wrap(x1[2] - x0[2]) / 2
    ds = np.cos(m) * (x1[0] - x0[0]) + np.sin(m) * (x1[1] - x0[1])
    return ds, wrap(x1[2] - x0[2])


def random_commands(rng, n_segments, rob):
    cmds = []
    for _ in range(n_segments):
        v = rng.uniform(0.0, rob["v_max"])
        w = rng.uniform(-1.5, 1.5)
        cmds.append((rng.uniform(1.5, 3.0), v, w))
    return cmds


def drive(cfg, seed, cmds, start):
    """Drive a command sequence; return lists of odometry and true increments."""
    rob = cfg["robot"]
    sim = TurtleBotSim(cfg, seed=seed)
    sim.reset(start)
    odom = WheelOdometry(rob["wheel_radius"], rob["wheel_base"], rob["encoder_ticks_per_rev"])
    odom.update(sim.step(0.0, 0.0).enc_ticks)
    x_prev = true_pose(sim)
    odo, tru, poses = [], [], [x_prev]
    for duration, v, w in cmds:
        for _ in range(round(duration / sim.dt)):
            m = sim.step(v, w)
            odo.append(odom.update(m.enc_ticks))
            x = true_pose(sim)
            tru.append(true_increment(x_prev, x))
            poses.append(x)
            x_prev = x
    sim.close()
    return np.array(odo), np.array(tru), np.array(poses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["sim"]["camera_every"] = 10 ** 9          # camera not needed, much faster
    rob = cfg["robot"]
    b = rob["wheel_base"]
    rng = np.random.default_rng(7)
    start = [2.5, 2.0, 0.0]

    # ---------------- stage 1: per-step residual fit
    runs = []
    res, mag = [], []
    for k in range(args.runs):
        cmds = random_commands(rng, 6, rob)
        odo, tru, poses = drive(cfg, 100 + k, cmds, start)
        runs.append((odo, poses))
        for (ds_o, dth_o), (ds_t, dth_t) in zip(odo, tru):
            for sgn in (-1, 1):                      # left, right wheel travel
                d_o = ds_o + sgn * dth_o * b / 2
                d_t = ds_t + sgn * dth_t * b / 2
                res.append(d_o - d_t)
                mag.append(abs(d_t))
    res, mag = np.array(res), np.array(mag)
    edges = np.quantile(mag, np.linspace(0, 1, 11))
    centres, var = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mag >= lo) & (mag <= hi)
        centres.append(mag[sel].mean())
        var.append(np.mean(res[sel] ** 2))
    q0, k_ = np.linalg.lstsq(np.column_stack([np.ones(len(centres)), centres]), var, rcond=None)[0]
    q0 = max(q0, 1e-12)
    k_ = max(k_, 1e-9)
    print(f"Stage 1 (per-step residuals): q0 = {q0:.3e} m^2,  k = {k_:.3e} m")

    # ---------------- stage 2: consistency scaling with prediction-only runs
    def mean_nees(scale):
        nees = []
        for odo, poses in runs:
            c = copy.deepcopy(cfg)
            c["ekf"]["odom_q0"], c["ekf"]["odom_k"] = scale * q0, scale * k_
            c["ekf"]["initial_sigma"] = [1e-6, 1e-6, 1e-6]
            f = EKF(poses[0], c)
            for (ds, dth), x_true in zip(odo, poses[1:]):
                f.predict(ds, dth)
                e = f.x - x_true
                e[2] = wrap(e[2])
                nees.append(e @ np.linalg.solve(f.P, e))
        return np.mean(nees[50:])                     # skip the first 2.5 s where P ~ 0

    lo, hi = 0.1, 1e4                                 # bisection on log scale: NEES falls as scale grows
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if mean_nees(mid) > 3.0:
            lo = mid
        else:
            hi = mid
    c = np.sqrt(lo * hi)
    print(f"Stage 2 (consistency): mean NEES with stage-1 values = {mean_nees(1.0):.1f} (ideal 3)")
    print(f"                       scale factor c = {c:.1f} gives mean NEES = {mean_nees(c):.2f}")
    print("\nPaste into configs/nominal.yaml under ekf:")
    print(f"  odom_q0: {c * q0:.3e}")
    print(f"  odom_k: {c * k_:.3e}")


if __name__ == "__main__":
    main()
