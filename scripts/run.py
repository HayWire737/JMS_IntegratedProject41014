"""Closed-loop run: sense -> estimate -> plan -> control -> simulate.

The planner and controller receive ONLY the estimate. Ground truth is read
through evaluation/ just to log and score the run.

    python scripts/run.py                          # EKF (default), saves results/run_ekf.png
    python scripts/run.py --estimator dr           # dead-reckoning baseline for comparison
    python scripts/run.py --model range_bearing    # rank-deficient measurement model
    python scripts/run.py --view                   # 3D viewer (macOS: mjpython)
"""
import argparse
import copy

import numpy as np

from _common import load_config, results_dir
from evaluation.ground_truth import RunLogger, true_pose          # evaluation only
from evaluation.metrics import print_summary, summary
from evaluation.plots import errors_with_bounds, trajectory_and_errors
from src.control.go_to_goal import GoToGoal
from src.estimation.dead_reckoning import DeadReckoning
from src.estimation.ekf import EKF
from src.planning.waypoints import WaypointFollower
from src.sensing.aruco import ArucoDetector
from src.sensing.lidar import min_range_in_sector
from src.sensing.odometry import WheelOdometry
from src.sim.turtlebot_sim import TurtleBotSim


def run_trial(cfg, estimator="ekf", seed=None, view=False):
    """One complete run. Returns (log arrays, estimator, planner)."""
    rob, ctl = cfg["robot"], cfg["controller"]
    sim = TurtleBotSim(cfg, seed=seed)
    if view:
        sim.open_viewer()
    odom = WheelOdometry(rob["wheel_radius"], rob["wheel_base"], rob["encoder_ticks_per_rev"])
    aruco = ArucoDetector(cfg["marker"], cfg["camera"])
    x0 = cfg["sim"]["start_pose"]
    est = EKF(x0, cfg) if estimator == "ekf" else DeadReckoning(x0)
    controller = GoToGoal(ctl, rob)                           # swap for LQRTracker in Part D
    planner = WaypointFollower(cfg["waypoints"], ctl["goal_tol"])
    log = RunLogger()

    m = sim.step(0.0, 0.0)
    odom.update(m.enc_ticks)
    marker = None
    while m.t < cfg["sim"]["max_time"]:
        # --- estimation (sensor data only)
        ds, dth = odom.update(m.enc_ticks)
        if estimator == "ekf":
            est.predict(ds, dth, sim.dt)
        else:
            est.predict(ds, dth)
        seen = False
        if m.image is not None:
            marker = aruco.detect(m.image)
            seen = marker is not None
            est.update(marker)
        x_est = est.x

        # --- planning and control (estimate only)
        goal = planner.current(x_est)
        if goal is None:
            v, w = 0.0, 0.0
        else:
            v, w = controller.command(x_est, goal)
            if min_range_in_sector(m.lidar_ranges, m.lidar_angles, 0.0, np.radians(30)) < ctl["stop_distance"]:
                v = 0.0                                       # LiDAR safety stop

        log.log(m.t, x_est, true_pose(sim), (v, w), seen, getattr(est, "P", None))   # truth -> log only
        if goal is None:
            break
        m = sim.step(v, w)
    sim.close()
    return log.arrays(), est, planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--estimator", choices=["ekf", "dr"], default="ekf")
    ap.add_argument("--model", choices=["pose", "range_bearing"], help="override ekf.measurement_model")
    ap.add_argument("--R-scale", type=float, help="override ekf.R_scale")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--tag", help="name used for the output files")
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    if args.model:
        cfg["ekf"]["measurement_model"] = args.model
    if args.R_scale:
        cfg["ekf"]["R_scale"] = args.R_scale
    tag = args.tag or args.estimator

    log, est, planner = run_trial(cfg, args.estimator, args.seed, args.view)
    status = "reached all waypoints" if planner.done else f"stopped at waypoint {planner.index}"
    print(f"Run '{tag}': {status} after {log['t'][-1]:.1f} s (judged by the estimate)")
    print_summary(summary(log, cfg["waypoints"][-1], est if args.estimator == "ekf" else None))

    names = {"ekf": f"EKF ({cfg['ekf']['measurement_model']})", "dr": "dead reckoning"}
    out = results_dir() / f"run_{tag}.png"
    trajectory_and_errors(log, cfg, out, f"{names[args.estimator]} + P controller")
    print(f"Saved {out}")
    if args.estimator == "ekf":
        out2 = results_dir() / f"run_{tag}_bounds.png"
        errors_with_bounds(log, out2, f"{names['ekf']}: error vs ±3σ")
        print(f"Saved {out2}")
    np.savez(results_dir() / f"run_{tag}.npz", **log)


if __name__ == "__main__":
    main()
