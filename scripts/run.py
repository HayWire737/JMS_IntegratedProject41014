"""Step 6: closed-loop baseline. Sense -> estimate -> plan -> control -> simulate.

The controller and planner receive ONLY the estimate. Ground truth is read
through evaluation/ just to log and score the run.

    python scripts/run.py                  # headless, saves results/run_nominal.png
    python scripts/run.py --view           # with the 3D viewer (macOS: use mjpython)
"""
import argparse

import numpy as np

from _common import load_config, results_dir
from evaluation.ground_truth import RunLogger, true_pose          # evaluation only
from evaluation.metrics import print_summary, summary
from evaluation.plots import trajectory_and_errors
from src.control.go_to_goal import GoToGoal
from src.estimation.dead_reckoning import DeadReckoning
from src.planning.waypoints import WaypointFollower
from src.sensing.aruco import ArucoDetector
from src.sensing.lidar import min_range_in_sector
from src.sensing.odometry import WheelOdometry
from src.sim.turtlebot_sim import TurtleBotSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--tag", default="nominal", help="name used for the output files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rob, ctl = cfg["robot"], cfg["controller"]

    sim = TurtleBotSim(cfg)
    if args.view:
        sim.open_viewer()
    odom = WheelOdometry(rob["wheel_radius"], rob["wheel_base"], rob["encoder_ticks_per_rev"])
    aruco = ArucoDetector(cfg["marker"], cfg["camera"])
    estimator = DeadReckoning(cfg["sim"]["start_pose"])      # swap for EKF in Part C
    controller = GoToGoal(ctl, rob)                           # swap for LQRTracker in Part D
    planner = WaypointFollower(cfg["waypoints"], ctl["goal_tol"])
    log = RunLogger()

    m = sim.step(0.0, 0.0)
    odom.update(m.enc_ticks)
    v = w = 0.0
    marker = None
    while m.t < cfg["sim"]["max_time"]:
        # --- estimation (sensor data only)
        estimator.predict(*odom.update(m.enc_ticks))
        if m.image is not None:
            marker = aruco.detect(m.image)
            estimator.update(marker)
        x_est = estimator.x

        # --- planning and control (estimate only)
        goal = planner.current(x_est)
        if goal is None:
            v, w = 0.0, 0.0
        else:
            v, w = controller.command(x_est, goal)
            if min_range_in_sector(m.lidar_ranges, m.lidar_angles, 0.0, np.radians(30)) < ctl["stop_distance"]:
                v = 0.0                                       # LiDAR safety stop

        log.log(m.t, x_est, true_pose(sim), (v, w), marker is not None)   # truth -> log only
        if goal is None:
            break
        m = sim.step(v, w)

    data = log.arrays()
    status = "reached all waypoints" if planner.done else f"stopped at waypoint {planner.index}"
    print(f"Run '{args.tag}': {status} after {m.t:.1f} s (judged by the estimate)")
    if marker is not None:
        print(f"  last marker measurement: range {marker.range:.3f} m, "
              f"bearing {np.degrees(marker.bearing):.2f} deg")
    print_summary(summary(data, cfg["waypoints"][-1]))
    out = results_dir() / f"run_{args.tag}.png"
    trajectory_and_errors(data, cfg, out, f"Baseline: dead reckoning + P controller ({args.tag})")
    log.save(results_dir() / f"run_{args.tag}.npz")
    print(f"Saved {out}")
    sim.close()


if __name__ == "__main__":
    main()
