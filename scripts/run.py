"""Closed-loop run: sense -> estimate -> plan -> control -> simulate.

The planner and controller receive ONLY the estimate. Ground truth is read
through evaluation/ just to log and score the run.

    python scripts/run.py                          # EKF (default), saves results/run_ekf.png
    python scripts/run.py --estimator dr           # dead-reckoning baseline for comparison
    python scripts/run.py --model range_bearing    # rank-deficient measurement model
    python scripts/run.py --controller lqr         # LQR tracker (default comes from controller.type in the config)
    python scripts/run.py --no-dock                # stop at the last waypoint instead of docking
    python scripts/run.py --view                   # 3D viewer (macOS: mjpython)
"""
import argparse
import copy

import numpy as np

from _common import load_config, results_dir
from evaluation.ground_truth import RunLogger, true_pose          # evaluation only
from evaluation.metrics import docking_result, print_summary, summary
from evaluation.plots import errors_with_bounds, trajectory_and_errors
from src.control.docking import VisualDocking
from src.control.go_to_goal import GoToGoal
from src.control.lqr import LQRTracker
from src.estimation.dead_reckoning import DeadReckoning
from src.estimation.ekf import EKF
from src.planning.waypoints import WaypointFollower
from src.sensing.aruco import ArucoDetector
from src.sensing.lidar import min_range_in_sector
from src.sensing.odometry import WheelOdometry
from src.sim.turtlebot_sim import TurtleBotSim
from src.planning.avoidance import LidarAvoidance


def run_trial(cfg, estimator="ekf", seed=None, view=False, on_step=None):
    """One complete run. Returns (log arrays, estimator, planner).

    on_step, if given, is called every control step with a dict of what happened
    (used by scripts/record_demo.py to draw the video; it does not change the run).
    """
    rob, ctl = cfg["robot"], cfg["controller"]
    sim = TurtleBotSim(cfg, seed=seed)
    if view:
        sim.open_viewer()
    odom = WheelOdometry(rob["wheel_radius"], rob["wheel_base"], rob["encoder_ticks_per_rev"])
    aruco = ArucoDetector(cfg["marker"], cfg["camera"])
    x0 = cfg["sim"]["start_pose"]
    est = EKF(x0, cfg) if estimator == "ekf" else DeadReckoning(x0)
    if ctl.get("type", "p") == "lqr":
        controller = LQRTracker(ctl, rob, dt=sim.dt)          # LQR tracker
    else:
        controller = GoToGoal(ctl, rob)                       # baseline P controller
    planner = WaypointFollower(cfg["waypoints"], ctl["goal_tol"])
    avoid = LidarAvoidance(cfg) if cfg.get("avoidance", {}).get("enabled") else None
    dock_cfg = cfg.get("docking", {})
    dock = VisualDocking(dock_cfg, ctl, rob, dt=sim.dt) if dock_cfg.get("enabled") else None
    log = RunLogger()
    phase, dock_meas = [], []

    m = sim.step(0.0, 0.0)
    odom.update(m.enc_ticks)
    while m.t < cfg["sim"]["max_time"]:
        # --- estimation (sensor data only)
        ds, dth = odom.update(m.enc_ticks)
        if estimator == "ekf":
            est.predict(ds, dth, sim.dt)
        else:
            est.predict(ds, dth)
        marker = None
        if m.image is not None:
            marker = aruco.detect(m.image)
            est.update(marker)
        seen = marker is not None
        x_est = est.x
        if dock is not None:
            dock.observe(ds, dth, marker)                     # marker-relative pose (detections + odometry)

        # --- planning and control (estimate only)
        goal = planner.current(x_est)
        target = None
        docking = goal is None and dock is not None and not dock.finished
        if docking:
            v, w = dock.command()                             # visual docking after the last waypoint
        elif goal is None:
            v, w = 0.0, 0.0
        else:
            target = goal
            if avoid is not None:                             # detour around obstacles seen by the LiDAR
                target = avoid.target(x_est, goal, m.lidar_ranges, m.lidar_angles)
                if avoid.switched and hasattr(controller, "start_from_here"):
                    controller.start_from_here()              # new LQR segment from where the robot is
            v, w = controller.command(x_est, target)
        if min_range_in_sector(m.lidar_ranges, m.lidar_angles, 0.0, np.radians(30)) < ctl["stop_distance"]:
            v = 0.0                                           # LiDAR safety stop
            
        x_true = true_pose(sim)                                # truth -> log and recorder only
        log.log(m.t, x_est, x_true, (v, w), seen, getattr(est, "P", None))
        if on_step is not None:
            on_step({"t": m.t, "meas": m, "x_est": x_est.copy(), "P": getattr(est, "P", None),
                     "x_true": x_true, "goal": goal, "target": target, "marker": marker, "cmd": (v, w),
                     "phase": "route" if goal is not None else (dock.state if dock is not None else "done")})
        if dock is not None:
            phase.append("route" if goal is not None else dock.state)
            dock_meas.append((dock.last_range, dock.last_bearing))
        if goal is None and not docking:
            break
        m = sim.step(v, w)
    sim.close()
    out = log.arrays()
    if dock is not None:
        out["phase"], out["dock_meas"] = np.array(phase), np.array(dock_meas)
    return out, est, planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--estimator", choices=["ekf", "dr"], default="ekf")
    ap.add_argument("--model", choices=["pose", "range_bearing"], help="override ekf.measurement_model")
    ap.add_argument("--R-scale", type=float, help="override ekf.R_scale")
    ap.add_argument("--controller", choices=["p", "lqr"], help="override controller.type")
    ap.add_argument("--start", help="override start pose, 'x,y,yaw'")
    ap.add_argument("--waypoints", help="override waypoints, 'x,y;x,y;...'")
    ap.add_argument("--no-avoid", action="store_true", help="disable LiDAR obstacle avoidance")
    ap.add_argument("--no-dock", action="store_true", help="stop at the last waypoint instead of docking")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--tag", help="name used for the output files")
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    if args.model:
        cfg["ekf"]["measurement_model"] = args.model
    if args.R_scale:
        cfg["ekf"]["R_scale"] = args.R_scale
    if args.controller:
        cfg["controller"]["type"] = args.controller
    if args.start:
        cfg["sim"]["start_pose"] = [float(v) for v in args.start.split(",")]
    if args.waypoints:
        cfg["waypoints"] = [[float(v) for v in p.split(",")] for p in args.waypoints.split(";")]
    if args.no_avoid:
        cfg.setdefault("avoidance", {})["enabled"] = False
    if args.no_dock:
        cfg.setdefault("docking", {})["enabled"] = False
    ctl_type = cfg["controller"].get("type", "p")
    tag = args.tag or (args.estimator if ctl_type == "p" else f"{args.estimator}_{ctl_type}")

    log, est, planner = run_trial(cfg, args.estimator, args.seed, args.view)
    status = "reached all waypoints" if planner.done else f"stopped at waypoint {planner.index}"
    print(f"Run '{tag}': {status} after {log['t'][-1]:.1f} s (judged by the estimate)")
    final_goal = None if "phase" in log else cfg["waypoints"][-1]     # after docking, scored by docking_result
    print_summary(summary(log, final_goal, est if args.estimator == "ekf" else None))
    if "phase" in log:
        d = docking_result(log, cfg)
        print(f"Docking: {d['state']} (decided from the camera), "
              f"measured range {d['meas_range_m']:.3f} m, image bearing {d['meas_bearing_deg']:+.1f} deg")
        print(f"  vs truth: range {d['range_m']:.3f} m, image bearing {d['bearing_deg']:+.1f} deg, "
              f"lateral {d['lateral_m'] * 100:+.1f} cm, heading {d['heading_err_deg']:+.1f} deg "
              f"-> {'success' if d['success'] else 'fail'}")

    names = {"ekf": f"EKF ({cfg['ekf']['measurement_model']})", "dr": "dead reckoning"}
    out = results_dir() / f"run_{tag}.png"
    ctl_name = "LQR" if ctl_type == "lqr" else "P"
    trajectory_and_errors(log, cfg, out, f"{names[args.estimator]} + {ctl_name} controller")
    print(f"Saved {out}")
    if args.estimator == "ekf":
        out2 = results_dir() / f"run_{tag}_bounds.png"
        errors_with_bounds(log, out2, f"{names['ekf']}: error vs ±3σ")
        print(f"Saved {out2}")
    np.savez(results_dir() / f"run_{tag}.npz", **log)


if __name__ == "__main__":
    main()
