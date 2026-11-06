"""Part E: the full mission (LiDAR navigation, then visual docking) under disturbance and noise.

Every condition is the nominal config plus one change, run over several seeds.
Each trial is scored against ground truth (evaluation/ only):

  success       docked within the tolerance (docking.success_range / success_bearing_deg)
  est RMSE      EKF position error over the run
  max est err   worst EKF position error (how far a disturbance pushed the estimate off)
  track RMS     distance of the true path from the planned route (start -> waypoints)
  clearance     closest approach of the robot centre to an obstacle (body hits below robot_radius)
  NEES          EKF consistency (ideal 3)

    python scripts/robustness_study.py                    # 5 seeds, controller from the config
    python scripts/robustness_study.py --controller lqr --seeds 10
"""
import argparse
import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from _common import load_config, results_dir
from evaluation.metrics import docking_result, estimation_errors
from run import run_trial


def double_noise(cfg):
    n = cfg["noise"]
    n["encoder_tick_sigma"] *= 2
    n["lidar_sigma"] *= 2
    n["camera_pixel_sigma"] *= 2


def disturb(**d):
    return lambda cfg: cfg.setdefault("disturbances", {}).update(d)


CONDITIONS = {
    "Nominal": lambda cfg: None,
    "2× sensor noise": double_noise,
    "Camera blur (1 px)": lambda cfg: cfg["noise"].update(camera_blur_px=1),
    "Marker occluded 22–28 s": disturb(occlusion=[[22.0, 28.0]]),
    "Wheel radius bias ×5": lambda cfg: cfg["noise"].update(wheel_scale_error=[0.01, -0.01]),
    "Wheel slip (left +20 % for 1 s)": disturb(slip={"t": 6.0, "duration": 1.0, "wheel": "left", "extra": 0.2}),
    "External push (8 N, 0.3 s)": disturb(push={"t": 10.0, "duration": 0.3, "force": [0.0, 8.0]}),
    "Varied start pose": "random_start",
}


def random_start(cfg, rng, clearance=0.4):
    """A start pose on the west side of the room, clear of walls and obstacles."""
    obstacles = cfg["obstacles"]
    while True:
        x, y = rng.uniform(0.3, 2.6), rng.uniform(0.3, 3.7)
        free = True
        for ob in obstacles:
            if ob["type"] == "box":
                dx, dy = max(abs(x - ob["x"]) - ob["hx"], 0), max(abs(y - ob["y"]) - ob["hy"], 0)
                free &= np.hypot(dx, dy) > clearance
            else:
                free &= np.hypot(x - ob["x"], y - ob["y"]) - ob["r"] > clearance
        if free:
            return [float(x), float(y), float(rng.uniform(-np.pi, np.pi))]


def obstacle_clearance(p, cfg):
    d = []
    for ob in cfg["obstacles"]:
        if ob["type"] == "box":
            q = np.abs(p - [ob["x"], ob["y"]]) - [ob["hx"], ob["hy"]]
            d.append(np.hypot(*np.maximum(q, 0).T) + np.minimum(q.max(axis=1), 0))
        else:
            d.append(np.hypot(p[:, 0] - ob["x"], p[:, 1] - ob["y"]) - ob["r"])
    return np.min(d, axis=0)


def route_deviation(p, route):
    """Distance from each point to the nearest segment of the planned route."""
    d = np.full(len(p), np.inf)
    for a, b in zip(route[:-1], route[1:]):
        ab = b - a
        t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return d


def trial(cfg, seed):
    log, est, planner = run_trial(cfg, "ekf", seed)
    truth = log["truth"][:, :2]
    on_route = log["phase"] == "route" if "phase" in log else np.ones(len(truth), bool)
    route = np.array([cfg["sim"]["start_pose"][:2]] + cfg["waypoints"], dtype=float)
    err = estimation_errors(log)
    pos_err = np.hypot(err[:, 0], err[:, 1])
    P = log["P"]
    ok = np.all(np.isfinite(P), axis=(1, 2))
    nees = [e[:3] @ np.linalg.solve(Pk, e[:3]) for e, Pk in zip(err[ok], P[ok])]
    dock = docking_result(log, cfg)
    return {
        "success": bool(dock["success"]),
        "est_rmse_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "est_max_m": float(pos_err.max()),
        "track_rms_m": float(np.sqrt(np.mean(route_deviation(truth[on_route], route) ** 2))),
        "clearance_m": float(obstacle_clearance(truth, cfg).min()),
        "nees": float(np.mean(nees)),
        "dock_range_m": float(dock["range_m"]),
        "time_s": float(log["t"][-1]),
        "state": dock["state"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--controller", choices=["p", "lqr"], help="override controller.type")
    ap.add_argument("--only", help="run only conditions whose name contains this text")
    args = ap.parse_args()
    base = load_config(args.config)
    if args.controller:
        base["controller"]["type"] = args.controller
    ctl = base["controller"].get("type", "p")
    radius = base.get("avoidance", {}).get("robot_radius", 0.10)

    results, rows = {}, []
    for name, change in CONDITIONS.items():
        if args.only and args.only.lower() not in name.lower():
            continue
        trials = []
        for seed in range(args.seeds):
            cfg = copy.deepcopy(base)
            cfg["sim"]["seed"] = seed
            if change == "random_start":
                cfg["sim"]["start_pose"] = random_start(cfg, np.random.default_rng(100 + seed))
            else:
                change(cfg)
            trials.append(trial(cfg, seed))
        col = lambda k: np.array([t[k] for t in trials])
        summary = {
            "trials": len(trials),
            "success": int(col("success").sum()),
            "est_rmse_m": [float(col("est_rmse_m").mean()), float(col("est_rmse_m").std())],
            "est_max_m": float(col("est_max_m").max()),
            "track_rms_m": [float(col("track_rms_m").mean()), float(col("track_rms_m").std())],
            "min_clearance_m": float(col("clearance_m").min()),
            "collisions": int((col("clearance_m") < radius).sum()),
            "nees": float(col("nees").mean()),
            "time_s": float(col("time_s").mean()),
            "failures": sorted({t["state"] for t in trials if not t["success"]}),
        }
        results[name] = summary
        rows.append((name, summary))
        print(f"{name:34s} success {summary['success']}/{summary['trials']}  "
              f"est RMSE {100 * summary['est_rmse_m'][0]:5.1f} cm (max {100 * summary['est_max_m']:5.1f})  "
              f"track {100 * summary['track_rms_m'][0]:5.1f} cm  clearance {100 * summary['min_clearance_m']:5.1f} cm  "
              f"NEES {summary['nees']:5.2f}  {summary['time_s']:5.1f} s  {summary['failures'] or ''}")

    out = results_dir()
    tag = f"robustness_{ctl}"
    with open(out / f"{tag}.yaml", "w") as f:
        yaml.safe_dump({"controller": ctl, "seeds": args.seeds, "conditions": results}, f, sort_keys=False)

    # Markdown table for README section 8
    lines = ["| Condition | Trials | Success | Est. RMSE [cm] | Max est. error [cm] | Track RMS [cm] | Min clearance [cm] | NEES |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        lines.append(f"| {name} | {s['trials']} | {s['success']}/{s['trials']} | "
                     f"{100 * s['est_rmse_m'][0]:.1f} ± {100 * s['est_rmse_m'][1]:.1f} | {100 * s['est_max_m']:.1f} | "
                     f"{100 * s['track_rms_m'][0]:.1f} ± {100 * s['track_rms_m'][1]:.1f} | "
                     f"{100 * s['min_clearance_m']:.1f} | {s['nees']:.1f} |")
    (out / f"{tag}.md").write_text("\n".join(lines) + "\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [n for n, _ in rows]
    yy = np.arange(len(names))
    a1.barh(yy, [s["success"] / s["trials"] * 100 for _, s in rows], color="#4c72b0")
    a1.set(yticks=yy, yticklabels=names, xlim=(0, 100), xlabel="docking success (%)", title=f"Success ({ctl.upper()})")
    a1.invert_yaxis()
    a2.barh(yy, [100 * s["est_rmse_m"][0] for _, s in rows], xerr=[100 * s["est_rmse_m"][1] for _, s in rows],
            color="#dd8452", label="RMSE")
    a2.plot([100 * s["est_max_m"] for _, s in rows], yy, "k|", ms=12, mew=2, label="worst")
    a2.set(yticks=yy, yticklabels=[], xlabel="EKF position error (cm)", title="Estimation error")
    a2.invert_yaxis()
    a2.legend(fontsize=8)
    for a in (a1, a2):
        a.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out / f"{tag}.png", dpi=130)
    print(f"\nSaved {out / tag}.yaml, .md and .png")


if __name__ == "__main__":
    main()