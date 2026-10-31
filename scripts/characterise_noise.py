"""
Part B: this section characterises the noise of every sensor against the simulator ground truth.

Produces the numbers the EKF is tuned from:
  encoders: per step error in (ds, dth) >> process noise Q     
  lidar: per beam range spread and bias >> obstacle avoidance margins
  camera: marker range/bearing error vs distance  >> measurement noise R

This is an EVALUATION script, not a training script. 
It does not tune the EKF, it just produces the numbers that the EKF is tuned from. 
The estimator itself never sees truth.

the file contains all 3 sensors btw

"""

import argparse
import copy
import matplotlib

matplotlib.use('Agg')  # don't pop up a window
import matplotlib.pyplot as plt
import numpy as np
import yaml

from _common import load_config, results_dir
from evaluation.ground_truth import true_pose          # evaluation only
from src.frames.se2 import wrap
from src.sensing.aruco import ArucoDetector
from src.sensing.odometry import WheelOdometry
from src.sim.turtlebot_sim import TurtleBotSim

"""Here is where the encoders are"""

def true_increment(p0, p1):
    """Distance travelled along the heading, and heading change, between two true poses."""
    th_mid = p0[2] + wrap (p1[2] - p0[2]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return dx * np.cos(th_mid) + dy * np.sin(th_mid), wrap(p1[2] - p0[2])

def encoder_run(cfg, start, v, w, duration):

    sim = TurtleBotSim(cfg)
    sim.reset(start)
    rob = cfg['robot']
    odom = WheelOdometry(rob['wheel_radius'], rob['wheel_base'], rob['encoder_ticks_per_rev'])
    odom.update(sim.step(0.0, 0.0).enc_ticks)  # reset odometry to zero
    prev = true_pose(sim)
    rows = []
    for _ in range(round(duration / sim.dt)):
        ds, dth = odom.update(sim.step(v, w).enc_ticks)
        now = true_pose(sim)
        rows.append((ds, dth, *true_increment(prev, now)))
        prev = now
    sim.close()
    return np.array(rows)

def characterise_encoders(cfg, seeds=10):
    """Straight-line and spin runs, repeated over several seeds.

    The tick noise is a random walk, so one 12 s run gives a noisy estimate of the
    drift (it varied by about +-13% between seeds). Each statistic is therefore
    reported as mean and spread over `seeds` independent runs.
    """
    out, first = {}, {}
    for name, (v, w) in {"straight": (0.2, 0.0), "spin": (0.0, 1.0)}.items():
        per_seed = []
        for s in range(seeds):
            c = copy.deepcopy(cfg)
            c["sim"]["seed"] = cfg["sim"]["seed"] + s
            r = encoder_run(c, [0.5, 0.5, 0.0], v, w, 12.0)
            if s == 0:
                first[name] = r                    # one run kept for the histograms
            e_ds = r[:, 0] - r[:, 2]
            e_dth = wrap(r[:, 1] - r[:, 3])
            dist, turn = np.sum(np.abs(r[:, 2])), np.sum(np.abs(r[:, 3]))
            per_seed.append({
                "ds_err_std_m": e_ds.std(),
                "dth_err_std_rad": e_dth.std(),
                "distance_scale_error": np.sum(r[:, 0]) / np.sum(r[:, 2]) - 1 if dist > 0.1 else np.nan,
                "heading_drift_rad_per_m": np.sum(e_dth) / dist if dist > 0.1 else np.nan,
                "turn_scale_error": np.sum(r[:, 1]) / np.sum(r[:, 3]) - 1 if turn > 0.5 else np.nan,
            })
        stats = {"seeds": seeds, "steps_per_run": len(first[name])}
        for key in per_seed[0]:
            vals = np.array([p[key] for p in per_seed], dtype=float)
            if np.all(np.isnan(vals)):
                continue                           # e.g. distance scale is meaningless in a spin
            stats[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
        out[name] = stats
    return out, first

"""Here is where the lidar is"""

def characterise_lidar(cfg, n_scans=200):
    pose = cfg["sim"]["start_pose"]
    clean_cfg = copy.deepcopy(cfg)
    clean_cfg["noise"]["lidar_sigma"] = 0.0
    clean = TurtleBotSim(clean_cfg)
    clean.reset(pose)
    truth = clean.step(0.0, 0.0).lidar_ranges
    clean.close()

    sim = TurtleBotSim(cfg)
    sim.reset(pose)
    scans = np.array([sim.step(0.0, 0.0).lidar_ranges for _ in range(n_scans)])
    sim.close()

    ok = np.all(np.isfinite(scans), axis=0) & np.isfinite(truth)
    err = scans[:, ok] - truth[ok]
    per_beam_std = err.std(axis=0)
    out = {
        "scans": n_scans,
        "beams_with_returns": int(ok.sum()),
        "range_sigma_m": float(err.std()),
        "range_bias_m": float(err.mean()),
        "per_beam_sigma_min_m": float(per_beam_std.min()),
        "per_beam_sigma_max_m": float(per_beam_std.max()),
    }
    return out, err.ravel()

"""Here is where the camera is"""

def sightline_blocked(p, q, cfg, margin=0.05):
    """True if the straight line p -> q passes through an obstacle from the config."""
    for s in np.linspace(0, 1, 60):
        x, y = p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])
        for ob in cfg.get("obstacles", []):
            if ob["type"] == "box":
                if abs(x - ob["x"]) < ob["hx"] + margin and abs(y - ob["y"]) < ob["hy"] + margin:
                    return True
            elif np.hypot(x - ob["x"], y - ob["y"]) < ob["r"] + margin:
                return True
    return False


def camera_poses(cfg):
    """Grid of robot poses in front of the marker: distance, viewing angle and heading offset."""
    mx, my, myaw = cfg["marker"]["pose_world"]
    poses = []
    for r in np.arange(0.3, 4.41, 0.2):                    # distance from marker
        for view in np.radians([-40, -20, 0, 20, 40]):     # angle off the marker's normal
            x = mx + r * np.cos(myaw + view)
            y = my + r * np.sin(myaw + view)
            if not (0.2 < x < 4.8 and 0.2 < y < 3.8):
                continue
            if sightline_blocked((x, y), (mx, my), cfg):
                continue
            facing = np.arctan2(my - y, mx - x)
            for offset in np.radians([-20, 0, 20]):         # marker left/centre/right of image
                poses.append((x, y, wrap(facing + offset), view))
    return poses


def characterise_camera(cfg, repeats=3):
    sim = TurtleBotSim(cfg)
    det = ArucoDetector(cfg["marker"], cfg["camera"])
    mx, my, _ = cfg["marker"]["pose_world"]
    rows = []      # true_range, true_bearing, view_angle, meas_range, meas_bearing (nan if not seen)
    for x, y, yaw, view in camera_poses(cfg):
        sim.reset([x, y, yaw])
        p = true_pose(sim)
        r_true = np.hypot(mx - p[0], my - p[1])
        b_true = wrap(np.arctan2(my - p[1], mx - p[0]) - p[2])
        for _ in range(repeats):                            # repeats expose any per-frame noise
            m = det.detect(sim.render_camera())
            rows.append((r_true, b_true, view, m.range if m else np.nan, m.bearing if m else np.nan))
    sim.close()

    d = np.array(rows)
    seen = np.isfinite(d[:, 3])
    r_true = d[seen, 0]
    e_r = d[seen, 3] - r_true
    e_b = wrap(d[seen, 4] - d[seen, 1])

    # Error grows with distance (the marker covers fewer pixels), so one sigma
    # for all ranges would be wrong. Report it per 0.5 m band instead.
    bands = {}
    for lo in np.arange(0.0, 4.5, 0.5):
        sel = (r_true >= lo) & (r_true < lo + 0.5)
        if sel.sum() < 3:
            continue
        bands[f"{lo:.1f}-{lo + 0.5:.1f}m"] = {
            "n": int(sel.sum()),
            "range_err_mean_m": float(e_r[sel].mean()),
            "range_err_rms_m": float(np.sqrt(np.mean(e_r[sel] ** 2))),
            "bearing_err_rms_rad": float(np.sqrt(np.mean(e_b[sel] ** 2))),
        }
    # Pinhole geometry predicts range error ~ k * range^2; fit k by least squares.
    k = float(np.sum(np.abs(e_r) * r_true ** 2) / np.sum(r_true ** 4))

    out = {
        "frames": int(len(d)),
        "detected": int(seen.sum()),
        "missed_at_ranges_m": sorted({round(float(v), 2) for v in d[~seen, 0]}),
        "max_detected_range_m": float(r_true.max()),
        "range_err_vs_range_squared_k": k,
        "bearing_err_rms_rad": float(np.sqrt(np.mean(e_b ** 2))),
        "by_range": bands,
    }
    return out, d

# report
def plot(enc_runs, lidar_err, cam, out_path, k=None):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    for name, r in (enc_runs or {}).items():
        ax[0, 0].hist(r[:, 0] - r[:, 2], bins=40, alpha=0.6, label=f"{name}: ds")
        ax[0, 1].hist(wrap(r[:, 1] - r[:, 3]), bins=40, alpha=0.6, label=f"{name}: dth")
    ax[0, 0].set(title="Encoder: per-step distance error", xlabel="ds_odom - ds_true (m)")
    ax[0, 1].set(title="Encoder: per-step heading error", xlabel="dth_odom - dth_true (rad)")

    if lidar_err is not None:
        ax[1, 0].hist(lidar_err, bins=50, color="#2ca02c")
    ax[1, 0].set(title="LiDAR: range error (stationary)", xlabel="measured - true (m)")

    if cam is not None:
        seen = np.isfinite(cam[:, 3])
        ax[1, 1].plot(cam[seen, 0], cam[seen, 3] - cam[seen, 0], ".", label="range error (m)")
        ax[1, 1].plot(cam[~seen, 0], np.zeros((~seen).sum()), "r|", ms=15, label="not detected")
        if k is not None:
            rr = np.linspace(0, cam[:, 0].max(), 100)
            ax[1, 1].plot(rr, k * rr ** 2, "k--", lw=1, label=f"fit  {k:.3f} r²")
        twin = ax[1, 1].twinx()
        twin.plot(cam[seen, 0], np.degrees(wrap(cam[seen, 4] - cam[seen, 1])), "x", color="#ff7f0e", ms=4)
        twin.set_ylabel("bearing error (deg, orange x)")
        twin.set_ylim(-1, 1)
    ax[1, 1].axhline(0, color="0.5", lw=0.8)
    ax[1, 1].set(title="Camera/ArUco: error vs distance", xlabel="true range (m)", ylabel="range error (m)")

    for a in ax.ravel():
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def print_block(title, d, indent="  "):
    print(title)
    for k, v in d.items():
        if isinstance(v, dict):
            print_block(f"{indent}{k}:", v, indent + "  ")
        elif isinstance(v, float):
            print(f"{indent}{k:<28} {v: .3e}")
        else:
            print(f"{indent}{k:<28} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--seeds", type=int, default=10, help="repeat the encoder runs over this many seeds")
    ap.add_argument("--part", choices=["all", "encoders", "lidar", "camera"], default="all")
    args = ap.parse_args()
    cfg = load_config(args.config)

    report, enc_runs, lidar_err, cam = {"config": args.config}, None, None, None
    if args.part in ("all", "encoders"):
        report["encoders"], enc_runs = characterise_encoders(cfg, args.seeds)
        print_block("ENCODERS (per control step, dt = %.3f s)" % cfg["sim"]["control_dt"], report["encoders"])
    if args.part in ("all", "lidar"):
        report["lidar"], lidar_err = characterise_lidar(cfg)
        print_block("LIDAR", report["lidar"])
    if args.part in ("all", "camera"):
        report["camera"], cam = characterise_camera(cfg)
        print_block("CAMERA / ARUCO", report["camera"])

    out = results_dir()
    with open(out / "noise_characterisation.yaml", "w") as f:
        yaml.safe_dump(report, f, sort_keys=False)
    plot(enc_runs, lidar_err, cam, out / "noise_characterisation.png",
         k=report.get("camera", {}).get("range_err_vs_range_squared_k"))
    print(f"\nSaved {out / 'noise_characterisation.yaml'} and {out / 'noise_characterisation.png'}")


if __name__ == "__main__":
    main()