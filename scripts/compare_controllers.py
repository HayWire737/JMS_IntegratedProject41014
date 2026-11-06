"""P vs LQR on the nominal route, several seeds, EKF in the loop.

Docking is switched off so only the route is compared. Each trial is scored
against ground truth (evaluation/ only) with evaluation.metrics.tracking_metrics.

    python scripts/compare_controllers.py              # 5 seeds
    python scripts/compare_controllers.py --seeds 10 --workers 8
"""
import argparse
import copy
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import load_config, results_dir
from evaluation.metrics import tracking_metrics
from run import run_trial

METRICS = ["finished", "route_time_s", "track_rms_m", "track_max_m", "est_track_rms_m",
           "v_rms", "w_rms", "v_jitter", "w_jitter"]


def route_cfg(base, controller, seed):
    """Copy of the config for a route-only trial with the given controller and seed."""
    cfg = copy.deepcopy(base)
    cfg["controller"]["type"] = controller
    cfg.setdefault("docking", {})["enabled"] = False
    cfg["sim"]["seed"] = seed
    return cfg


def route_trial(cfg):
    """One route-only run. Returns the tracking metrics plus whether the route was finished."""
    log, _, planner = run_trial(cfg, "ekf", cfg["sim"]["seed"])
    return {"finished": float(planner.done), **tracking_metrics(log, cfg)}


def run_many(cfgs, workers):
    """Run route_trial on each config, in parallel when workers > 1."""
    if workers <= 1:
        return [route_trial(c) for c in cfgs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(route_trial, cfgs))


def mean_std(rows, key):
    x = np.array([r[key] for r in rows], dtype=float)
    return float(np.mean(x)), float(np.std(x))


def default_workers():
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--tag", default="compare_controllers")
    args = ap.parse_args()
    base = load_config(args.config)

    seeds = list(range(args.seeds))
    jobs = [(c, s) for c in ("p", "lqr") for s in seeds]
    results = run_many([route_cfg(base, c, s) for c, s in jobs], args.workers)
    rows = [{"controller": c, "seed": s, **r} for (c, s), r in zip(jobs, results)]

    out = results_dir()
    with open(out / f"{args.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = [f"P vs LQR, nominal route, EKF in the loop, {args.seeds} seeds, mean ± std. "
          "Cross-track is the true path's distance to the waypoint polyline.", "",
          "| Controller | Finished | Route time [s] | Track RMS [cm] | Track max [cm] | Est. track RMS [cm] "
          "| RMS v [m/s] | RMS ω [rad/s] | ω jitter [rad/s/step] | v jitter [m/s/step] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in ("p", "lqr"):
        sub = [r for r in rows if r["controller"] == c]
        m = {k: mean_std(sub, k) for k in METRICS}
        n_fin = int(sum(r["finished"] for r in sub))
        md.append(f"| {c.upper()} | {n_fin}/{len(sub)} | {m['route_time_s'][0]:.1f} ± {m['route_time_s'][1]:.1f} "
                  f"| {100 * m['track_rms_m'][0]:.2f} ± {100 * m['track_rms_m'][1]:.2f} "
                  f"| {100 * m['track_max_m'][0]:.1f} ± {100 * m['track_max_m'][1]:.1f} "
                  f"| {100 * m['est_track_rms_m'][0]:.2f} ± {100 * m['est_track_rms_m'][1]:.2f} "
                  f"| {m['v_rms'][0]:.3f} ± {m['v_rms'][1]:.3f} | {m['w_rms'][0]:.3f} ± {m['w_rms'][1]:.3f} "
                  f"| {m['w_jitter'][0]:.3f} ± {m['w_jitter'][1]:.3f} | {m['v_jitter'][0]:.4f} ± {m['v_jitter'][1]:.4f} |")
    text = "\n".join(md) + "\n"
    print(text)
    (out / f"{args.tag}.md").write_text(text, encoding="utf-8")
    print(f"Saved {out / (args.tag + '.csv')} and {args.tag}.md")


if __name__ == "__main__":
    main()
