"""Repeated docking trials: full run (waypoints, then visual docking) over several seeds.

The docking stage decides from the camera whether it has docked. The final
docking error is then measured against ground truth (evaluation/ only), and a
trial counts as a success when the true range and image bearing meet
docking.success_range and docking.success_bearing_deg.

    python scripts/dock_study.py                     # 5 seeds, controller from the config
    python scripts/dock_study.py --controller lqr --seeds 10
"""
import argparse
import copy
import csv

import numpy as np

from _common import load_config, results_dir
from evaluation.metrics import docking_result
from run import run_trial

COLUMNS = ["range_m", "range_err_m", "bearing_deg", "lateral_m", "heading_err_deg", "dock_time_s", "total_time_s"]


def dock_trials(cfg, seeds, estimator="ekf"):
    """Run one full trial per seed. Returns one docking_result row per seed."""
    rows = []
    for seed in seeds:
        c = copy.deepcopy(cfg)
        c["sim"]["seed"] = seed
        log, _, planner = run_trial(c, estimator, seed)
        rows.append({"seed": seed, "route_done": planner.done, **docking_result(log, c)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--controller", choices=["p", "lqr"], help="override controller.type for the route")
    ap.add_argument("--tag", default="dock_study")
    args = ap.parse_args()
    cfg = copy.deepcopy(load_config(args.config))
    cfg["docking"]["enabled"] = True
    if args.controller:
        cfg["controller"]["type"] = args.controller

    rows = dock_trials(cfg, range(args.seeds))
    print(f"{'seed':>4} {'state':>8} {'range m':>8} {'meas m':>7} {'bearing':>8} {'lateral':>8} "
          f"{'heading':>8} {'dock s':>7} {'total s':>8}  result")
    for r in rows:
        print(f"{r['seed']:>4} {r['state']:>8} {r['range_m']:8.3f} {r['meas_range_m']:7.3f} "
              f"{r['bearing_deg']:+7.1f}° {r['lateral_m'] * 100:+6.1f}cm {r['heading_err_deg']:+7.1f}° "
              f"{r['dock_time_s']:7.1f} {r['total_time_s']:8.1f}  {'success' if r['success'] else 'FAIL'}")

    n = len(rows)
    n_ok = sum(r["success"] for r in rows)
    n_self = sum(r["state"] == "docked" for r in rows)
    stats = {k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows])) for k in COLUMNS}
    lo, hi = cfg["docking"]["success_range"]
    print(f"\nSuccess (truth: {lo}-{hi} m, |bearing| <= {cfg['docking']['success_bearing_deg']}°): "
          f"{n_ok}/{n}   self-reported docked: {n_self}/{n}")
    for k, (m, s) in stats.items():
        print(f"  {k:<16} {m:+.4f} ± {s:.4f}")

    out = results_dir()
    with open(out / f"{args.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ctl = cfg["controller"].get("type", "p").upper()
    md = [f"Docking, {n} seeds, {ctl} controller on the route, mean ± std against ground truth.", "",
          "| Metric | Value |", "|---|---|",
          f"| Success rate | {n_ok}/{n} |",
          f"| Self-reported docked | {n_self}/{n} |",
          f"| Final range [m] | {stats['range_m'][0]:.3f} ± {stats['range_m'][1]:.3f} |",
          f"| Range error vs {cfg['docking']['stop_distance']} m [cm] | "
          f"{stats['range_err_m'][0] * 100:+.1f} ± {stats['range_err_m'][1] * 100:.1f} |",
          f"| Image bearing [deg] | {stats['bearing_deg'][0]:+.2f} ± {stats['bearing_deg'][1]:.2f} |",
          f"| Lateral offset [cm] | {stats['lateral_m'][0] * 100:+.1f} ± {stats['lateral_m'][1] * 100:.1f} |",
          f"| Heading error [deg] | {stats['heading_err_deg'][0]:+.2f} ± {stats['heading_err_deg'][1]:.2f} |",
          f"| Docking time [s] | {stats['dock_time_s'][0]:.1f} ± {stats['dock_time_s'][1]:.1f} |"]
    (out / f"{args.tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved {out / (args.tag + '.csv')} and {args.tag}.md")


if __name__ == "__main__":
    main()
