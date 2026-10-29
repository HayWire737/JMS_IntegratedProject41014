"""Part C evaluation: repeated trials and the R trade-off study.

Runs the full closed loop over several seeds for:
  - dead reckoning (baseline)
  - EKF, range+bearing model
  - EKF, pose model, with R scaled by 0.1, 0.3, 1, 3, 10
and reports mean +- std of the metrics, plus a trade-off figure
(estimator consistency vs accuracy as R changes).

    python scripts/ekf_study.py --seeds 5
Writes results/ekf_study.csv, results/ekf_study.md (table for the project page),
results/ekf_study_tradeoff.png.
"""
import argparse
import copy
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import load_config, results_dir
from evaluation.metrics import summary
from run import run_trial

METRICS = ["pos_rmse_m", "pos_err_final_m", "heading_rmse_rad", "mean_nees",
           "rejected_by_gate", "true_dist_to_goal_m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    base = load_config(args.config)

    conditions = [("dead reckoning", "dr", {}),
                  ("EKF range+bearing", "ekf", {"measurement_model": "range_bearing"})]
    scales = [0.1, 0.3, 1.0, 3.0, 10.0]
    conditions += [(f"EKF pose, R x{s:g}", "ekf", {"R_scale": s}) for s in scales]

    rows, agg = [], {}
    for name, est, overrides in conditions:
        res = []
        for seed in range(args.seeds):
            cfg = copy.deepcopy(base)
            cfg["ekf"].update(overrides)
            cfg["sim"]["seed"] = seed
            log, f, planner = run_trial(cfg, est, seed)
            s = summary(log, cfg["waypoints"][-1], f if est == "ekf" else None)
            s["success"] = float(planner.done)
            res.append(s)
            rows.append({"condition": name, "seed": seed, **s})
        agg[name] = {k: (np.mean([r.get(k, np.nan) for r in res]), np.std([r.get(k, np.nan) for r in res]))
                     for k in METRICS + ["success"]}
        m = agg[name]
        print(f"{name:<22} pos RMSE {m['pos_rmse_m'][0]:.3f}±{m['pos_rmse_m'][1]:.3f} m   "
              f"final {m['pos_err_final_m'][0]:.3f} m   NEES {m['mean_nees'][0]:.2f}   "
              f"gated {m['rejected_by_gate'][0]:.1f}")

    out = results_dir()
    keys = sorted({k for r in rows for k in r})
    with open(out / "ekf_study.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    md = [f"Mean ± std over {args.seeds} seeds (seed changes all sensor noise).", "",
          "| Condition | Pos. RMSE [m] | Final pos. error [m] | Heading RMSE [rad] | Mean NEES (ideal 3) | Gated / run | Waypoints reached |",
          "|---|---|---|---|---|---|---|"]
    for name, m in agg.items():
        f = lambda k, d=3: f"{m[k][0]:.{d}f} ± {m[k][1]:.{d}f}" if np.isfinite(m[k][0]) else "–"
        md.append(f"| {name} | {f('pos_rmse_m')} | {f('pos_err_final_m')} | {f('heading_rmse_rad')} | "
                  f"{f('mean_nees', 2)} | {f('rejected_by_gate', 1)} | {m['success'][0] * 100:.0f} % |")
    (out / "ekf_study.md").write_text("\n".join(md) + "\n")

    # Trade-off figure: R scale vs accuracy and consistency
    names = [f"EKF pose, R x{s:g}" for s in scales]
    rmse = np.array([agg[n]["pos_rmse_m"] for n in names])
    nees_ = np.array([agg[n]["mean_nees"] for n in names])
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.errorbar(scales, rmse[:, 0], rmse[:, 1], fmt="o-", color="#1f77b4", capsize=3, label="position RMSE")
    ax1.set_xscale("log")
    ax1.set_xlabel("R scale factor (1 = calibrated)")
    ax1.set_ylabel("position RMSE [m]", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.errorbar(scales, nees_[:, 0], nees_[:, 1], fmt="s--", color="#d62728", capsize=3, label="mean NEES")
    ax2.axhline(3, color="#d62728", lw=0.8, ls=":")
    ax2.set_yscale("log")
    ax2.set_ylabel("mean NEES (ideal 3)", color="#d62728")
    ax1.set_title("R trade-off: small R = overconfident, large R = underconfident")
    ax1.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "ekf_study_tradeoff.png", dpi=120)
    print(f"\nSaved {out / 'ekf_study.md'}, ekf_study.csv, ekf_study_tradeoff.png")


if __name__ == "__main__":
    main()
