"""Error and consistency metrics for a logged run."""
import numpy as np
from scipy.stats import chi2


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def estimation_errors(log):
    e = log["est"] - log["truth"]
    e[:, 2] = wrap(e[:, 2])
    return e


def nees(log):
    """Normalised estimation error squared per step, e^T P^-1 e (nan without P)."""
    e = estimation_errors(log)
    out = np.full(len(e), np.nan)
    for i, (ei, Pi) in enumerate(zip(e, log["P"])):
        if np.all(np.isfinite(Pi)):
            out[i] = ei @ np.linalg.solve(Pi, ei)
    return out


def nees_bounds(n_steps_avg=1, dof=3, prob=0.95):
    """Two-sided chi-square bounds for the (time-averaged) NEES."""
    lo = chi2.ppf((1 - prob) / 2, dof * n_steps_avg) / n_steps_avg
    hi = chi2.ppf(1 - (1 - prob) / 2, dof * n_steps_avg) / n_steps_avg
    return lo, hi


def summary(log, final_goal=None, ekf=None):
    e = estimation_errors(log)
    pos_err = np.hypot(e[:, 0], e[:, 1])
    out = {
        "duration_s": float(log["t"][-1] - log["t"][0]),
        "pos_rmse_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "pos_err_final_m": float(pos_err[-1]),
        "heading_rmse_rad": float(np.sqrt(np.mean(e[:, 2] ** 2))),
        "marker_visible_frac": float(np.mean(log["marker_seen"])),
    }
    n = nees(log)
    if np.isfinite(n).any():
        out["mean_nees"] = float(np.nanmean(n))
        lo, hi = nees_bounds()
        out["nees_in_95pct_bounds"] = float(np.mean((n >= lo) & (n <= hi)))
    if ekf is not None and ekf.history:
        nis = np.array([h["nis"] for h in ekf.history])
        acc = np.array([h["accepted"] for h in ekf.history])
        out["marker_updates"] = int(acc.sum())
        out["rejected_by_gate"] = int((~acc).sum())
        out["median_nis"] = float(np.median(nis[acc])) if acc.any() else float("nan")
    if final_goal is not None:
        out["true_dist_to_goal_m"] = float(np.hypot(*(log["truth"][-1, :2] - np.asarray(final_goal))))
    return out


def print_summary(s):
    for k, v in s.items():
        print(f"  {k:<22} {v:.3f}" if isinstance(v, float) else f"  {k:<22} {v}")
