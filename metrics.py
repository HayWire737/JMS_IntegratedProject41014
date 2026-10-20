"""Error metrics for a logged run."""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def estimation_errors(log):
    e = log["est"] - log["truth"]
    e[:, 2] = wrap(e[:, 2])
    return e


def summary(log, final_goal=None):
    e = estimation_errors(log)
    pos_err = np.hypot(e[:, 0], e[:, 1])
    out = {
        "duration_s": float(log["t"][-1] - log["t"][0]),
        "pos_rmse_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "pos_err_final_m": float(pos_err[-1]),
        "heading_rmse_rad": float(np.sqrt(np.mean(e[:, 2] ** 2))),
        "marker_visible_frac": float(np.mean(log["marker_seen"])),
    }
    if final_goal is not None:
        out["true_dist_to_goal_m"] = float(np.hypot(*(log["truth"][-1, :2] - np.asarray(final_goal))))
    return out


def print_summary(s):
    for k, v in s.items():
        print(f"  {k:<22} {v:.3f}")
