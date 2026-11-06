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


def route_polyline(cfg):
    """Planned route as an (N, 2) polyline: start position, then the waypoints."""
    return np.array([cfg["sim"]["start_pose"][:2]] + [w[:2] for w in cfg["waypoints"]], dtype=float)


def distance_to_polyline(p, route):
    """Distance from each (N, 2) point to the nearest segment of the polyline."""
    p = np.atleast_2d(p)
    d = np.full(len(p), np.inf)
    for a, b in zip(route[:-1], route[1:]):
        ab = b - a
        L2 = ab @ ab
        t = np.clip((p - a) @ ab / L2, 0.0, 1.0) if L2 > 1e-12 else np.zeros(len(p))   # zero-length: distance to a
        d = np.minimum(d, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return d


def tracking_metrics(log, cfg):
    """Route tracking and control effort over the route phase (docking steps excluded).

    track_*      cross-track distance of the TRUE path to the waypoint polyline
    est_track_*  the same for the estimated path (what the controller sees)
    v_rms, w_rms RMS of the commanded speed and yaw rate
    v_jitter, w_jitter  RMS step-to-step change of the commands
    """
    on = log["phase"] == "route" if "phase" in log else np.ones(len(log["t"]), bool)
    route = route_polyline(cfg)
    d_true = distance_to_polyline(log["truth"][on, :2], route)
    d_est = distance_to_polyline(log["est"][on, :2], route)
    v, w = log["cmd"][on, 0], log["cmd"][on, 1]
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")
    return {"route_time_s": float(log["t"][on][-1] - log["t"][0]),
            "track_rms_m": rms(d_true), "track_max_m": float(d_true.max()),
            "est_track_rms_m": rms(d_est),
            "v_rms": rms(v), "w_rms": rms(w),
            "v_jitter": rms(np.diff(v)), "w_jitter": rms(np.diff(w))}


def docking_error(x_true, cfg):
    """Final docking pose against truth: range and image bearing to the marker, offset from
    the marker normal, heading error from facing the marker, and whether it meets the
    success criteria in docking.success_range / docking.success_bearing_deg."""
    dk = cfg.get("docking", {})
    mx, my, myaw = cfg["marker"]["pose_world"]
    x, y, th = x_true
    dx, dy = mx - x, my - y
    c, s = np.cos(th), np.sin(th)
    xb, yb = c * dx + s * dy, -s * dx + c * dy          # marker centre in base_link
    rng = float(np.hypot(dx, dy))
    bearing = float(np.arctan2(yb, xb - cfg["camera"]["x"]))   # from the optical axis, left positive
    lateral = float(-np.sin(myaw) * -dx + np.cos(myaw) * -dy)  # robot offset from the marker normal
    heading_err = float(wrap(th - (myaw + np.pi)))
    lo, hi = dk.get("success_range", [0.25, 0.35])
    ok = lo <= rng <= hi and abs(np.degrees(bearing)) <= dk.get("success_bearing_deg", 15.0)
    return {"range_m": rng, "range_err_m": rng - dk.get("stop_distance", 0.30),
            "bearing_deg": float(np.degrees(bearing)), "lateral_m": lateral,
            "heading_err_deg": float(np.degrees(heading_err)), "success": bool(ok)}


def docking_result(log, cfg):
    """Docking outcome of a logged run: the stage's own decision and measurements, the
    true final error (docking_error) and the time spent docking."""
    phase = log["phase"]
    out = docking_error(log["truth"][-1], cfg)
    docking = phase != "route"
    meas = log["dock_meas"][-1]
    out.update({"state": str(phase[-1]),
                "meas_range_m": float(meas[0]), "meas_bearing_deg": float(np.degrees(meas[1])),
                "dock_time_s": float(log["t"][-1] - log["t"][docking][0]) if docking.any() else float("nan"),
                "total_time_s": float(log["t"][-1] - log["t"][0])})
    return out


def print_summary(s):
    for k, v in s.items():
        print(f"  {k:<22} {v:.3f}" if isinstance(v, float) else f"  {k:<22} {v}")
