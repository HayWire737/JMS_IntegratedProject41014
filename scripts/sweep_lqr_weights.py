"""LQR weight sweep: tracking error against control effort on the nominal route.

One-at-a-time scaling around the config values. Each diagonal entry of Q_lqr
and R_lqr is scaled by each factor in FACTORS with the others held fixed. All of
Q is also scaled together. Only the ratio of Q to R changes the gain, so
"all of Q x f" is the same controller as "all of R x 1/f". v_ref is swept
separately. Docking is off and the EKF is in the loop. Every point is scored
against ground truth (evaluation/ only).

A second set of runs (--investigate seeds) repeats the baseline and the two
heavier-weight settings from earlier ad hoc runs (all Q x3, all R x0.25), plus
all Q x4 as a check that it matches R x0.25.

    python scripts/sweep_lqr_weights.py                  # sweep (5 seeds) + investigation (20 seeds)
    python scripts/sweep_lqr_weights.py --plot-only      # redraw the figure from the saved CSV
    python scripts/sweep_lqr_weights.py --diagnose 10    # where the extra error of all Q x4 comes from

--diagnose runs the baseline and all Q x4 with the EKF and with dead reckoning
in the loop and reports (a) whether the EKF differs from dead reckoning on the
first segment, and (b) the odometry heading error against distance travelled,
split into the start, the straight-line drift and the change through each corner.
"""
import argparse
import copy
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import ROOT, load_config, results_dir
from compare_controllers import default_workers, mean_std, route_cfg
from evaluation.metrics import distance_to_polyline, estimation_errors, route_polyline, tracking_metrics, wrap
from run import run_trial

FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]
V_REFS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22]
CORNER_RADIUS = 0.30       # points this close to an interior waypoint count as "corner"
FAMILIES = [("Q e_x", "Q_lqr", 0), ("Q e_y", "Q_lqr", 1), ("Q e_θ", "Q_lqr", 2),
            ("R v", "R_lqr", 0), ("R ω", "R_lqr", 1), ("all Q (= R x 1/f)", "Q_lqr", None)]
INVESTIGATE = [("baseline", "Q_lqr", None, 1.0), ("all Q x3", "Q_lqr", None, 3.0),
               ("all R x0.25", "R_lqr", None, 0.25), ("all Q x4", "Q_lqr", None, 4.0)]

INK, INK_2, GRID, SURFACE, SERIES = "#1f1e1c", "#5f5e58", "#e4e3de", "#fcfcfb", "#2a78d6"


def scaled(lqr, key, idx, f):
    """Copy of the LQR settings with one weight (or all of key when idx is None) scaled by f."""
    out = copy.deepcopy(lqr)
    w = list(out[key])
    out[key] = [x * f for x in w] if idx is None else [x * f if i == idx else x for i, x in enumerate(w)]
    return out


def diag_trial(cfg):
    """Route-only run: tracking metrics plus diagnostics on where the error comes from."""
    log, _, planner = run_trial(cfg, "ekf", cfg["sim"]["seed"])
    m = {"finished": float(planner.done), **tracking_metrics(log, cfg)}
    route = route_polyline(cfg)
    p = log["truth"][:, :2]
    d = distance_to_polyline(p, route)
    corners = route[1:-1]
    near = (np.min([np.hypot(*(p - w).T) for w in corners], axis=0) < CORNER_RADIUS if len(corners)
            else np.zeros(len(p), bool))
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")
    e = estimation_errors(log)
    v, w = log["cmd"][:, 0], log["cmd"][:, 1]
    m.update({"straight_track_rms_m": rms(d[~near]), "corner_track_rms_m": rms(d[near]),
              "est_err_rms_m": rms(np.hypot(e[:, 0], e[:, 1])),
              "turn_in_place_frac": float(np.mean((v == 0.0) & (np.abs(w) > 1e-6))),
              "w_sat_frac": float(np.mean(np.abs(w) >= 0.99 * cfg["robot"]["w_max"]))})
    return m


def heading_profile(cfg):
    """Run once and return odometry/EKF error figures along the route (for --diagnose)."""
    est = cfg["_estimator"]
    log, _, _ = run_trial(cfg, est, cfg["sim"]["seed"])
    e, tr, t = estimation_errors(log), log["truth"], log["t"]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(tr[:, :2], axis=0).T))]
    rot = np.r_[0.0, np.cumsum(np.abs(wrap(np.diff(tr[:, 2]))))]
    route = route_polyline(cfg)
    last = len(t) - 1
    out = {"t12_pos_err_m": float(np.hypot(*e[min(np.searchsorted(t, 12.0), last), :2])),
           "rotation_total_deg": float(np.degrees(rot[-1]))}
    for d in DIAG_DISTANCES:
        k = min(np.searchsorted(s, d), len(s) - 1)
        out[f"head_err_deg@{d:g}m"] = float(np.degrees(e[k, 2]))
        out[f"pos_err_m@{d:g}m"] = float(np.hypot(*e[k, :2]))
    for i, wp in enumerate(route[1:-1], 1):                  # heading error gained while inside each corner
        near = np.flatnonzero(np.hypot(*(tr[:, :2] - wp).T) < CORNER_RADIUS)
        out[f"corner{i}_head_change_deg"] = float(np.degrees(e[near[-1], 2] - e[near[0], 2])) if len(near) else np.nan
        out[f"corner{i}_rotation_deg"] = float(np.degrees(rot[near[-1]] - rot[near[0]])) if len(near) else np.nan
    k0, k1 = (min(np.searchsorted(s, d), last) for d in (0.3, 1.8))   # straight part of the first segment
    out["straight_drift_deg_per_m"] = (float(np.degrees(e[k1, 2] - e[k0, 2]) / (s[k1] - s[k0]))
                                       if s[k1] > s[k0] else float("nan"))
    return out


DIAG_DISTANCES = [0.3, 1.0, 2.0, 3.0, 4.0]


def diagnose(base, lqr, seeds, workers):
    """Baseline vs all Q x4, each with the EKF and with dead reckoning in the loop."""
    from concurrent.futures import ProcessPoolExecutor
    jobs, cfgs = [], []
    for name, f in (("baseline", 1.0), ("all Q x4", 4.0)):
        for est in ("ekf", "dr"):
            for sd in seeds:
                cfg = route_cfg(base, "lqr", sd)
                cfg["controller"]["lqr"] = scaled(lqr, "Q_lqr", None, f)
                cfg["_estimator"] = est
                jobs.append((name, est))
                cfgs.append(cfg)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(heading_profile, cfgs))
    rows = [{"setting": n, "estimator": e, **r} for (n, e), r in zip(jobs, res)]
    pm = lambda rs, k, sc=1.0, fmt=".2f": (f"{sc * np.mean([r[k] for r in rs]):{fmt}} ± "
                                           f"{sc * np.std([r[k] for r in rs]):{fmt}}")
    n = len(seeds)
    lines = [f"Diagnosis, {n} seeds, mean ± std. Errors are estimate minus truth.", "",
             "Position error at t = 12 s (end of the first segment), EKF vs dead reckoning in the loop:", "",
             "| Setting | EKF [cm] | Dead reckoning [cm] |", "|---|---|---|"]
    for name in ("baseline", "all Q x4"):
        ek = [r for r in rows if r["setting"] == name and r["estimator"] == "ekf"]
        dr = [r for r in rows if r["setting"] == name and r["estimator"] == "dr"]
        lines.append(f"| {name} | {pm(ek, 't12_pos_err_m', 100)} | {pm(dr, 't12_pos_err_m', 100)} |")
    lines += ["", "Dead reckoning in the loop: odometry heading error along the route:", "",
              "| Quantity | baseline | all Q x4 |", "|---|---|---|"]
    dr = {name: [r for r in rows if r["setting"] == name and r["estimator"] == "dr"] for name in ("baseline", "all Q x4")}
    keys = ([(f"heading error at {d:g} m [deg]", f"head_err_deg@{d:g}m", 1.0) for d in DIAG_DISTANCES] +
            [("straight-line drift, 0.3-1.8 m [deg/m]", "straight_drift_deg_per_m", 1.0),
             ("heading error change through corner 1 [deg]", "corner1_head_change_deg", 1.0),
             ("rotation through corner 1 [deg]", "corner1_rotation_deg", 1.0),
             ("heading error change through corner 2 [deg]", "corner2_head_change_deg", 1.0),
             ("rotation through corner 2 [deg]", "corner2_rotation_deg", 1.0),
             ("total rotation over the route [deg]", "rotation_total_deg", 1.0)] +
            [(f"position error at {d:g} m [cm]", f"pos_err_m@{d:g}m", 100.0) for d in DIAG_DISTANCES])
    for label, k, sc in keys:
        lines.append(f"| {label} | {pm(dr['baseline'], k, sc)} | {pm(dr['all Q x4'], k, sc)} |")
    return rows, lines


def run_settings(base, settings, seeds, workers):
    """settings: list of (family, label, factor, lqr dict). Returns one row per (setting, seed)."""
    from concurrent.futures import ProcessPoolExecutor
    jobs, cfgs = [], []
    for fam, label, f, lqr in settings:
        for s in seeds:
            cfg = route_cfg(base, "lqr", s)
            cfg["controller"]["lqr"] = lqr
            jobs.append((fam, label, f, s))
            cfgs.append(cfg)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(diag_trial, cfgs))
    return [{"family": fam, "label": label, "factor": f, "seed": s, **r} for (fam, label, f, s), r in zip(jobs, res)]


def sweep_settings(lqr):
    out = [("baseline", "baseline", 1.0, copy.deepcopy(lqr))]
    for fam, key, idx in FAMILIES:
        out += [(fam, f"{fam} x{f:g}", f, scaled(lqr, key, idx, f)) for f in FACTORS if f != 1.0]
    out += [("v_ref", f"v_ref {v:g}", v, {**copy.deepcopy(lqr), "v_ref": v}) for v in V_REFS if v != lqr["v_ref"]]
    return out


def aggregate(rows, keys):
    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["label"], r["factor"]), []).append(r)
    return {g: {k: mean_std(rs, k) for k in keys} | {"n": len(rs)} for g, rs in groups.items()}


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in ("family", "label"):
                r[k] = float(v)
    return rows


def plot(rows, lqr, path):
    agg = aggregate(rows, ["track_rms_m", "w_jitter"])
    base = agg[("baseline", "baseline", 1.0)]
    fams = [f for f, _, _ in FAMILIES] + ["v_ref"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, fam in zip(axes.flat, fams):
        pts = sorted([(g[2], a) for g, a in agg.items() if g[0] == fam] +
                     [(lqr["v_ref"] if fam == "v_ref" else 1.0, base)], key=lambda t: t[0])
        x = [a["w_jitter"][0] for _, a in pts]
        y = [100 * a["track_rms_m"][0] for _, a in pts]
        ax.errorbar(x, y, xerr=[a["w_jitter"][1] for _, a in pts], yerr=[100 * a["track_rms_m"][1] for _, a in pts],
                    color=SERIES, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, elinewidth=1, capsize=0, zorder=2)
        ax.plot(base["w_jitter"][0], 100 * base["track_rms_m"][0], marker="o", ms=13, mfc="none", mec=INK, mew=2,
                zorder=3)
        for (f, a), xi, yi in zip(pts, x, y):
            lab = f"{f:g}" if fam == "v_ref" else f"x{f:g}"
            ax.annotate(lab, (xi, yi), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK_2)
        ax.set_title(fam, fontsize=10, color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0.0, 0.9, "Ring: current setting\n"
            f"Q_lqr = {lqr['Q_lqr']}\nR_lqr = {lqr['R_lqr']}\nv_ref = {lqr['v_ref']} m/s\n\n"
            "Mean ± std over seeds.\nLabels: scale factor\n(v_ref panel: m/s).", transform=ax.transAxes,
            va="top", fontsize=9, color=INK_2)
    for a in axes[1, :3]:
        a.set_xlabel("yaw-command jitter, RMS Δω per step [rad/s]", fontsize=9, color=INK_2)
    for a in axes[:, 0]:
        a.set_ylabel("true cross-track RMS [cm]", fontsize=9, color=INK_2)
    fig.suptitle("LQR weight sweep: tracking error vs control effort (nominal route, EKF in the loop)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def table(agg, title):
    lines = [title, "", "| Setting | n | Track RMS [cm] | Straight [cm] | Corner [cm] | Est. track RMS [cm] "
             "| EKF pos. RMSE [cm] | Route time [s] | RMS ω [rad/s] | ω jitter | Turn-in-place frac | Finished |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (fam, label, f), a in agg.items():
        c = lambda k: f"{100 * a[k][0]:.2f} ± {100 * a[k][1]:.2f}"
        lines.append(f"| {label} | {a['n']} | {c('track_rms_m')} | {c('straight_track_rms_m')} | {c('corner_track_rms_m')} "
                     f"| {c('est_track_rms_m')} | {c('est_err_rms_m')} | {a['route_time_s'][0]:.1f} ± {a['route_time_s'][1]:.1f} "
                     f"| {a['w_rms'][0]:.3f} | {a['w_jitter'][0]:.3f} ± {a['w_jitter'][1]:.3f} "
                     f"| {a['turn_in_place_frac'][0]:.3f} | {a['finished'][0] * a['n']:.0f}/{a['n']} |")
    return lines


KEYS = ["finished", "route_time_s", "track_rms_m", "track_max_m", "est_track_rms_m", "v_rms", "w_rms",
        "v_jitter", "w_jitter", "straight_track_rms_m", "corner_track_rms_m", "est_err_rms_m",
        "turn_in_place_frac", "w_sat_frac"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per sweep point")
    ap.add_argument("--investigate", type=int, default=20, help="seeds for the heavier-weight check (0 = skip)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--diagnose", type=int, default=0, help="seeds for the diagnosis only (skips the sweep)")
    args = ap.parse_args()
    base = load_config(args.config)
    lqr = base["controller"]["lqr"]
    out = results_dir()
    fig_path = ROOT / "docs" / "figures" / "lqr_weight_sweep.png"

    if args.diagnose:
        rows, lines = diagnose(base, lqr, range(args.diagnose), args.workers)
        write_csv(rows, out / "sweep_lqr_diagnose.csv")
        text = "\n".join(lines) + "\n"
        print(text)
        (out / "sweep_lqr_diagnose.md").write_text(text, encoding="utf-8")
        return

    if args.plot_only:
        plot(read_csv(out / "sweep_lqr_weights.csv"), lqr, fig_path)
        print(f"Saved {fig_path}")
        return

    rows = run_settings(base, sweep_settings(lqr), range(args.seeds), args.workers)
    write_csv(rows, out / "sweep_lqr_weights.csv")
    plot(rows, lqr, fig_path)
    md = table(aggregate(rows, KEYS), f"LQR weight sweep, one-at-a-time scaling, {args.seeds} seeds per point, "
                                      "mean ± std against ground truth.")

    if args.investigate:
        inv = [(lab, lab, f, scaled(lqr, key, idx, f)) for lab, key, idx, f in INVESTIGATE]
        inv_rows = run_settings(base, inv, range(args.investigate), args.workers)
        write_csv(inv_rows, out / "sweep_lqr_investigate.csv")
        md += [""] + table(aggregate(inv_rows, KEYS), f"Heavier weights, {args.investigate} seeds.")

    text = "\n".join(md) + "\n"
    print(text)
    (out / "sweep_lqr_weights.md").write_text(text, encoding="utf-8")
    print(f"Saved {fig_path} and results/sweep_lqr_weights.md")


if __name__ == "__main__":
    main()
