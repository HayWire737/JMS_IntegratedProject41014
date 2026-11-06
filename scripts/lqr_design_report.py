"""LQR design check: gains and closed-loop poles across reference speeds.

For each speed the tracking-error model is linearised and discretised at the
control period, the LQR gain K is solved for the config weights, and the poles
of Ad - Bd K are taken from LQRTracker.closed_loop_poles. Each pole is also
given as a continuous-time equivalent (time constant, damping ratio). The
script checks that every |pole| < 1 and that only the Q/R ratio sets the gain.

    python scripts/lqr_design_report.py
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import ROOT, load_config, results_dir
from src.control.lqr import LQRTracker
from src.model.diff_drive import discretise, error_dynamics

INK, INK_2, GRID, SURFACE = "#1f1e1c", "#5f5e58", "#e4e3de", "#fcfcfb"
RAMP = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]     # ordinal: slow -> fast
MODE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]                    # categorical slots 1-3
MODE_NAMES = ["along-track (e_x)", "e_y/e_θ fast", "e_y/e_θ slow (path convergence)"]


def modes(lqr, v, dt):
    """Closed-loop poles at speed v: the along-track mode (eigenvector along e_x), then the
    two coupled cross-track/heading modes, fast before slow."""
    v_lin = round(max(v, lqr.p["v_lin_min"]), 2)          # same 1 cm/s grid as LQRTracker.gain
    Ad, Bd = discretise(*error_dynamics(v_lin, 0.0), dt)
    z, V = np.linalg.eig(Ad - Bd @ lqr.gain(v_lin))
    ix = int(np.argmax(np.abs(V[0, :])))
    rest = sorted((k for k in range(3) if k != ix), key=lambda k: abs(z[k]))
    return [z[ix]] + [z[k] for k in rest]


def continuous_equivalent(z, dt):
    """Continuous pole s = ln(z)/dt, its time constant -1/Re(s) and damping ratio -Re(s)/|s|."""
    s = np.log(z.astype(complex)) / dt
    return s, -1.0 / s.real, -s.real / np.abs(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ctl, rob, dt = cfg["controller"], cfg["robot"], cfg["sim"]["control_dt"]
    lqr = LQRTracker(ctl, rob, dt=dt)
    p = lqr.p
    speeds = [("v = 0 (uses v_lin_min)", 0.0), ("v_lin_min", p["v_lin_min"]),
              ("docking v_ref", cfg.get("docking", {}).get("lqr", {}).get("v_ref", 0.08)),
              ("v_ref", p["v_ref"]), ("v_max", rob["v_max"])]

    lines = [f"Q_lqr = {p['Q_lqr']}, R_lqr = {p['R_lqr']}, dt = {dt} s", ""]
    all_stable, plotted = True, []
    for name, v in speeds:
        K = lqr.gain(v)
        z = lqr.closed_loop_poles(v)
        stable = bool(np.all(np.abs(z) < 1.0))
        all_stable &= stable
        lines.append(f"{name}: v = {v:.3f} m/s (linearised at {max(v, p['v_lin_min']):.3f})")
        lines.append(f"  K = [[{K[0, 0]:+.3f} {K[0, 1]:+.3f} {K[0, 2]:+.3f}]")
        lines.append(f"       [{K[1, 0]:+.3f} {K[1, 1]:+.3f} {K[1, 2]:+.3f}]]   (rows: dv, dw; cols: e_x, e_y, e_theta)")
        zm = np.array(modes(lqr, v, dt))
        _, tau, zeta = continuous_equivalent(zm, dt)
        for mode, zi, ti, ci in zip(MODE_NAMES, zm, tau, zeta):
            lines.append(f"  {mode:<32} pole {zi.real:+.4f}{zi.imag:+.4f}j  |z| = {abs(zi):.4f}  "
                         f"tau = {ti:.2f} s  zeta = {ci:.2f}  decay distance v*tau = {max(v, p['v_lin_min']) * ti:.2f} m")
        lines.append(f"  all |pole| < 1: {stable}")
        if v > 0 or not plotted:
            plotted.append((name, max(v, p["v_lin_min"]), z))
    lines.append("")
    lines.append(f"Stable at every speed: {all_stable}")

    # Only the ratio of Q to R sets the gain
    K0 = lqr.gain(p["v_ref"])
    for f in (0.25, 4.0):
        a = LQRTracker({"lqr": {**p, "Q_lqr": [q * f for q in p["Q_lqr"]]}}, rob, dt=dt).gain(p["v_ref"])
        b = LQRTracker({"lqr": {**p, "R_lqr": [r / f for r in p["R_lqr"]]}}, rob, dt=dt).gain(p["v_ref"])
        lines.append(f"Q x{f:g} vs R x{1 / f:g}: max |K difference| = {np.max(np.abs(a - b)):.1e} "
                     f"(K changes by {np.max(np.abs(a - K0)):.3f} from the baseline)")

    text = "\n".join(lines)
    print(text)
    (results_dir() / "lqr_design_report.txt").write_text(text + "\n", encoding="utf-8")

    # Left: poles in the z-plane with the unit circle. Right: |pole| of each mode against speed.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.25]})
    th = np.linspace(0, 2 * np.pi, 400)
    plotted = [t for t in plotted if t[0] != "v = 0 (uses v_lin_min)"]
    ax = axes[0]
    ax.plot(np.cos(th), np.sin(th), color=INK_2, lw=1.2, zorder=1)
    for (name, v, z), c in zip(plotted, RAMP[1:]):
        ax.scatter(z.real, z.imag, s=70, marker="x", color=c, lw=2, zorder=3, label=f"{name} ({v:.2f} m/s)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("Re z", fontsize=9, color=INK_2)
    ax.set_ylabel("Im z", fontsize=9, color=INK_2)
    ax.set_title("Closed-loop poles of Ad − Bd K (all real)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="lower left")

    ax = axes[1]
    vs = np.round(np.arange(p["v_lin_min"], rob["v_max"] + 1e-9, 0.01), 2)
    mags = np.abs(np.array([modes(lqr, v, dt) for v in vs]))
    for i, dy in zip(range(3), (7, -9, 0)):
        ax.plot(vs, mags[:, i], color=MODE_COLOURS[i], lw=2)
        ax.annotate(MODE_NAMES[i], (vs[-1], mags[-1, i]), xytext=(6, dy), textcoords="offset points",
                    fontsize=8, color=INK_2, va="center")
    ax.axhline(1.0, color=INK_2, lw=1, ls="--")
    ax.annotate("stability limit |z| = 1", (vs[0], 1.0), xytext=(0, 4), textcoords="offset points", fontsize=8,
                color=INK_2)
    for name, v, _ in plotted:
        ax.axvline(v, color=GRID, lw=1, zorder=0)
        ax.annotate(name, (v, 0.90), rotation=90, xytext=(3, 0), textcoords="offset points", fontsize=7,
                    color=INK_2, va="bottom")
    ax.set_xlim(0.04, rob["v_max"] + 0.11)
    ax.set_ylim(0.85, 1.01)
    ax.set_xlabel("reference speed v [m/s]", fontsize=9, color=INK_2)
    ax.set_ylabel("|pole|", fontsize=9, color=INK_2)
    ax.set_title("Pole magnitude per mode vs speed", fontsize=10, color=INK, loc="left")
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.6)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
    fig.suptitle(f"LQR closed-loop poles, Q = diag{tuple(p['Q_lqr'])}, R = diag{tuple(p['R_lqr'])}, "
                 f"dt = {dt} s: all inside the unit circle = {all_stable}", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "lqr_poles.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Saved {out} and results/lqr_design_report.txt")


if __name__ == "__main__":
    main()
