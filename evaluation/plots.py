"""Standard figures for a logged run."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from evaluation.metrics import estimation_errors


def _draw_room(ax, cfg):
    ax.add_patch(Rectangle((0, 0), 5, 4, fill=False, lw=2, color="0.3"))
    for ob in cfg.get("obstacles", []):
        if ob["type"] == "box":
            ax.add_patch(Rectangle((ob["x"] - ob["hx"], ob["y"] - ob["hy"]), 2 * ob["hx"], 2 * ob["hy"],
                                   color="#a0522d", alpha=0.6))
        else:
            ax.add_patch(Circle((ob["x"], ob["y"]), ob["r"], color="#a0522d", alpha=0.6))
    mx, my, _ = cfg["marker"]["pose_world"]
    ax.plot(mx, my, "s", color="#1f4fbf", ms=10, label="ArUco dock")


def trajectory_and_errors(log, cfg, out_path, title="Run"):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    _draw_room(ax, cfg)
    ax.plot(log["truth"][:, 0], log["truth"][:, 1], "k-", lw=2, label="true")
    ax.plot(log["est"][:, 0], log["est"][:, 1], "--", color="#d62728", lw=2, label="estimate")
    wp = np.array(cfg["waypoints"])
    ax.plot(wp[:, 0], wp[:, 1], "o", mfc="none", mec="#2ca02c", ms=10, mew=2, label="waypoints")
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.2, 4.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)

    e = estimation_errors(log)
    t = log["t"]
    ax2.plot(t, np.hypot(e[:, 0], e[:, 1]), label="position error [m]")
    ax2.plot(t, np.abs(e[:, 2]), label="|heading error| [rad]")
    seen = log["marker_seen"]
    if seen.any():
        ax2.fill_between(t, 0, 1, where=seen, transform=ax2.get_xaxis_transform(),
                         color="#1f4fbf", alpha=0.12, label="marker in view")
    ax2.set_xlabel("time [s]")
    ax2.set_title("Estimation error (estimate − truth)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def errors_with_bounds(log, out_path, title="Estimation error with 3-sigma bounds"):
    """Each state error against +-3 sigma from the filter covariance.

    A consistent filter keeps ~99 % of the error inside the bounds; bounds that
    grow show uncertainty building up while the marker is out of view.
    """
    e = estimation_errors(log)
    t = log["t"]
    sig = np.sqrt(np.stack([log["P"][:, i, i] for i in range(3)], axis=1))
    labels = ["x error [m]", "y error [m]", "heading error [rad]"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    seen = log["marker_seen"]
    for i, ax in enumerate(axes):
        ax.plot(t, e[:, i], color="#d62728", lw=1.5, label="error (estimate − truth)")
        ax.fill_between(t, -3 * sig[:, i], 3 * sig[:, i], color="0.6", alpha=0.35, label="±3σ (filter)")
        if seen.any():
            ax.fill_between(t, 0, 1, where=seen, transform=ax.get_xaxis_transform(),
                            color="#1f4fbf", alpha=0.08, label="marker in view")
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left", ncol=3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
