"""Part B: draw the frame diagram for the project page.

Every number comes from the config, so the figure always matches the code:
  world      fixed room frame, origin at the south-west corner
  base_link  robot centre between the wheels, x forward, y left   (the EKF state)
  lidar      scan centre, axes parallel to base_link               (lidar.x, lidar.z)
  camera     OpenCV optical frame: z forward, x right, y down      (camera.x, camera.z)
  marker     ArUco frame on the dock, z out of the marker face     (marker.pose_world)

There is no separate odom frame: the start pose is known, so odometry is
integrated directly in the world frame and the EKF corrects that estimate.

    python scripts/plot_frames.py          # writes docs/figures/frame_diagram.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from _common import ROOT, load_config

X_COL, Y_COL, Z_COL = "#d62728", "#2ca02c", "#1f77b4"     # x red, y green, z blue (ROS convention)


def axes(ax, origin, yaw, length, label, x_name="x", y_name="y", fs=9, label_at=None):
    """Draw a frame's x (red) and y (green) axes at `origin`, rotated by `yaw`."""
    ox, oy = origin
    for ang, col, name in ((yaw, X_COL, x_name), (yaw + np.pi / 2, Y_COL, y_name)):
        dx, dy = length * np.cos(ang), length * np.sin(ang)
        ax.annotate("", (ox + dx, oy + dy), (ox, oy), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        ax.text(ox + 1.18 * dx, oy + 1.18 * dy, name, color=col, fontsize=fs, ha="center", va="center")
    ax.plot(ox, oy, "ko", ms=3)
    lx, ly = label_at if label_at is not None else (ox - 0.08 * length, oy - 0.22 * length)
    ax.text(lx, ly, label, fontsize=fs, ha="right", va="top", weight="bold")


def arrow(ax, start, vec, col, name, fs=9):
    ax.annotate("", (start[0] + vec[0], start[1] + vec[1]), start,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    ax.text(start[0] + 1.2 * vec[0], start[1] + 1.2 * vec[1], name, color=col, fontsize=fs,
            ha="center", va="center")


def world_view(ax, cfg, pose):
    ax.add_patch(Rectangle((0, 0), 5, 4, fill=False, lw=2, color="0.3"))
    for ob in cfg["obstacles"]:
        if ob["type"] == "box":
            ax.add_patch(Rectangle((ob["x"] - ob["hx"], ob["y"] - ob["hy"]), 2 * ob["hx"], 2 * ob["hy"],
                                   color="#a0522d", alpha=0.5))
        else:
            ax.add_patch(Circle((ob["x"], ob["y"]), ob["r"], color="#a0522d", alpha=0.5))
    mx, my, myaw = cfg["marker"]["pose_world"]
    ax.add_patch(Rectangle((mx, my - 0.2), 0.14, 0.4, color="#1f4fbf", alpha=0.6))       # dock
    axes(ax, (0, 0), 0.0, 0.6, "world")
    axes(ax, pose[:2], pose[2], 0.45, "base_link")
    # marker frame drawn in the plane: its z (out of the face) points along the marker normal
    arrow(ax, (mx, my), (0.45 * np.cos(myaw), 0.45 * np.sin(myaw)), Z_COL, "z")
    ax.text(mx - 0.1, my + 0.45, "marker", fontsize=9, weight="bold", ha="center")
    ax.plot([pose[0], mx], [pose[1], my], ":", color="0.4", lw=1)
    ax.text((pose[0] + mx) / 2 + 0.1, (pose[1] + my) / 2 - 0.12, "camera measures\nmarker pose",
            fontsize=8, ha="center", va="top", color="0.3")
    ax.set(xlim=(-0.4, 5.4), ylim=(-0.4, 4.3), aspect="equal", title="(a) Room, top view",
           xlabel="x (m)", ylabel="y (m)")
    ax.text(0.02, 3.9, f"world → marker: ({mx}, {my}) m,\nnormal at {np.degrees(myaw):.0f}°, "
                       f"centre {cfg['camera']['z']} m high", fontsize=8, va="top")
    ax.text(pose[0] - 1.35, pose[1] + 0.75, "world → base_link:\nEKF state [x, y, θ]", fontsize=8,
            ha="center")


def robot_top(ax, cfg):
    r, half = cfg["robot"]["wheel_radius"], cfg["robot"]["wheel_base"] / 2
    lid, cam = cfg["lidar"], cfg["camera"]
    ax.add_patch(Rectangle((-0.09, -0.07), 0.14, 0.14, color="0.25", alpha=0.25))      # chassis
    for s in (1, -1):
        ax.add_patch(Rectangle((-r, s * half - 0.009), 2 * r, 0.018, color="0.1"))       # wheels
    ax.add_patch(Circle((lid["x"], 0), 0.035, fill=False, color="0.3", lw=1.5))
    axes(ax, (0, 0), 0.0, 0.055, "base_link", fs=8, label_at=(0.05, -0.012))
    # lidar axes are parallel to base_link and only 2 cm behind, so mark its origin instead
    ax.plot(lid["x"], 0, "o", color="0.3", ms=5)
    ax.annotate(f"lidar origin\n(axes ∥ base_link)", (lid["x"], 0), (-0.17, 0.09), fontsize=8,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
    # camera optical frame seen from above: z forward (= base x), x right (= base -y)
    c = (cam["x"], 0.0)
    arrow(ax, c, (0.08, 0), Z_COL, "z cam", fs=8)
    arrow(ax, c, (0, -0.08), X_COL, "x cam", fs=8)
    ax.plot(*c, "ks", ms=4)
    ax.text(cam["x"] + 0.01, 0.045, "camera\n(y cam into page)", fontsize=8)
    ax.text(-0.19, -0.19, f"base_link → lidar: ({lid['x']}, 0, {lid['z']}) m, no rotation\n"
                          f"base_link → camera: ({cam['x']}, 0, {cam['z']}) m,\n"
                          f"   z_cam = x_base, x_cam = −y_base, y_cam = −z_base\n"
                          f"wheel track {cfg['robot']['wheel_base']} m, radius {r} m",
            fontsize=8, va="top")
    ax.set(xlim=(-0.2, 0.25), ylim=(-0.32, 0.17), aspect="equal", title="(b) Robot, top view",
           xlabel="x base (m)", ylabel="y base (m)")


def robot_side(ax, cfg):
    r, lid, cam = cfg["robot"]["wheel_radius"], cfg["lidar"], cfg["camera"]
    ax.axhline(0, color="0.4", lw=1)
    ax.add_patch(Rectangle((-0.09, 0.02), 0.14, 0.10, color="0.25", alpha=0.25))       # chassis
    ax.add_patch(Circle((0, r), r, color="0.1", alpha=0.8))                              # wheel
    ax.add_patch(Rectangle((lid["x"] - 0.035, lid["z"] - 0.02), 0.07, 0.04, color="0.3", alpha=0.6))
    axes(ax, (0, 0), 0.0, 0.08, "base_link", x_name="x", y_name="z", fs=8, label_at=(-0.01, -0.005))
    c = (cam["x"], cam["z"])
    arrow(ax, c, (0.07, 0), Z_COL, "z cam", fs=8)
    arrow(ax, c, (0, -0.06), Y_COL, "y cam", fs=8)
    ax.plot(*c, "ks", ms=4)
    ax.plot(lid["x"], lid["z"], "ko", ms=3)
    ax.text(lid["x"] - 0.04, lid["z"] + 0.03, f"lidar scan plane, z = {lid['z']} m", fontsize=8)
    ax.plot([0.165, 0.22], [cam["z"]] * 2, ":", color=Z_COL, lw=1)
    ax.text(0.215, cam["z"] - 0.01, "marker centre\nheight (same)", fontsize=8, color=Z_COL, ha="right", va="top")
    ax.set(xlim=(-0.15, 0.22), ylim=(-0.03, 0.25), aspect="equal", title="(c) Robot, side view",
           xlabel="x base (m)", ylabel="z (m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--pose", default="3.3,1.7,0.35", help="example robot pose x,y,yaw for panel (a)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pose = np.array([float(v) for v in args.pose.split(",")])

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
    world_view(fig.add_subplot(gs[0]), cfg, pose)
    robot_top(fig.add_subplot(gs[1]), cfg)
    robot_side(fig.add_subplot(gs[2]), cfg)
    fig.suptitle("Frames: world → base_link (EKF) → lidar / camera → marker      "
                 "x red, y green, z blue", fontsize=11)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "frame_diagram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()