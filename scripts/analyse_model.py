"""Part A: controllability and observability analysis.

Prints rank tables and saves results/part_A_observability.png.

    python scripts/analyse_model.py
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import load_config, results_dir
from src.estimation.ekf import EKF
from src.model.diff_drive import (controllability_matrix, discretise, error_dynamics, jacobians,
                                  local_observability_rank, motion_model, rank,
                                  range_bearing_jacobian)


def pose_H(f, x):
    return f.measurement_model(x)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dt = cfg["sim"]["control_dt"]
    L = np.array(cfg["marker"]["pose_world"])
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # ------------------------------------------------------- controllability
    out("CONTROLLABILITY of the linearised tracking-error model  e_dot = A e + B du")
    out("  reference (v_r, w_r)      rank C (cont.)   rank C (discrete, dt)   sigma_min(C_d)")
    for v_r, w_r in [(0.2, 0.0), (0.2, 0.8), (0.05, 0.0), (0.0, 0.8), (0.0, 0.0)]:
        A, B = error_dynamics(v_r, w_r)
        Ad, Bd = discretise(A, B, dt)
        Cc = controllability_matrix(A, B)
        n = 3
        Cd = np.hstack([np.linalg.matrix_power(Ad, k) @ Bd for k in range(n)])
        smin = np.linalg.svd(Cd, compute_uv=False)[-1]
        out(f"  ({v_r:4.2f} m/s, {w_r:4.2f} rad/s)   {rank(Cc):>8}         {rank(Cd, 1e-12):>10}             {smin:.2e}")
    out("  -> full rank (3) whenever the reference moves (v_r != 0 or w_r != 0). The rotating")
    out("     error frame couples the lateral error to the inputs through w_r and v_r.")
    out("     At rest (v_r = w_r = 0) rank drops to 2: no input moves the robot sideways (nonholonomic).")
    out("     sigma_min shrinks as the reference slows: lateral control authority vanishes, which is")
    out("     why the final docking approach needs a moving reference or a different control law.")
    out()

    # ----------------------------------------------------------- observability
    out("OBSERVABILITY of the pose [x, y, theta]")
    out("  Odometry only: no measurement of absolute pose (C = 0)        rank 0 -> unobservable, drift unbounded")

    # A trajectory approaching the marker on an arc, marker always in view
    x = np.array([2.2, 1.4, 0.25])
    xs, us = [], []
    for k in range(60):
        ds, dth = 0.2 * dt, 0.15 * dt * np.cos(k / 15)
        xs.append(x.copy())
        us.append((ds, dth))
        x = motion_model(x, ds, dth)
    vis = [True] * len(xs)
    f = EKF(xs[0], cfg)

    H_rb0 = range_bearing_jacobian(xs[0], L)
    out(f"  Range+bearing, one measurement, robot static                  rank {rank(H_rb0)}")
    out(f"  Range+bearing, 60 measurements while moving                   rank {local_observability_rank(xs, us, L, vis)}")
    dx, dy = L[0] - xs[0][0], L[1] - xs[0][1]
    null = np.array([dy, -dx, 1.0])
    out(f"     null direction [dy, -dx, 1] -> H @ null = {np.round(H_rb0 @ null, 12)}")
    out("     = rotating the robot about the marker: range and bearing do not change, odometry is")
    out("       identical in the robot frame, so NO amount of motion makes this direction observable.")

    # Pose measurement: stacked rank
    Phi, rows = np.eye(3), []
    for xk, (ds, dth) in zip(xs, us):
        rows.append(pose_H(f, xk) @ Phi)
        Phi = jacobians(xk, ds, dth)[0] @ Phi
    out(f"  Marker pose (x, y, yaw), one measurement                      rank {rank(pose_H(f, xs[0]))}")
    out(f"  Marker pose, 60 measurements while moving                     rank {rank(np.vstack(rows), 1e-6)}")
    out("     the measured marker orientation fixes the rotation about the marker -> fully observable.")
    out()

    # ---------------------------------------------- weak observability vs range
    ranges = np.linspace(0.6, 3.0, 25)
    min_info_pose, min_info_rb = [], []
    for r in ranges:
        xr = np.array([L[0] - r, L[1], 0.0])
        Hp = pose_H(f, xr)
        Rp = f.measurement_noise(r)
        Ip = Hp.T @ np.linalg.inv(Rp) @ Hp                 # Fisher information of one measurement
        min_info_pose.append(np.linalg.eigvalsh(Ip)[0])
        Hr = range_bearing_jacobian(xr, L)
        sr = [cfg["ekf"]["marker_sigma"][k][0] + cfg["ekf"]["marker_sigma"][k][1] * r ** 2
              for k in ("range", "bearing")]
        Ir = Hr.T @ np.diag(1 / np.square(sr)) @ Hr
        min_info_rb.append(max(np.linalg.eigvalsh(Ir)[0], 1e-12))
    out("WEAKEST DIRECTION: smallest eigenvalue of H^T R^-1 H (information from one measurement)")
    for r, ip in zip(ranges[::6], min_info_pose[::6]):
        out(f"  range {r:.1f} m   pose model {ip:9.2f}   (std in weakest direction {1/np.sqrt(ip):.3f})   range+bearing 0")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(ranges, min_info_pose, "o-", label="marker pose (x, y, yaw)")
    ax.semilogy(ranges, min_info_rb, "s--", label="range + bearing (numerically 0)")
    ax.set_xlabel("range to marker [m]")
    ax.set_ylabel("smallest eigenvalue of HᵀR⁻¹H")
    ax.set_title("Information in the weakest pose direction, one measurement")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir() / "part_A_observability.png", dpi=120)
    (results_dir() / "part_A_analysis.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved {results_dir() / 'part_A_analysis.txt'} and part_A_observability.png")


if __name__ == "__main__":
    main()
