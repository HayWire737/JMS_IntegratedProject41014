"""Differential-drive kinematic model of the TurtleBot3 (Part A).

State  x = [x, y, theta]   base_link pose in the world frame
Input  u = [v, w]          forward speed (m/s), yaw rate (rad/s)

Continuous time
    x_dot     = v cos(theta)
    y_dot     = v sin(theta)
    theta_dot = w
Wheel speeds (r = wheel radius, b = track width)
    v = r (w_R + w_L) / 2,   w = r (w_R - w_L) / b

Assumptions: planar motion, no wheel slip, rigid wheels of known radius and
track width, point contact. The derivations are in docs/part_A_C_notes.md.
"""
import numpy as np

from src.frames.se2 import wrap


# ----------------------------------------------------------------- kinematics
def body_to_wheels(v, w, r, b):
    return (v - w * b / 2) / r, (v + w * b / 2) / r


def wheels_to_body(w_left, w_right, r, b):
    return r * (w_right + w_left) / 2, r * (w_right - w_left) / b


def continuous_dynamics(x, u):
    """x_dot = f_c(x, u)."""
    v, w = u
    return np.array([v * np.cos(x[2]), v * np.sin(x[2]), w])


# ----------------------------------------------------- discrete odometry model
def motion_model(x, ds, dth):
    """Discrete motion model driven by odometry increments (midpoint rule).

    ds  = distance travelled by the robot centre over the step
    dth = change in heading over the step
    Uses the heading halfway through the step, which is exact for a constant
    curvature arc to second order and much better than using the start heading.
    """
    m = x[2] + dth / 2
    return np.array([x[0] + ds * np.cos(m), x[1] + ds * np.sin(m), wrap(x[2] + dth)])


def jacobians(x, ds, dth):
    """Jacobians of motion_model, used by the EKF prediction.

    F = d f / d x            (3x3)  how state uncertainty propagates
    G = d f / d [ds, dth]    (3x2)  how odometry noise enters the state
    """
    m = x[2] + dth / 2
    s, c = np.sin(m), np.cos(m)
    F = np.array([[1.0, 0.0, -ds * s],
                  [0.0, 1.0, ds * c],
                  [0.0, 0.0, 1.0]])
    G = np.array([[c, -0.5 * ds * s],
                  [s, 0.5 * ds * c],
                  [0.0, 1.0]])
    return F, G


# --------------------------------------------- linearised tracking-error model
def error_dynamics(v_r, w_r):
    """Linearised tracking-error dynamics about a reference (v_r, w_r).

    Error expressed in the robot frame:
        e = R(theta)^T (p_ref - p),   e_theta = theta_ref - theta
    Input deviation: du = u_ref_feedforward - u. Linearising gives
        e_dot = A e + B du
    with
        A = [[0,    w_r, 0  ],        B = [[1, 0],
             [-w_r, 0,   v_r],             [0, 0],
             [0,    0,   0  ]]             [0, 1]]
    """
    A = np.array([[0.0, w_r, 0.0],
                  [-w_r, 0.0, v_r],
                  [0.0, 0.0, 0.0]])
    B = np.array([[1.0, 0.0],
                  [0.0, 0.0],
                  [0.0, 1.0]])
    return A, B


def discretise(A, B, dt):
    """Zero-order-hold discretisation via the matrix exponential."""
    from scipy.linalg import expm
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


# ------------------------------------------------------- structural analysis
def controllability_matrix(A, B):
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def observability_matrix(A, C):
    n = A.shape[0]
    blocks = [C]
    for _ in range(n - 1):
        blocks.append(blocks[-1] @ A)
    return np.vstack(blocks)


def rank(M, tol=1e-9):
    return int(np.linalg.matrix_rank(M, tol))


def range_bearing(x, landmark):
    """Range and bearing from base_link to a known landmark (world frame)."""
    dx, dy = landmark[0] - x[0], landmark[1] - x[1]
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])


def range_bearing_jacobian(x, landmark):
    """H = d h / d x for the range/bearing measurement (2x3)."""
    dx, dy = landmark[0] - x[0], landmark[1] - x[1]
    q = dx ** 2 + dy ** 2
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / q, -dx / q, -1.0]])


def local_observability_rank(x_traj, u_traj, landmark, visible):
    """Rank of the stacked time-varying observability matrix along a trajectory.

    O = [H_0; H_1 F_0; H_2 F_1 F_0; ...] using only steps where the marker is
    visible. Rank 3 means the full pose can be reconstructed from the
    measurements in that window.
    """
    Phi = np.eye(3)
    rows = []
    for x, (ds, dth), seen in zip(x_traj, u_traj, visible):
        if seen:
            rows.append(range_bearing_jacobian(x, landmark) @ Phi)
        F, _ = jacobians(x, ds, dth)
        Phi = F @ Phi
    if not rows:
        return 0
    return rank(np.vstack(rows), tol=1e-6)
