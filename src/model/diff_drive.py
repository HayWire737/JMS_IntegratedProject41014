"""Differential-drive kinematic model of the TurtleBot3.

State  x = [x, y, theta]   (base_link pose in the world frame)
Input  u = [v, w]          (forward speed, yaw rate)

Continuous time:  x_dot = v cos(theta),  y_dot = v sin(theta),  theta_dot = w
Wheel speeds:     v = r (w_R + w_L) / 2,  w = r (w_R - w_L) / b

Assumptions to state on the project page: planar motion, no wheel slip,
rigid wheels of known radius r and track width b.
"""
import numpy as np

from src.frames.se2 import wrap


def motion_model(x, ds, dth):
    """Discrete odometry motion model (midpoint integration).

    ds  = distance travelled by the robot centre over the step
    dth = change in heading over the step
    """
    th_mid = x[2] + dth / 2
    return np.array([x[0] + ds * np.cos(th_mid), x[1] + ds * np.sin(th_mid), wrap(x[2] + dth)])


def body_to_wheels(v, w, r, b):
    return (v - w * b / 2) / r, (v + w * b / 2) / r


def wheels_to_body(w_left, w_right, r, b):
    return r * (w_right + w_left) / 2, r * (w_right - w_left) / b


# ---------------------------------------------------------------- TODO (Part A)
# Modelling lead:
#  - jacobians(x, ds, dth) -> F = d f / d x, G = d f / d [ds, dth]   (needed by the EKF)
#  - linearise the tracking-error dynamics about a reference (v_r, w_r)
#  - controllability rank of (A, B), and why it drops when v_r = 0
#  - observability of (A, C) with odometry only vs. with marker range/bearing
def jacobians(x, ds, dth):
    raise NotImplementedError("Part A: derive F and G for the EKF")
