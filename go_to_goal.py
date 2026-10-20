"""Baseline controller: polar proportional go-to-goal on the ESTIMATED pose.

rho   = distance to goal
alpha = heading error to the goal
v = k_rho * rho * cos(alpha),  w = k_alpha * alpha   (both saturated)

The cos(alpha) term makes the robot slow down while it is pointing away from
the goal, so it turns first and then drives. To be replaced by LQR (Part D).
"""
import numpy as np

from src.frames.se2 import wrap


class GoToGoal:
    def __init__(self, ctrl_cfg, robot_cfg):
        self.k_rho = ctrl_cfg["k_rho"]
        self.k_alpha = ctrl_cfg["k_alpha"]
        self.v_max = robot_cfg["v_max"]
        self.w_max = robot_cfg["w_max"]

    def command(self, x_est, goal):
        dx, dy = goal[0] - x_est[0], goal[1] - x_est[1]
        rho = np.hypot(dx, dy)
        alpha = wrap(np.arctan2(dy, dx) - x_est[2])
        v = np.clip(self.k_rho * rho * max(np.cos(alpha), 0.0), 0.0, self.v_max)
        w = np.clip(self.k_alpha * alpha, -self.w_max, self.w_max)
        return float(v), float(w)
