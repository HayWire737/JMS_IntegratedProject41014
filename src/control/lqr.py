"""LQR trajectory tracker.  TODO (Part D, control lead).

Suggested approach:
  1. Express the tracking error in the robot frame: e = R(theta)^T (x_ref - x)
  2. Linearise the error dynamics about the reference (v_r, w_r)  -> A(v_r), B
  3. Discretise with the control period, solve the discrete Riccati equation
     (scipy.linalg.solve_discrete_are) for gain K given weights Q_lqr, R_lqr
  4. u = u_ref - K e, then saturate to v_max, w_max

Same interface as GoToGoal so run.py can swap controllers:
    ctrl.command(x_est, goal) -> (v, w)
Record Q_lqr / R_lqr in the config and justify them on the project page.
"""


class LQRTracker:
    def __init__(self, ctrl_cfg, robot_cfg):
        raise NotImplementedError

    def command(self, x_est, goal):
        raise NotImplementedError
