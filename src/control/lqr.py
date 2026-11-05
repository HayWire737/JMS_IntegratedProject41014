"""LQR waypoint tracker. Works from the estimated pose only.

Same interface as GoToGoal, so scripts/run.py can swap one for the other:

    ctrl = LQRTracker(ctrl_cfg, robot_cfg, dt=sim.dt)
    v, w = ctrl.command(x_est, goal)

How it works
------------
1. Reference. The waypoint follower supplies one waypoint at a time. The line
   from the previous waypoint (or the first estimated pose) to the current one
   is treated as a segment, and a reference point moves along it at a scheduled
   speed v_r. The speed ramps down over the last stretch of each segment
   (v_r <= slowdown_gain * remaining distance, never below v_min).
2. Error. The gap between the reference and the robot is expressed in the robot
   frame, giving along-track, cross-track and heading errors:
       e = R(theta)^T (p_ref - p),   e_theta = theta_ref - theta
3. Gain. The error dynamics are linearised about (v_r, w_r = 0) using
   error_dynamics from src/model/diff_drive.py and discretised at the control
   period. The discrete Riccati equation then gives the LQR gain K for the
   weights Q_lqr and R_lqr. With du = u_ref - u = -K e, the command is
       v = v_r cos(e_theta) + (K e)[0],   w = w_r + (K e)[1]
4. Limits. The command is clipped to the robot's speed limits and an
   acceleration limit.

Behaviour worth knowing
-----------------------
* Standstill. When the reference speed is zero the lateral error cannot be
  controlled, so the gain is computed at a small minimum speed (v_lin_min).
* Sharp turns. The linearisation only holds for small heading errors. When the
  heading error to a new segment is large (align_enter), the robot rotates in
  place until it is small again (align_exit), then resumes tracking. Using two
  thresholds avoids switching back and forth near the limit.
* Lag. The reference slows down when the robot falls behind (saturation, a
  LiDAR stop, a disturbance), so it does not run away from the robot.
* Direction. The forward speed is never negative, since the LiDAR safety stop
  in run.py only checks ahead of the robot.
"""
import numpy as np
from scipy.linalg import solve_discrete_are

from src.frames.se2 import wrap
from src.model.diff_drive import discretise, error_dynamics

DEFAULTS = {
    "Q_lqr": [30.0, 60.0, 8.0],   # weights on [e_x (m), e_y (m), e_theta (rad)]
    "R_lqr": [4.0, 1.0],          # weights on [dv (m/s), dw (rad/s)]
    "v_ref": 0.15,                # cruise speed along a segment [m/s]
    "v_min": 0.03,                # creep speed floor while slowing into a waypoint [m/s]
    "slowdown_gain": 1.0,         # v_r <= gain * remaining segment length [1/s]
    "v_lin_min": 0.05,            # speed used to linearise when v_r is smaller [m/s]
    "w_turn_max": 1.5,            # yaw-rate cap while rotating in place [rad/s]
    "align_enter": 0.60,          # heading error that triggers an in-place turn [rad]
    "align_exit": 0.15,           # heading error at which tracking resumes [rad]
    "pos_hold": 0.30,             # reference stops when position error reaches this [m]
    "ang_hold": 0.80,             # ... or heading error reaches this [rad]
    "a_max": 0.5,                 # linear acceleration limit [m/s^2], null = off
    "alpha_max": 4.0,             # angular acceleration limit [rad/s^2], null = off
}


def dlqr(Ad, Bd, Q, R):
    """Solve the discrete LQR problem. Returns the gain K (du = -K e) and the Riccati solution P."""
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


class LQRTracker:
    def __init__(self, ctrl_cfg, robot_cfg, dt=0.05):
        self.p = {**DEFAULTS, **(ctrl_cfg.get("lqr") or {})}
        self.dt = float(dt)
        self.Q = np.diag(np.asarray(self.p["Q_lqr"], dtype=float))
        self.R = np.diag(np.asarray(self.p["R_lqr"], dtype=float))
        self.v_max = robot_cfg["v_max"]
        self.w_max = robot_cfg["w_max"]
        self._gains = {}
        self.history = []          # one record per control step, for plots and analysis
        self.reset()

    def reset(self):
        """Clear the current segment and history, e.g. before reusing the object for another run."""
        self._goal = None
        self._p0 = np.zeros(2)
        self._u = np.array([1.0, 0.0])
        self._L = 0.0
        self._th_r = 0.0
        self._s = 0.0
        self._align = False
        self._throttle = 1.0
        self._u_prev = np.zeros(2)
        self.history = []

    def start_from_here(self):
        """Begin the next segment at the robot's current position instead of the previous
        waypoint. Called by the obstacle avoidance when it inserts or drops a detour, since
        the robot is then not at the previous waypoint. History is kept."""
        self._goal = None

    # ------------------------------------------------------------------ gains
    def gain(self, v_r):
        """LQR gain K for reference speed v_r, cached on a 1 cm/s grid."""
        v_lin = max(float(v_r), self.p["v_lin_min"])
        key = round(v_lin, 2)
        if key not in self._gains:
            A, B = error_dynamics(key, 0.0)
            Ad, Bd = discretise(A, B, self.dt)
            self._gains[key], _ = dlqr(Ad, Bd, self.Q, self.R)
        return self._gains[key]

    def closed_loop_poles(self, v_r=None):
        """Eigenvalues of Ad - Bd K at speed v_r (the loop is stable when all lie inside the unit circle)."""
        v_lin = max(self.p["v_ref"] if v_r is None else float(v_r), self.p["v_lin_min"])
        key = round(v_lin, 2)
        A, B = error_dynamics(key, 0.0)
        Ad, Bd = discretise(A, B, self.dt)
        return np.linalg.eigvals(Ad - Bd @ self.gain(key))

    # ---------------------------------------------------------------- segment
    def _new_segment(self, x, goal):
        """Start a new segment from the previous waypoint (or the current position) to the goal."""
        start = self._goal if self._goal is not None else x[:2]
        d = goal - start
        if np.linalg.norm(d) < 1e-6:           # goal coincides with the previous waypoint
            start = x[:2]
            d = goal - start
        self._p0 = np.array(start, dtype=float)
        self._L = float(np.linalg.norm(d))
        self._u = d / self._L if self._L > 1e-9 else np.array([np.cos(x[2]), np.sin(x[2])])
        self._th_r = float(np.arctan2(self._u[1], self._u[0]))
        self._s = float(np.clip((x[:2] - self._p0) @ self._u, 0.0, self._L))
        self._goal = np.array(goal, dtype=float)
        self._throttle = 1.0

    def _scheduled_speed(self):
        """Reference speed at the current progress: cruise speed, slowing down into the waypoint."""
        remaining = self._L - self._s
        if remaining <= 1e-9:
            return 0.0
        return min(self.p["v_ref"], max(self.p["v_min"], self.p["slowdown_gain"] * remaining))

    # ---------------------------------------------------------------- command
    def command(self, x_est, goal):
        """Velocity command (v, w) from the estimated pose [x, y, theta] and the current waypoint."""
        x = np.asarray(x_est, dtype=float)[:3]
        g = np.asarray(goal, dtype=float)[:2]
        p = self.p
        if self._goal is None or not np.array_equal(g, self._goal):
            self._new_segment(x, g)

        # Where the robot sits along the segment, and how far its heading is from the segment direction
        s_proj = float(np.clip((x[:2] - self._p0) @ self._u, 0.0, self._L))
        th_err = float(wrap(self._th_r - x[2]))

        # Switch between rotating in place and tracking, with hysteresis
        if self._align and abs(th_err) < p["align_exit"]:
            self._align = False
        elif not self._align and abs(th_err) > p["align_enter"]:
            self._align = True

        # Move the reference point along the segment (it waits while the robot is turning)
        if self._align:
            self._s = s_proj
            v_r = 0.0
        else:
            v_r = self._scheduled_speed() * self._throttle
            self._s = min(self._L, self._s + v_r * self.dt)
        ref = self._p0 + self._u * self._s

        # Tracking error in the robot frame
        c, s = np.cos(x[2]), np.sin(x[2])
        d = ref - x[:2]
        e = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], th_err])

        # LQR control law
        K = self.gain(v_r)
        if self._align:
            # Turning on the spot cannot remove a lateral offset, so only the heading
            # term is used. Including the lateral term can cancel the heading term and
            # leave the robot stalled short of the target heading.
            v = 0.0
            w = float(np.clip(K[1, 2] * e[2], -p["w_turn_max"], p["w_turn_max"]))
        else:
            ke = K @ e
            v = v_r * np.cos(e[2]) + ke[0]
            w = ke[1]

        # Slow the reference down when the robot lags behind it
        pos_err = float(np.hypot(e[0], e[1]))
        self._throttle = max(0.0, 1.0 - pos_err / p["pos_hold"]) * max(0.0, 1.0 - abs(e[2]) / p["ang_hold"])

        # Apply the speed and acceleration limits (v is kept non-negative)
        v = float(np.clip(v, 0.0, self.v_max))
        w = float(np.clip(w, -self.w_max, self.w_max))
        v, w = self._limit_acceleration(v, w)
        self._u_prev = np.array([v, w])

        self.history.append({"s": self._s, "v_r": v_r, "e": e.copy(), "align": self._align,
                             "cmd": (v, w), "throttle": self._throttle})
        return v, w

    def _limit_acceleration(self, v, w):
        """Limit how much the command can change in one step."""
        a, al = self.p["a_max"], self.p["alpha_max"]
        if a is not None:
            v = float(np.clip(v, self._u_prev[0] - a * self.dt, self._u_prev[0] + a * self.dt))
        if al is not None:
            w = float(np.clip(w, self._u_prev[1] - al * self.dt, self._u_prev[1] + al * self.dt))
        return v, w
