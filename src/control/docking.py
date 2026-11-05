"""Visual docking to the ArUco-marked charger. Works from marker detections and wheel odometry only.

Takes over once the waypoint follower has finished:

    dock = VisualDocking(cfg["docking"], cfg["controller"], cfg["robot"], dt=sim.dt)
    dock.observe(ds, dth, marker_or_None)     # every control step, from the start of the run
    v, w = dock.command()                     # only after the last waypoint

How it works
------------
1. Relative pose. A detection gives the marker pose in base_link. Inverting it
   gives the robot pose q = [x, y, theta] in the marker frame (x out of the
   marker face, so a robot facing the marker has theta = pi). Between camera
   frames q is carried forward with the wheel-odometry increments. The marker
   frame is fixed in the world, so q follows the same unicycle model as the
   world pose. Neither the EKF estimate nor the marker's map pose is used.
2. Approach. A separate LQRTracker (settings from controller.lqr, overridden by
   docking.lqr) tracks the marker normal: the segment runs from the robot's
   projection on the normal to the stop point (stop_distance, 0). Ending on the
   normal, facing the marker, puts the marker at the image centre.
3. Decision. The stage is decided on fresh detections only. Once the measured
   range is within range_tol of stop_distance (or closer), the robot is docked
   if the marker is within bearing_tol_deg of the image centre, otherwise the
   attempt has failed (the controller does not reverse). When the odometry
   says the stop point has been reached, the robot holds still until the next
   frame confirms it.
4. Marker loss. With no detection for lost_timeout the robot stops and rotates
   in place towards the last known marker bearing. Docking fails after
   max_time.
"""
import numpy as np

from src.control.lqr import LQRTracker
from src.frames.se2 import inverse, wrap
from src.model.diff_drive import motion_model

DEFAULTS = {
    "stop_distance": 0.30,     # target range, base_link origin to marker centre [m]
    "range_tol": 0.01,         # stop once a fresh range is within this of stop_distance [m]
    "bearing_tol_deg": 10.0,   # docked if the marker is this close to the image centre [deg]
    "lost_timeout": 2.0,       # time without a detection before searching [s]
    "search_delay": 0.5,       # wait this long for a first frame before searching [s]
    "search_rate": 0.4,        # in-place rotation rate while searching [rad/s]
    "max_time": 30.0,          # docking gives up after this [s]
    "lqr": {},                 # overrides of controller.lqr for the approach
}

SEARCH, APPROACH, DOCKED, FAILED = "search", "approach", "docked", "failed"


def image_bearing(meas):
    """Horizontal angle of the marker centre from the optical axis, left positive [rad]."""
    t = meas.t_cam
    return float(np.arctan2(-t[0], t[2]))


def robot_in_marker_frame(meas):
    """Robot pose [x, y, theta] in the marker frame from one detection."""
    return inverse(meas.pose)


class VisualDocking:
    def __init__(self, dock_cfg, ctrl_cfg, robot_cfg, dt=0.05):
        self.p = {**DEFAULTS, **(dock_cfg or {})}
        self.dt = float(dt)
        lqr_cfg = {**(ctrl_cfg.get("lqr") or {}), **(self.p["lqr"] or {})}
        self.lqr = LQRTracker({"lqr": lqr_cfg}, robot_cfg, dt=dt)
        self.stop = float(self.p["stop_distance"])
        self.goal = np.array([self.stop, 0.0])
        self.q = None              # robot pose in the marker frame
        self.age = np.inf          # time since the last detection [s]
        self.fresh = None          # detection from this step, or None
        self.last_range = np.nan   # last measured range [m]
        self.last_bearing = np.nan # last measured image bearing [rad]
        self.state = None          # None until command() is first called
        self.t = 0.0               # time spent docking [s]
        self.reason = ""           # why docking failed

    @property
    def finished(self):
        return self.state in (DOCKED, FAILED)

    @property
    def docked(self):
        return self.state == DOCKED

    # ------------------------------------------------------------ perception
    def observe(self, ds, dth, meas):
        """Update the relative pose from a detection, or carry it forward with odometry."""
        self.fresh = meas
        if meas is not None:
            self.q = robot_in_marker_frame(meas)
            self.age = 0.0
            self.last_range = float(meas.range)
            self.last_bearing = image_bearing(meas)
        else:
            if self.q is not None:
                self.q = motion_model(self.q, ds, dth)
            self.age += self.dt

    def marker_bearing(self):
        """Bearing of the marker from base_link according to the current relative pose [rad]."""
        x, y, th = self.q
        return float(wrap(np.arctan2(-y, -x) - th))

    # --------------------------------------------------------------- control
    def command(self):
        """Velocity command (v, w) for this step."""
        if self.state is None:
            self.state = APPROACH if self.age <= self.p["lost_timeout"] else SEARCH
            if self.state == APPROACH:
                self._start_approach()
        if self.finished:
            return 0.0, 0.0
        self.t += self.dt
        if self.t > self.p["max_time"]:
            return self._fail("timed out")

        if self.state == SEARCH:
            if self.fresh is not None:
                self._start_approach()
            elif self.t < self.p["search_delay"] and self.q is None:
                return 0.0, 0.0
            else:
                side = 1.0 if self.q is None or self.marker_bearing() >= 0 else -1.0
                return 0.0, side * self.p["search_rate"]

        if self.age > self.p["lost_timeout"]:
            self.state = SEARCH
            self.lqr.reset()
            return 0.0, 0.0

        if self.fresh is not None and self.fresh.range <= self.stop + self.p["range_tol"]:
            if abs(image_bearing(self.fresh)) <= np.radians(self.p["bearing_tol_deg"]):
                self.state = DOCKED
                return 0.0, 0.0
            return self._fail("marker off centre at the stop distance")

        if np.hypot(*self.q[:2]) <= self.stop:      # odometry says arrived: wait for a frame
            return 0.0, 0.0
        return self.lqr.command(self.q, self.goal)

    def _start_approach(self):
        """New segment along the marker normal, from the robot's projection on it to the stop point."""
        self.state = APPROACH
        self.lqr.reset()
        start_x = max(float(self.q[0]), self.stop + 0.05)
        self.lqr.start_from([start_x, 0.0])

    def _fail(self, reason):
        self.state = FAILED
        self.reason = reason
        return 0.0, 0.0
