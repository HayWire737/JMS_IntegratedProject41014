"""Wheel-encoder odometry: integer ticks -> (distance, heading change) per step."""
import numpy as np


class WheelOdometry:
    def __init__(self, wheel_radius, wheel_base, ticks_per_rev):
        self.r = wheel_radius
        self.b = wheel_base
        self.rad_per_tick = 2 * np.pi / ticks_per_rev
        self._last = None

    def update(self, ticks):
        """Return (ds, dth) since the previous call. The first call returns zeros."""
        ticks = np.asarray(ticks)
        if self._last is None:
            self._last = ticks
            return 0.0, 0.0
        d_left, d_right = (ticks - self._last) * self.rad_per_tick * self.r
        self._last = ticks
        ds = (d_right + d_left) / 2
        dth = (d_right - d_left) / self.b
        return float(ds), float(dth)
