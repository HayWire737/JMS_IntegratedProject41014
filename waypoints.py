"""Sequential waypoint follower."""
import numpy as np


class WaypointFollower:
    def __init__(self, waypoints, tol):
        self.waypoints = [np.array(w, dtype=float) for w in waypoints]
        self.tol = tol
        self.index = 0

    @property
    def done(self):
        return self.index >= len(self.waypoints)

    def current(self, x_est):
        """Current goal, advancing when the ESTIMATED pose is within tolerance."""
        while not self.done and np.hypot(*(self.waypoints[self.index] - x_est[:2])) < self.tol:
            self.index += 1
        return None if self.done else self.waypoints[self.index]
