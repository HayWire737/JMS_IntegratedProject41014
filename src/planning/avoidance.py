"""LiDAR obstacle avoidance: detour waypoints around obstacles on the way to a goal.

Works from the LiDAR scan and the ESTIMATED pose only, and sits between the
waypoint follower and the controller, so the P controller and the LQR tracker
use it unchanged:

    target = avoid.target(x_est, goal, ranges, angles)
    v, w = controller.command(x_est, target)

Each step:
1. The scan is converted to points in base_link (scan_to_base, which corrects
   for the LiDAR mount and drops max-range returns).
2. The straight path to the goal is checked: it is blocked if any point lies
   inside a corridor as wide as the robot plus a safety margin, up to the
   lookahead distance (or the goal, if nearer).
3. If blocked, candidate headings are tried in order of how far they turn from
   the goal direction (5 deg steps, both sides). The first heading whose own
   corridor is clear for the detour distance becomes a temporary waypoint.
4. The detour is held until the robot reaches it or the direct path to the goal
   clears, so the target does not flicker from step to step. The side it went
   round is remembered, so a tie between left and right does not oscillate.
"""
import numpy as np

from src.frames.se2 import transform_points, wrap
from src.sensing.lidar import scan_to_base


class LidarAvoidance:
    def __init__(self, cfg):
        a = cfg["avoidance"]
        self.lidar_cfg = cfg["lidar"]
        self.r_safe = a["robot_radius"] + a["margin"]   # corridor half-width
        self.lookahead = a["lookahead"]                 # how far ahead obstacles matter
        self.step = a["detour_distance"]                # how far out a detour waypoint goes
        self.reach = a["detour_reached"]                # detour counts as reached within this
        self.max_turn = np.radians(a["max_turn_deg"])   # widest detour tried, each side
        self.detour = None                              # current detour waypoint (world), or None
        self.side = 0                                   # +1 left, -1 right, 0 none yet
        self.switched = False                           # target changed this step
        self.detours = 0                                # count, for logging

    def blocked(self, pts, heading, length):
        """True if any point lies in the corridor from the robot along `heading` for `length`."""
        if len(pts) == 0:
            return False
        c, s = np.cos(heading), np.sin(heading)
        along = pts[:, 0] * c + pts[:, 1] * s
        across = np.abs(-pts[:, 0] * s + pts[:, 1] * c)
        return bool(np.any((along > 0) & (along < length) & (across < self.r_safe)))

    def target(self, x_est, goal, ranges, angles):
        """Where the controller should head: the goal itself, or a detour around an obstacle."""
        self.switched = False
        pts = scan_to_base(ranges, angles, self.lidar_cfg)
        goal_base = transform_points([0.0, 0.0, -x_est[2]], np.asarray(goal) - x_est[:2])[0]
        dist = float(np.hypot(*goal_base))
        bearing = float(np.arctan2(goal_base[1], goal_base[0]))
        direct_blocked = self.blocked(pts, bearing, min(dist, self.lookahead))

        if self.detour is not None:
            reached = np.hypot(*(self.detour - x_est[:2])) < self.reach
            if not reached and direct_blocked:
                return self.detour
            self.detour, self.switched = None, True     # reached it, or the way is clear again

        if not direct_blocked:
            return goal

        # Try headings in order of how far they turn from the goal, preferring the side used last
        offsets = np.radians(np.arange(5, np.degrees(self.max_turn) + 1, 5))
        first = self.side if self.side else 1
        for off in offsets:
            for sgn in (first, -first):
                h = bearing + sgn * off
                if not self.blocked(pts, h, self.step + self.r_safe):
                    self.side = sgn
                    local = np.array([self.step * np.cos(h), self.step * np.sin(h)])
                    self.detour = transform_points(x_est, local)[0]
                    self.switched = True
                    self.detours += 1
                    return self.detour
        return goal       # boxed in: keep the goal and let the LiDAR safety stop hold the robot