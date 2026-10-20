"""2D LiDAR helpers."""
import numpy as np

from src.frames.se2 import wrap


def to_points(ranges, angles):
    """Valid returns as an (N, 2) array of points in the lidar frame."""
    ok = np.isfinite(ranges)
    return np.column_stack([ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])])


def min_range_in_sector(ranges, angles, centre, half_width):
    """Closest return within [centre - half_width, centre + half_width]."""
    sel = np.abs(wrap(angles - centre)) <= half_width
    return float(np.min(ranges[sel])) if sel.any() else np.inf

# TODO (planning lead): obstacle avoidance from LiDAR, e.g. vector field
# histogram or a local potential field, replacing hand-placed waypoints.
