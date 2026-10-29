"""GROUND TRUTH ACCESS — EVALUATION ONLY.

This is the only module in the repository that reads the robot's true pose
from the simulator. Nothing under src/ may import it (enforced by
tests/test_no_ground_truth.py). Scripts use it only to log and score runs.
"""
import numpy as np


def true_pose(sim):
    """True base_link pose [x, y, yaw] in the world frame."""
    q = sim._data.qpos   # freejoint: [x, y, z, qw, qx, qy, qz]
    yaw = np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
    return np.array([q[0], q[1], yaw])


class RunLogger:
    """Collects estimate, truth and commands each step for metrics and plots."""

    def __init__(self):
        self.rows = {"t": [], "est": [], "truth": [], "cmd": [], "marker_seen": [], "P": []}

    def log(self, t, x_est, x_true, cmd, marker_seen, P=None):
        self.rows["P"].append(np.full((3, 3), np.nan) if P is None else np.array(P, dtype=float))
        self.rows["t"].append(t)
        self.rows["est"].append(np.array(x_est, dtype=float))
        self.rows["truth"].append(np.array(x_true, dtype=float))
        self.rows["cmd"].append(np.array(cmd, dtype=float))
        self.rows["marker_seen"].append(bool(marker_seen))

    def arrays(self):
        return {k: np.array(v) for k, v in self.rows.items()}

    def save(self, path):
        np.savez(path, **self.arrays())
