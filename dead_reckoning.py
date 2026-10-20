"""Baseline estimator: integrate wheel odometry only (no correction).

This drifts without bound. It exists so the pipeline runs end to end today,
and as the comparison baseline for the EKF.
"""
import numpy as np

from src.model.diff_drive import motion_model


class DeadReckoning:
    def __init__(self, x0):
        self.x = np.array(x0, dtype=float)

    def predict(self, ds, dth):
        self.x = motion_model(self.x, ds, dth)
        return self.x

    def update(self, measurement):
        """Dead reckoning ignores exteroceptive measurements."""
        return self.x
