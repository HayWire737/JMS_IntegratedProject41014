"""EKF fusing wheel odometry with ArUco range/bearing.  TODO (Part C, estimation lead).

Keep the same interface as DeadReckoning so run.py can swap between them:

    ekf = EKF(x0, P0, cfg)
    ekf.predict(ds, dth)       # odometry step: x = f(x, u), P = F P F^T + G Q G^T
    ekf.update(measurement)    # MarkerMeasurement or None (no update when the marker is not seen)
    ekf.x, ekf.P               # estimate and covariance, used by the controller and the video plots

Design decisions to make and defend on the project page:
  - Q from encoder resolution / slip characterisation
  - R from repeated static marker measurements at several distances
  - measurement model h(x) = [range, bearing] to the known marker pose in config['marker']
  - innovation gating, and NEES/NIS consistency checks
"""
import numpy as np


class EKF:
    def __init__(self, x0, P0, cfg):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.cfg = cfg

    def predict(self, ds, dth):
        raise NotImplementedError

    def update(self, measurement):
        raise NotImplementedError
