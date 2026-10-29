"""Extended Kalman filter: wheel odometry + ArUco marker measurements (Part C).

State   x = [x, y, theta] (base_link in world),  covariance P (3x3)

Prediction (every control step, from encoder increments)
    x = f(x, ds, dth)                        motion_model, midpoint rule
    P = F P F^T + G Q G^T                    F, G from src.model.diff_drive.jacobians
    Q = cov([ds, dth]) built from per-wheel noise: var(d_wheel) = q0 + k |d_wheel|

Update (whenever the marker is detected), two measurement models:
    "pose":          z = marker pose in base_link [x_m, y_m, yaw_m]   (3 measurements)
    "range_bearing": z = [range, bearing] to the marker centre          (2 measurements)
    y = z - h(x)  (angles wrapped),  S = H P H^T + R,  K = P H^T S^-1
    x = x + K y,  P = (I - K H) P (I - K H)^T + K R K^T   (Joseph form, stays symmetric PSD)
    R depends on range: sigma_i(r) = a_i + b_i r^2, fitted by scripts/calibrate_marker_noise.py
    Innovations with NIS = y^T S^-1 y above the chi-square gate are rejected as outliers.
    Measurements beyond the calibrated range are ignored (the R model is not valid there).
    Divergence recovery: after N consecutive rejections the innovations are consistent
    with each other but not with the estimate, so the estimate (not the sensor) is
    assumed wrong: P is inflated and the measurement is accepted.

Why the pose model is the default: with range/bearing to a single point the
pose is NOT observable (rotating the robot about the marker leaves range and
bearing unchanged), see scripts/analyse_model.py and docs/part_A_C_notes.md.
"""
import numpy as np
from scipy.stats import chi2

from src.frames.se2 import wrap
from src.model.diff_drive import jacobians, motion_model, range_bearing, range_bearing_jacobian


class EKF:
    def __init__(self, x0, cfg):
        e = cfg["ekf"]
        self.x = np.array(x0, dtype=float)
        self.P = np.diag(np.square(e["initial_sigma"]))
        self.b = cfg["robot"]["wheel_base"]
        self.q0 = e["odom_q0"]
        self.k = e["odom_k"]
        self.model = e["measurement_model"]
        self.r_coeffs = {k: np.array(v) for k, v in e["marker_sigma"].items()}
        self.r_scale = e.get("R_scale", 1.0)
        self.landmark = np.array(cfg["marker"]["pose_world"], dtype=float)
        dof = 3 if self.model == "pose" else 2
        self.gate = chi2.ppf(e["gate_probability"], dof) if e.get("gate_probability") else np.inf
        self.max_range = e.get("max_marker_range", np.inf)
        self.reset_after = e.get("reset_after_rejections", 0)
        self.inflate = e.get("reset_inflation", 100.0)
        self._rejections = 0
        self.history = []            # per update: dict(t, nis, accepted, dof, reset)
        self.t = 0.0

    # ----------------------------------------------------------- prediction
    def odometry_covariance(self, ds, dth):
        """Q for [ds, dth], from independent per-wheel distance noise."""
        d_left = ds - dth * self.b / 2
        d_right = ds + dth * self.b / 2
        var_l = self.q0 + self.k * abs(d_left)
        var_r = self.q0 + self.k * abs(d_right)
        # ds = (d_r + d_l)/2, dth = (d_r - d_l)/b
        J = np.array([[0.5, 0.5], [-1 / self.b, 1 / self.b]])
        return J @ np.diag([var_l, var_r]) @ J.T

    def predict(self, ds, dth, dt=0.0):
        F, G = jacobians(self.x, ds, dth)
        Q = self.odometry_covariance(ds, dth)
        self.x = motion_model(self.x, ds, dth)
        self.P = F @ self.P @ F.T + G @ Q @ G.T
        self.t += dt
        return self.x

    # --------------------------------------------------------------- update
    def measurement_model(self, x):
        """Predicted measurement h(x) and Jacobian H."""
        L = self.landmark
        if self.model == "range_bearing":
            return range_bearing(x, L), range_bearing_jacobian(x, L)
        dx, dy = L[0] - x[0], L[1] - x[1]
        c, s = np.cos(x[2]), np.sin(x[2])
        hx = c * dx + s * dy             # marker position in base_link
        hy = -s * dx + c * dy
        h = np.array([hx, hy, wrap(L[2] - x[2])])
        H = np.array([[-c, -s, hy],
                      [s, -c, -hx],
                      [0.0, 0.0, -1.0]])
        return h, H

    def measurement_noise(self, rng):
        """R as a function of range, from the calibration fit sigma = a + b r^2."""
        keys = ["range", "bearing"] if self.model == "range_bearing" else ["x", "y", "yaw"]
        sig = [self.r_coeffs[k][0] + self.r_coeffs[k][1] * rng ** 2 for k in keys]
        return self.r_scale * np.diag(np.square(sig))

    def update(self, meas):
        """Apply a MarkerMeasurement, or do nothing if meas is None."""
        if meas is None or meas.range > self.max_range:
            return self.x
        if self.model == "range_bearing":
            z = np.array([meas.range, meas.bearing])
            ang = [1]
        else:
            z = meas.pose
            ang = [2]
        h, H = self.measurement_model(self.x)
        y = z - h
        y[ang] = wrap(y[ang])
        R = self.measurement_noise(meas.range)
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)
        accepted = nis <= self.gate
        reset = False
        if not accepted:
            self._rejections += 1
            if self.reset_after and self._rejections >= self.reset_after:
                self.P = self.P * self.inflate           # divergence recovery
                S = H @ self.P @ H.T + R
                S_inv = np.linalg.inv(S)
                accepted = reset = True
        self.history.append({"t": self.t, "nis": nis, "accepted": accepted, "dof": len(z), "reset": reset})
        if not accepted:
            return self.x
        self._rejections = 0
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return self.x
