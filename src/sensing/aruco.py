"""ArUco marker detection and range/bearing measurement in base_link."""
from dataclasses import dataclass

import cv2
import numpy as np

from src.frames.se2 import camera_point_to_base


def nominal_intrinsics(cam_cfg):
    """Ideal pinhole intrinsics implied by the simulated camera's vertical field of view.

    OpenCV puts pixel CENTRES at integer coordinates, so the centre of a W x H
    image is ((W - 1) / 2, (H - 1) / 2), not (W / 2, H / 2). Using W / 2 shifts
    every bearing by half a pixel (0.07 deg here).
    """
    h, w = cam_cfg["height"], cam_cfg["width"]
    f = (h / 2) / np.tan(np.radians(cam_cfg["fovy_deg"]) / 2)
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    return K, np.zeros(5)


def camera_intrinsics(cam_cfg):
    """Calibrated intrinsics from the config (camera.intrinsics), or nominal if absent.

    The calibrated values come from scripts/calibrate_camera.py.
    """
    cal = cam_cfg.get("intrinsics")
    if cal is None:
        return nominal_intrinsics(cam_cfg)
    K = np.array([[cal["fx"], 0, cal["cx"]], [0, cal["fy"], cal["cy"]], [0, 0, 1]], dtype=float)
    return K, np.array(cal.get("dist", [0, 0, 0, 0, 0]), dtype=float)


@dataclass
class MarkerMeasurement:
    marker_id: int
    range: float # metres, base_link origin to marker centre (planar)
    bearing: float # rad, angle of marker centre in base_link (left positive)
    p_base: np.ndarray # marker centre in base_link [x, y, z]
    yaw_base: float # rad, direction of the marker face normal in base_link
    corners: np.ndarray # 4x2 pixel corners, for drawing / pixel-noise studies
    t_cam: np.ndarray = None # marker centre in the camera frame (after range correction)

    @property
    def pose(self):
        """Planar marker pose in base_link: [x, y, yaw]. Used by the pose EKF update."""
        return np.array([self.p_base[0], self.p_base[1], self.yaw_base])


class ArucoDetector:
    def __init__(self, marker_cfg, cam_cfg, K=None, dist=None):
        self.cam_cfg = cam_cfg
        self.size = marker_cfg["size"]
        self.target_id = marker_cfg["id"]
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker_cfg["dictionary"]))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # sub-pixel corners
        params.minDistanceToBorder = 0
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        if K is None:
            K, dist = camera_intrinsics(cam_cfg)
        self.K, self.dist = K, dist
        # Range bias correction, bias(d) = a2 d^2 + a3 d^3 (scripts/calibrate_marker_range.py)
        self.range_correction = marker_cfg.get("range_correction")
        s = self.size / 2   # marker corners in the marker frame (OpenCV order)
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

    def correct_range(self, t_cam):
        """Remove the calibrated range bias by shortening the camera-to-marker vector.

        The outer corners are found slightly inside the marker, so it looks smaller
        and therefore further away; the error is along the viewing ray, so the
        whole vector is scaled. Directions (and so bearings) are unchanged.
        """
        if not self.range_correction:
            return t_cam
        a2, a3 = self.range_correction
        d = np.linalg.norm(t_cam)
        return t_cam * (1.0 - a2 * d - a3 * d ** 2)       # d - bias(d), as a scale factor

    def detect(self, image_rgb):
        """Return a MarkerMeasurement for the target marker, or None if not seen."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        for c, i in zip(corners, ids.ravel()):
            if i != self.target_id:
                continue
            img_pts = c.reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(self.obj_pts, img_pts, self.K, self.dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                return None
            t_cam = self.correct_range(tvec.ravel())
            p_base = camera_point_to_base(t_cam, self.cam_cfg)
            # Marker z axis (out of the face, towards the camera) in the camera frame,
            # rotated into base_link: base = (z_cam, -x_cam, -y_cam)
            n_cam = cv2.Rodrigues(rvec)[0][:, 2]
            yaw = float(np.arctan2(-n_cam[0], n_cam[2]))
            return MarkerMeasurement(int(i), float(np.hypot(p_base[0], p_base[1])),
                                     float(np.arctan2(p_base[1], p_base[0])), p_base, yaw, img_pts, t_cam)
        return None
