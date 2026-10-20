"""ArUco marker detection and range/bearing measurement in base_link."""
from dataclasses import dataclass

import cv2
import numpy as np

from src.frames.se2 import camera_point_to_base


def nominal_intrinsics(cam_cfg):
    """Pinhole intrinsics implied by the simulated camera's vertical field of view.

    TODO (Part B): replace with values from a proper calibration
    (e.g. cv2.calibrateCamera on rendered checkerboard images) and compare.
    """
    h, w = cam_cfg["height"], cam_cfg["width"]
    f = (h / 2) / np.tan(np.radians(cam_cfg["fovy_deg"]) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    return K, np.zeros(5)


@dataclass
class MarkerMeasurement:
    marker_id: int
    range: float           # metres, base_link origin to marker centre (planar)
    bearing: float         # rad, angle of marker centre in base_link (left positive)
    p_base: np.ndarray     # marker centre in base_link [x, y, z]
    corners: np.ndarray    # 4x2 pixel corners, for drawing / pixel-noise studies


class ArucoDetector:
    def __init__(self, marker_cfg, cam_cfg, K=None, dist=None):
        self.cam_cfg = cam_cfg
        self.size = marker_cfg["size"]
        self.target_id = marker_cfg["id"]
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker_cfg["dictionary"]))
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        if K is None:
            K, dist = nominal_intrinsics(cam_cfg)
        self.K, self.dist = K, dist
        s = self.size / 2   # marker corners in the marker frame (OpenCV order)
        self.obj_pts = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)

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
            p_base = camera_point_to_base(tvec.ravel(), self.cam_cfg)
            return MarkerMeasurement(int(i), float(np.hypot(p_base[0], p_base[1])),
                                     float(np.arctan2(p_base[1], p_base[0])), p_base, img_pts)
        return None
