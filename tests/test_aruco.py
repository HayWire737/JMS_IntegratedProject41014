"""ArUco detection tests (Part B)."""
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.frames.se2 import wrap
from src.sim.turtlebot_sim import TurtleBotSim
from src.sensing.aruco import ArucoDetector, camera_intrinsics, nominal_intrinsics

with open(ROOT / "configs/nominal.yaml") as f:
    CFG = yaml.safe_load(f)


def _pose(rng, bearing):
    """Robot straight in front of the marker at `rng`, turned so the marker sits at `bearing`."""
    mx, my, _ = CFG["marker"]["pose_world"]
    x, y = mx - rng, my
    return [x, y, wrap(0.0 - bearing)]


def test_close_marker_detected_until_it_leaves_the_image():
    """At docking range the marker must be detected at every bearing until it actually
    leaves the image. With OpenCV's default minDistanceToBorder (3 px) it dropped out
    at isolated bearings (e.g. 11 deg at 0.3 m) while fully visible."""
    cfg = copy.deepcopy(CFG)
    cfg["noise"]["camera_pixel_sigma"] = 0.0
    sim = TurtleBotSim(cfg)
    det = ArucoDetector(cfg["marker"], cfg["camera"])
    seen = []
    for b in range(0, 15):
        sim.reset(_pose(0.3, np.radians(b)))
        seen.append(det.detect(sim.render_camera()) is not None)
    sim.close()
    missed = [b for b, s in zip(range(0, 15), seen) if not s]
    assert not missed, f"marker missed at bearings {missed} deg while fully in view"



def test_principal_point_uses_pixel_centre_convention():
    K, _ = nominal_intrinsics(CFG["camera"])
    assert K[0, 2] == (CFG["camera"]["width"] - 1) / 2
    assert K[1, 2] == (CFG["camera"]["height"] - 1) / 2


def test_detector_uses_calibrated_intrinsics():
    K_cal, _ = camera_intrinsics(CFG["camera"])
    det = ArucoDetector(CFG["marker"], CFG["camera"])
    assert np.allclose(det.K, K_cal)


def test_marker_dead_ahead_has_no_bearing_bias():
    """With W/2 as the image centre every bearing was offset by half a pixel (0.069 deg)."""
    cfg = copy.deepcopy(CFG)
    cfg["noise"]["camera_pixel_sigma"] = 0.0
    sim = TurtleBotSim(cfg)
    det = ArucoDetector(cfg["marker"], cfg["camera"])
    bearings = []
    for rng in (0.6, 1.0, 1.5, 2.0):
        sim.reset(_pose(rng, 0.0))
        bearings.append(det.detect(sim.render_camera()).bearing)
    sim.close()
    # 0.05 deg = 0.36 px: allows for the calibration's own uncertainty (about 0.3 px in cx)
    # while still failing the half-pixel bug (0.069 deg).
    assert np.degrees(np.max(np.abs(bearings))) < 0.05, f"bearings {np.degrees(bearings)} deg"