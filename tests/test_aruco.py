"""ArUco detection tests (Part B)."""
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.frames.se2 import wrap
from src.sensing.aruco import ArucoDetector
from src.sim.turtlebot_sim import TurtleBotSim

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