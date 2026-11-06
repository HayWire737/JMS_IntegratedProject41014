"""Part E disturbance hooks in the simulator."""
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.ground_truth import true_pose          # test only: checks the physics
from src.sensing.aruco import ArucoDetector
from src.sim.turtlebot_sim import TurtleBotSim

with open(ROOT / "configs/nominal.yaml") as f:
    CFG = yaml.safe_load(f)


def _cfg(**dist):
    cfg = copy.deepcopy(CFG)
    cfg["disturbances"] = dist
    return cfg


def _drive(cfg, steps=60, v=0.2):
    sim = TurtleBotSim(cfg)
    sim.reset([1.0, 2.0, 0.0])
    ticks = [sim.step(v, 0.0).enc_ticks for _ in range(steps)]
    pose = true_pose(sim)
    sim.close()
    return np.array(ticks), pose


def test_no_disturbances_changes_nothing():
    ticks_a, pose_a = _drive(copy.deepcopy(CFG))
    ticks_b, pose_b = _drive(_cfg())
    assert np.array_equal(ticks_a, ticks_b) and np.allclose(pose_a, pose_b)


def test_occlusion_hides_the_marker_only_inside_the_window():
    cfg = _cfg(occlusion=[[0.1, 0.3]])
    sim = TurtleBotSim(cfg)
    sim.reset([3.5, 2.0, 0.0])
    det = ArucoDetector(cfg["marker"], cfg["camera"])
    seen = {}
    for _ in range(8):
        m = sim.step(0.0, 0.0)
        if m.image is not None:
            seen[round(m.t, 2)] = det.detect(m.image) is not None
    sim.close()
    assert seen[0.2] is False and seen[0.4] is True


def test_push_knocks_the_robot_sideways():
    _, calm = _drive(_cfg(), v=0.15)
    _, pushed = _drive(_cfg(push={"t": 1.0, "duration": 0.3, "force": [0.0, 8.0]}), v=0.15)
    assert abs(pushed[2] - calm[2]) > np.radians(5)          # turned by the push


def test_slip_makes_one_encoder_over_count():
    ticks_ok, pose_ok = _drive(_cfg())
    ticks_slip, pose_slip = _drive(_cfg(slip={"t": 0.5, "duration": 1.0, "wheel": "left", "extra": 0.3}))
    assert ticks_slip[-1, 0] - ticks_ok[-1, 0] > 500          # left wheel reports extra rotation
    assert abs(ticks_slip[-1, 1] - ticks_ok[-1, 1]) < 5       # right wheel unaffected
    assert np.allclose(pose_slip, pose_ok, atol=0.01)         # but the robot itself drove the same