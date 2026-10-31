"""Simulator noise tests (simulation environment).

Each sensor's noise must come from its own random stream. If the camera drew
from the same generator as the encoders, then changing anything camera-related
(pixel noise, blur, frame rate) or even the controller (which changes how often
frames are rendered) would silently change the encoder noise too, and
comparisons between experiments would be confounded.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim.turtlebot_sim import TurtleBotSim


def _cfg(**noise):
    with open(ROOT / "configs/nominal.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["noise"].update(noise)
    return cfg


def _drive(cfg, steps=40):
    """Drive a gentle arc; return encoder ticks and LiDAR ranges at every step."""
    sim = TurtleBotSim(cfg)
    ticks, ranges = [], []
    for _ in range(steps):
        m = sim.step(0.15, 0.3)
        ticks.append(m.enc_ticks)
        ranges.append(m.lidar_ranges)
    sim.close()
    return np.array(ticks), np.array(ranges)


def test_camera_noise_does_not_change_other_sensors():
    ticks_off, lidar_off = _drive(_cfg(camera_pixel_sigma=0.0, camera_blur_px=0))
    ticks_on, lidar_on = _drive(_cfg(camera_pixel_sigma=8.0, camera_blur_px=1))
    assert np.array_equal(ticks_off, ticks_on), "camera noise changed the encoder readings"
    assert np.array_equal(lidar_off, lidar_on), "camera noise changed the LiDAR readings"


def test_camera_noise_is_reproducible_and_present():
    cfg = _cfg(camera_pixel_sigma=8.0)
    frames = []
    for _ in range(2):                      # same seed twice -> identical first frame
        sim = TurtleBotSim(cfg)
        frames.append(sim.render_camera().astype(int))
        frames.append(sim.render_camera().astype(int))
        sim.close()
    assert np.array_equal(frames[0], frames[2]), "same seed gave different images"
    assert np.abs(frames[0] - frames[1]).mean() > 1.0, "consecutive frames identical: no noise"