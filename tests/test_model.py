"""Basic checks on the kinematic model and odometry."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.diff_drive import body_to_wheels, motion_model, wheels_to_body
from src.sensing.odometry import WheelOdometry


def test_straight_line():
    x = motion_model(np.array([0.0, 0.0, 0.0]), 1.0, 0.0)
    assert np.allclose(x, [1.0, 0.0, 0.0])


def test_quarter_circle_ends_on_arc():
    # Many small steps along a quarter circle of radius 1 should end near (1, 1, pi/2)
    x = np.zeros(3)
    n = 1000
    for _ in range(n):
        x = motion_model(x, (np.pi / 2) / n, (np.pi / 2) / n)
    assert np.allclose(x, [1.0, 1.0, np.pi / 2], atol=1e-4)


def test_wheel_conversion_roundtrip():
    r, b = 0.033, 0.160
    wl, wr = body_to_wheels(0.15, 0.7, r, b)
    assert np.allclose(wheels_to_body(wl, wr, r, b), (0.15, 0.7))


def test_odometry_spin_in_place():
    r, b, n = 0.033, 0.160, 4096
    odom = WheelOdometry(r, b, n)
    odom.update([0, 0])
    # Right wheel forward, left back, each by the arc for a 90 degree turn
    arc = (np.pi / 2) * b / 2
    ticks = arc / r / (2 * np.pi) * n
    ds, dth = odom.update([-ticks, ticks])
    assert abs(ds) < 1e-9 and np.isclose(dth, np.pi / 2)
