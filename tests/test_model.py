"""Checks on the kinematic model, its Jacobians and the structural analysis (Part A)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.diff_drive import (body_to_wheels, controllability_matrix, error_dynamics, jacobians,
                                  motion_model, range_bearing, range_bearing_jacobian, rank,
                                  wheels_to_body)
from src.sensing.odometry import WheelOdometry


def numerical_jacobian(f, x, eps=1e-6):
    fx = f(x)
    J = np.zeros((len(fx), len(x)))
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = eps
        d = f(x + dx) - fx
        if len(fx) == 3:
            d[2] = (d[2] + np.pi) % (2 * np.pi) - np.pi
        J[:, i] = d / eps
    return J


def test_straight_line():
    assert np.allclose(motion_model(np.zeros(3), 1.0, 0.0), [1.0, 0.0, 0.0])


def test_quarter_circle_ends_on_arc():
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
    ticks = (np.pi / 2) * b / 2 / r / (2 * np.pi) * n
    ds, dth = odom.update([-ticks, ticks])
    assert abs(ds) < 1e-9 and np.isclose(dth, np.pi / 2)


def test_motion_jacobians_match_finite_differences():
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.uniform([-2, -2, -np.pi], [2, 2, np.pi])
        ds, dth = rng.uniform(-0.05, 0.05), rng.uniform(-0.3, 0.3)
        F, G = jacobians(x, ds, dth)
        assert np.allclose(F, numerical_jacobian(lambda s: motion_model(s, ds, dth), x), atol=1e-5)
        Gn = numerical_jacobian(lambda u: motion_model(x, u[0], u[1]), np.array([ds, dth]))
        assert np.allclose(G, Gn, atol=1e-5)


def test_range_bearing_jacobian_matches_finite_differences():
    rng = np.random.default_rng(2)
    lm = np.array([4.84, 2.0])
    for _ in range(20):
        x = rng.uniform([0, 0, -np.pi], [4, 4, np.pi])
        Hn = numerical_jacobian(lambda s: range_bearing(s, lm), x)
        assert np.allclose(range_bearing_jacobian(x, lm), Hn, atol=1e-4)


def test_controllable_when_moving_not_at_rest():
    A, B = error_dynamics(0.2, 0.0)
    assert rank(controllability_matrix(A, B)) == 3
    A, B = error_dynamics(0.0, 0.0)
    assert rank(controllability_matrix(A, B)) == 2      # cannot move sideways
