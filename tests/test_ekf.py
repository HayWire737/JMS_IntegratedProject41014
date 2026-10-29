"""Unit tests for the EKF (Part C)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.estimation.dead_reckoning import DeadReckoning
from src.estimation.ekf import EKF
from tests.test_model import numerical_jacobian

CFG = yaml.safe_load(open(ROOT / "configs/nominal.yaml"))


def fake_measurement(z_pose, rng=1.0):
    """MarkerMeasurement stand-in with the fields the EKF reads."""
    z = np.asarray(z_pose, dtype=float)
    return SimpleNamespace(pose=z, range=rng, bearing=float(np.arctan2(z[1], z[0])))


def test_pose_measurement_jacobian():
    f = EKF([0, 0, 0], CFG)
    rng = np.random.default_rng(3)
    for _ in range(20):
        x = rng.uniform([0, 0, -np.pi], [4, 4, np.pi])
        H = f.measurement_model(x)[1]
        Hn = numerical_jacobian(lambda s: f.measurement_model(s)[0], x)
        assert np.allclose(H, Hn, atol=1e-4)


def test_predict_only_matches_dead_reckoning_and_grows_covariance():
    f, dr = EKF([1, 1, 0.3], CFG), DeadReckoning([1, 1, 0.3])
    traces = [np.trace(f.P)]
    for k in range(100):
        ds, dth = 0.01, 0.02 * np.sin(k / 10)
        f.predict(ds, dth)
        dr.predict(ds, dth)
        traces.append(np.trace(f.P))
    assert np.allclose(f.x, dr.x)
    assert np.all(np.diff(traces) > 0)          # no information -> uncertainty only grows


def test_update_moves_towards_truth_and_shrinks_covariance():
    # Close to the marker (0.85 m) the yaw measurement is accurate enough to separate a
    # sideways offset from a heading error; at long range that direction is only weakly
    # observable (see docs/part_A_C_notes.md), so convergence there is slow.
    truth = np.array([4.0, 2.0, 0.1])
    f = EKF(truth + [0.1, -0.08, 0.05], CFG)
    f.P = np.diag([0.1, 0.1, 0.05]) ** 2
    z = f.measurement_model(truth)[0]            # perfect measurement from the true pose
    P0, e0 = f.P.copy(), np.linalg.norm(f.x[:2] - truth[:2])
    f.update(fake_measurement(z, rng=0.85))
    assert np.linalg.norm(f.x[:2] - truth[:2]) < 0.7 * e0       # one update: closer
    assert np.all(np.diag(f.P) < np.diag(P0))                    # and more certain
    for _ in range(20):
        f.update(fake_measurement(z, rng=0.85))
    assert np.linalg.norm(f.x[:2] - truth[:2]) < 0.01            # repeated updates converge
    assert np.allclose(f.P, f.P.T)


def test_gate_rejects_outlier():
    f = EKF([3.0, 2.0, 0.0], CFG)
    z = f.measurement_model(f.x)[0] + [0.5, 0.5, 0.0]   # 70 cm off: an outlier
    x0 = f.x.copy()
    f.update(fake_measurement(z, rng=0.85))
    assert np.allclose(f.x, x0) and not f.history[-1]["accepted"]


def test_measurement_beyond_calibrated_range_ignored():
    f = EKF([0.5, 0.5, 0.0], CFG)
    x0, P0 = f.x.copy(), f.P.copy()
    f.update(fake_measurement([4.0, 1.0, 3.0], rng=4.5))
    assert np.allclose(f.x, x0) and np.allclose(f.P, P0) and not f.history
