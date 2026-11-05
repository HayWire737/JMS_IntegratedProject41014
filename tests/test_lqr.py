"""Unit tests for the LQR tracker. Pure kinematics, no MuJoCo, so they run in seconds."""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.control.lqr import LQRTracker
from src.model.diff_drive import controllability_matrix, discretise, error_dynamics, rank
from src.planning.waypoints import WaypointFollower

CFG = yaml.safe_load(open(ROOT / "configs/nominal.yaml"))
CTL, ROB = CFG["controller"], CFG["robot"]
DT = CFG["sim"]["control_dt"]


def make(**lqr_overrides):
    ctl = {**CTL, "lqr": {**CTL["lqr"], **lqr_overrides}}
    return LQRTracker(ctl, ROB, dt=DT)


def unicycle(x, v, w, dt=DT):
    """Exact-arc kinematic step (midpoint heading)."""
    m = x[2] + 0.5 * w * dt
    return np.array([x[0] + v * dt * np.cos(m), x[1] + v * dt * np.sin(m), x[2] + w * dt])


# ------------------------------------------------------------------ gains
def test_closed_loop_is_stable_at_every_reference_speed():
    ctrl = make()
    for v in (0.0, 0.03, 0.05, 0.15, 0.22):
        poles = ctrl.closed_loop_poles(v)
        assert np.all(np.abs(poles) < 1.0), f"unstable at v_r={v}: {poles}"


def test_gain_signs_match_the_error_convention():
    """e_x > 0 (reference ahead) must speed up; e_theta > 0 (reference to the left) must turn left."""
    K = make().gain(0.15)
    assert K.shape == (2, 3)
    assert K[0, 0] > 0 and K[1, 2] > 0


def test_floor_speed_keeps_the_gain_defined_when_the_reference_stops():
    """At v_r = 0 the lateral error is uncontrollable (rank 2), so the gain is computed at v_lin_min instead."""
    A, B = error_dynamics(0.0, 0.0)
    assert rank(controllability_matrix(*discretise(A, B, DT))) == 2
    ctrl = make()
    K0, Kf = ctrl.gain(0.0), ctrl.gain(ctrl.p["v_lin_min"])
    assert np.all(np.isfinite(K0)) and np.allclose(K0, Kf)


# ------------------------------------------------------------ one-step behaviour
def test_on_the_line_it_cruises_straight():
    ctrl = make()
    x, goal = np.array([0.5, 0.5, 0.0]), np.array([2.5, 0.5])
    for _ in range(30):
        v, w = ctrl.command(x, goal)
        x = unicycle(x, v, w)
    assert abs(w) < 1e-9 and abs(x[1] - 0.5) < 1e-9
    assert abs(v - ctrl.p["v_ref"]) < 0.01


def test_steers_back_toward_the_line_from_either_side():
    for offset, sign in ((+0.08, -1), (-0.08, +1)):
        ctrl = make()
        goal = np.array([3.0, 0.0])
        ctrl.command(np.array([0.0, 0.0, 0.0]), goal)               # defines the segment along +x
        _, w = ctrl.command(np.array([0.3, offset, 0.0]), goal)
        assert np.sign(w) == sign, f"offset {offset}: w={w}"


def test_large_heading_error_rotates_in_place_with_hysteresis():
    ctrl = make()
    goal = np.array([0.0, 2.0])                                     # segment points along +y
    v, w = ctrl.command(np.array([0.0, 0.0, 0.0]), goal)            # 90 degrees off
    assert v == 0.0 and w > 0
    v, _ = ctrl.command(np.array([0.0, 0.0, np.pi / 2 - 0.3]), goal)    # 0.3 rad off: still turning
    assert v == 0.0
    v, _ = ctrl.command(np.array([0.0, 0.0, np.pi / 2 - 0.10]), goal)  # inside align_exit: tracking again
    assert v > 0.0
    ctrl2 = make()
    v, w = ctrl2.command(np.array([0.0, 0.0, 0.0]), np.array([0.0, -2.0]))
    assert v == 0.0 and w < 0


def test_rotation_in_place_is_not_blocked_by_a_lateral_offset():
    """While turning in place, a lateral offset must not cancel the heading correction and stall the turn."""
    ctrl = make()
    goal = np.array([2.7, 2.0])
    v, w = ctrl.command(np.array([2.6, 0.6, 0.0]), goal)            # ~86 degrees off: enters the turn
    assert v == 0.0 and w > 0
    v, w = ctrl.command(np.array([2.53, 0.60, 1.32]), goal)         # 7 cm to the side, 0.18 rad short
    assert v == 0.0 and w > 0.2                                     # still turning, not stalled


def test_commands_respect_every_limit():
    rng = np.random.default_rng(0)
    ctrl = make()
    goal = np.array([3.0, 1.0])
    prev = np.zeros(2)
    for _ in range(400):
        x = rng.uniform([0, 0, -np.pi], [3, 2, np.pi])
        v, w = ctrl.command(x, goal)
        assert 0.0 <= v <= ROB["v_max"] and abs(w) <= ROB["w_max"]
        assert abs(v - prev[0]) <= ctrl.p["a_max"] * DT + 1e-9
        assert abs(w - prev[1]) <= ctrl.p["alpha_max"] * DT + 1e-9
        prev = np.array([v, w])


def test_reference_waits_for_a_robot_that_cannot_move():
    """If the robot is held (LiDAR stop, push), the reference must not run away from it."""
    ctrl = make()
    x, goal = np.array([0.5, 0.5, 0.0]), np.array([4.0, 0.5])
    for _ in range(600):
        ctrl.command(x, goal)                                       # pose never changes
    assert np.hypot(*ctrl.history[-1]["e"][:2]) <= ctrl.p["pos_hold"] + 0.02


# ------------------------------------------------------------ closed loop (kinematic)
def cross_track(p, polyline):
    best = np.inf
    for a, b in zip(polyline[:-1], polyline[1:]):
        t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
        best = min(best, np.linalg.norm(p - (a + t * (b - a))))
    return best


def test_follows_the_project_waypoints_on_a_noisy_estimate():
    """Whole waypoint list, controller fed only a noisy estimate (1 cm, 0.01 rad)."""
    rng = np.random.default_rng(1)
    ctrl = make()
    planner = WaypointFollower(CFG["waypoints"], CTL["goal_tol"])
    x = np.array(CFG["sim"]["start_pose"], dtype=float)
    poly = [x[:2]] + [np.array(w, dtype=float) for w in CFG["waypoints"]]
    worst, t = 0.0, 0.0
    while t < 120.0 and not planner.done:
        est = x + rng.normal(0.0, [0.01, 0.01, 0.01])
        goal = planner.current(est)
        if goal is None:
            break
        v, w = ctrl.command(est, goal)
        assert v >= 0.0
        x = unicycle(x, v, w)
        worst = max(worst, cross_track(x[:2], poly))
        t += DT
    assert planner.done, f"stopped at waypoint {planner.index} after {t:.0f} s"
    assert np.hypot(*(x[:2] - np.array(CFG["waypoints"][-1]))) < CTL["goal_tol"] + 0.03
    assert worst < CTL["goal_tol"] + 0.03     # corner cutting is bounded by the waypoint tolerance


def test_recovers_from_a_disturbance():
    """A sideways shove mid-segment is removed by the feedback."""
    ctrl = make()
    x, goal = np.array([0.0, 0.0, 0.0]), np.array([6.0, 0.0])
    for k in range(400):
        if k == 60:
            x = x + np.array([0.0, 0.15, 0.2])
        v, w = ctrl.command(x, goal)
        x = unicycle(x, v, w)
        if x[0] > 5.0:
            break
    assert abs(x[1]) < 0.01 and abs(x[2]) < 0.02
