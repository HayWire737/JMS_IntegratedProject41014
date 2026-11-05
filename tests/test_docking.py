"""Unit tests for the visual docking stage. Pure kinematics with synthetic detections, no MuJoCo."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import docking_error
from src.control.docking import APPROACH, DOCKED, FAILED, SEARCH, VisualDocking, image_bearing, robot_in_marker_frame
from src.frames.se2 import wrap
from src.sensing.aruco import MarkerMeasurement

CFG = yaml.safe_load(open(ROOT / "configs/nominal.yaml"))
DOCK, CTL, ROB, CAM = CFG["docking"], CFG["controller"], CFG["robot"], CFG["camera"]
DT = CFG["sim"]["control_dt"]
MARKER = np.array(CFG["marker"]["pose_world"])
HALF_FOV = np.radians(37.0)        # horizontal half field of view


def make(**overrides):
    return VisualDocking({**DOCK, **overrides}, CTL, ROB, dt=DT)


def unicycle(x, v, w, dt=DT):
    m = x[2] + 0.5 * w * dt
    return np.array([x[0] + v * dt * np.cos(m), x[1] + v * dt * np.sin(m), wrap(x[2] + w * dt)])


def detection(x_world, rng=None, sigma=0.0):
    """Synthetic detection of the marker from world pose x_world, or None when out of view."""
    dx, dy = MARKER[0] - x_world[0], MARKER[1] - x_world[1]
    c, s = np.cos(x_world[2]), np.sin(x_world[2])
    xb, yb = c * dx + s * dy, -s * dx + c * dy
    yaw = wrap(MARKER[2] - x_world[2])
    if rng is not None and sigma > 0:
        xb, yb, yaw = xb + rng.normal(0, sigma), yb + rng.normal(0, sigma), yaw + rng.normal(0, 5 * sigma)
    if xb - CAM["x"] < 0.15 or abs(np.arctan2(yb, xb - CAM["x"])) > HALF_FOV:
        return None
    p_base = np.array([xb, yb, CAM["z"]])
    t_cam = np.array([-yb, 0.0, xb - CAM["x"]])           # base_link -> optical frame
    return MarkerMeasurement(0, float(np.hypot(xb, yb)), float(np.arctan2(yb, xb)), p_base, float(yaw),
                             np.zeros((4, 2)), t_cam)


def run(dock, x0, steps=800, sigma=0.0, seed=0, visible=lambda k: True):
    """Closed loop: camera every 4 steps, odometry increments from the true motion."""
    rng = np.random.default_rng(seed)
    x, ds, dth = np.array(x0, dtype=float), 0.0, 0.0
    for k in range(steps):
        meas = detection(x, rng, sigma) if (k % 4 == 0 and visible(k)) else None
        dock.observe(ds, dth, meas)
        v, w = dock.command()
        if dock.finished:
            break
        x_new = unicycle(x, v, w)
        ds, dth = v * DT, w * DT
        x = x_new
    return x


def start_in_front(dist=0.54, lateral=0.0, heading=0.0):
    """World pose at `dist` in front of the marker, offset sideways, facing it plus `heading`."""
    n = np.array([np.cos(MARKER[2]), np.sin(MARKER[2])])
    perp = np.array([-n[1], n[0]])
    p = MARKER[:2] + dist * n + lateral * perp
    return np.array([p[0], p[1], wrap(MARKER[2] + np.pi + heading)])


# ------------------------------------------------------------ geometry
def test_relative_pose_from_a_detection():
    x = start_in_front(0.5, lateral=0.1, heading=0.2)
    q = robot_in_marker_frame(detection(x))
    assert np.allclose(q[:2], [0.5, 0.1], atol=1e-9)
    assert abs(wrap(q[2] - (np.pi + 0.2))) < 1e-9


def test_image_bearing_is_zero_when_facing_the_marker():
    assert abs(image_bearing(detection(start_in_front(0.4, lateral=0.0)))) < 1e-12
    assert image_bearing(detection(start_in_front(0.4, heading=-0.2))) > 0     # robot turned right: marker to the left


def test_odometry_carries_the_relative_pose_between_frames():
    dock = make()
    x = start_in_front(0.6, lateral=0.05, heading=0.1)
    dock.observe(0.0, 0.0, detection(x))
    for _ in range(4):
        x_new = unicycle(x, 0.1, 0.3)
        dock.observe(0.1 * DT, 0.3 * DT, None)
        x = x_new
    expected = robot_in_marker_frame(detection(x))
    assert np.allclose(dock.q[:2], expected[:2], atol=1e-6) and abs(wrap(dock.q[2] - expected[2])) < 1e-9


# ------------------------------------------------------------ closed loop
@pytest.mark.parametrize("lateral,heading", [(0.0, 0.0), (0.06, 0.0), (-0.06, 0.15), (0.03, -0.25)])
def test_docks_from_around_the_last_waypoint(lateral, heading):
    dock = make()
    x = run(dock, start_in_front(0.54, lateral, heading), sigma=0.002)
    assert dock.state == DOCKED, dock.reason
    err = docking_error(x, CFG)
    assert err["success"], err
    assert abs(err["range_err_m"]) < 0.02
    assert abs(err["lateral_m"]) < 0.7 * abs(lateral) + 0.005     # offset shrinks over the short approach


def test_stops_and_stays_stopped_once_docked():
    dock = make()
    run(dock, start_in_front(0.5))
    assert dock.docked
    for _ in range(5):
        dock.observe(0.0, 0.0, None)
        assert dock.command() == (0.0, 0.0)


def test_waits_for_a_fresh_frame_before_deciding():
    """Odometry alone never declares docking: the robot holds at the stop point until a frame arrives."""
    dock = make(lost_timeout=20.0)
    x = start_in_front(0.5)
    dock.observe(0.0, 0.0, detection(x))
    for _ in range(int(10.0 / DT)):                    # drive with no further frames
        v, w = dock.command()
        x_new = unicycle(x, v, w)
        dock.observe(v * DT, w * DT, None)
        x = x_new
    assert dock.state == APPROACH and (v, w) == (0.0, 0.0)
    assert abs(np.hypot(*(x[:2] - MARKER[:2])) - DOCK["stop_distance"]) < 0.01
    dock.observe(0.0, 0.0, detection(x))
    dock.command()
    assert dock.state == DOCKED


def test_searches_towards_the_marker_when_it_starts_out_of_view():
    dock = make()
    x = start_in_front(0.6, heading=1.2)                # marker far to the right, out of view
    assert detection(x) is None
    dock.observe(0.0, 0.0, None)
    assert dock.command() == (0.0, 0.0)                 # waits for a first frame
    x = run(dock, x)
    assert dock.state == DOCKED, dock.reason
    assert docking_error(x, CFG)["success"]


def test_search_turns_towards_the_last_known_bearing():
    dock = make()
    dock.observe(0.0, 0.0, detection(start_in_front(0.6, heading=-0.3)))   # marker on the left
    for _ in range(int((DOCK["lost_timeout"] + 0.5) / DT)):
        dock.observe(0.0, 0.0, None)
    v, w = dock.command()
    assert dock.state == SEARCH
    assert v == 0.0 and w > 0.0


def test_keeps_approaching_through_short_dropouts():
    dock = make()
    x = run(dock, start_in_front(0.54, 0.04), visible=lambda k: not (10 <= k < 40))
    assert dock.state == DOCKED and docking_error(x, CFG)["success"]


def test_gives_up_after_max_time_without_the_marker():
    dock = make(max_time=5.0)
    run(dock, start_in_front(0.6), visible=lambda k: False)
    assert dock.state == FAILED and "timed out" in dock.reason
    assert dock.command() == (0.0, 0.0)


def test_fails_rather_than_docks_when_the_marker_is_off_centre():
    dock = make()
    x = start_in_front(0.305, heading=0.3)              # at the stop distance, marker 22 deg off centre
    dock.observe(0.0, 0.0, detection(x))
    assert dock.command() == (0.0, 0.0)
    assert dock.state == FAILED


def test_never_reverses():
    dock = make()
    rng = np.random.default_rng(1)
    x = start_in_front(0.54, 0.05, 0.2)
    for k in range(600):
        dock.observe(0.0, 0.0, detection(x, rng, 0.003) if k % 4 == 0 else None)
        v, w = dock.command()
        assert v >= 0.0
        if dock.finished:
            break
        x = unicycle(x, v, w)


def test_docking_error_against_truth():
    err = docking_error(start_in_front(0.30), CFG)
    assert abs(err["range_m"] - 0.30) < 1e-9 and abs(err["bearing_deg"]) < 1e-9 and err["success"]
    assert not docking_error(start_in_front(0.40), CFG)["success"]
    assert not docking_error(start_in_front(0.30, heading=0.35), CFG)["success"]
    assert abs(docking_error(start_in_front(0.30, lateral=0.02), CFG)["lateral_m"] - 0.02) < 1e-9


# ------------------------------------------------------------ simulator
def test_docks_in_the_simulator():
    """Full run in MuJoCo (route, then docking from real camera frames). Truth is read
    here only to score the result."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from run import run_trial
    from evaluation.metrics import docking_result
    log, _, planner = run_trial(CFG, "ekf", seed=0)
    assert planner.done, "did not finish the route"
    res = docking_result(log, CFG)
    assert res["state"] == DOCKED
    assert res["success"], res
    assert abs(res["meas_range_m"] - res["range_m"]) < 0.01     # camera range agrees with truth
