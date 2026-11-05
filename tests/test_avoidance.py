"""LiDAR obstacle avoidance tests (planning)."""
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.planning.avoidance import LidarAvoidance

with open(ROOT / "configs/nominal.yaml") as f:
    CFG = yaml.safe_load(f)

ANGLES = 2 * np.pi * np.arange(72) / 72


def scan_of_wall(x_wall, half_width):
    """Scan (lidar frame) of a wall across the path at x = x_wall, |y| < half_width."""
    ranges = np.full(72, np.inf)
    for i, a in enumerate(ANGLES):
        if np.cos(a) > 1e-6:
            r = x_wall / np.cos(a)
            if abs(r * np.sin(a)) < half_width:
                ranges[i] = r
    return ranges


def test_blocked_uses_the_robot_corridor():
    av = LidarAvoidance(CFG)
    ahead = np.array([[0.5, 0.0]])
    beside = np.array([[0.5, av.r_safe + 0.05]])
    assert av.blocked(ahead, 0.0, 1.0)
    assert not av.blocked(beside, 0.0, 1.0)
    assert not av.blocked(ahead, 0.0, 0.3)          # beyond the length checked
    assert not av.blocked(ahead, np.pi, 1.0)        # behind the robot


def test_clear_path_goes_straight_to_the_goal():
    av = LidarAvoidance(CFG)
    goal = np.array([2.0, 0.0])
    target = av.target(np.zeros(3), goal, np.full(72, np.inf), ANGLES)
    assert np.array_equal(target, goal) and av.detour is None


def test_blocked_path_gets_a_clear_detour_and_holds_it():
    av = LidarAvoidance(CFG)
    goal = np.array([2.0, 0.0])
    ranges = scan_of_wall(0.6, 0.4)                 # 0.8 m wide wall 0.6 m ahead
    target = av.target(np.zeros(3), goal, ranges, ANGLES)
    assert av.detour is not None and av.switched
    assert abs(np.hypot(*target) - av.step) < 1e-9  # detour placed detour_distance away
    heading = np.arctan2(target[1], target[0])
    assert abs(heading) > 0.1                        # it turned away from the wall
    # still blocked next step: the same detour is kept (no flicker)
    again = av.target(np.zeros(3), goal, ranges, ANGLES)
    assert np.array_equal(again, target) and not av.switched


def test_detour_dropped_once_the_way_is_clear():
    av = LidarAvoidance(CFG)
    goal = np.array([2.0, 0.0])
    av.target(np.zeros(3), goal, scan_of_wall(0.6, 0.4), ANGLES)
    target = av.target(np.zeros(3), goal, np.full(72, np.inf), ANGLES)
    assert np.array_equal(target, goal) and av.detour is None and av.switched


def test_drives_around_an_obstacle_in_the_simulator():
    """Straight line from the start to the goal clips the box at (2.0, 1.5): without
    avoidance the safety stop strands the robot; with it, it reaches the goal."""
    from run import run_trial
    cfg = copy.deepcopy(CFG)
    cfg["sim"]["start_pose"], cfg["waypoints"] = [0.5, 1.5, 0.0], [[4.3, 2.0]]
    cfg["sim"]["max_time"] = 40.0
    log, _, planner = run_trial(cfg, "ekf", seed=0)
    assert planner.done, "did not reach the goal"
    p = log["truth"][:, :2]                         # truth is fine here: this is a test, not src/
    q = np.abs(p - [2.0, 1.5]) - [0.2, 0.2]
    clearance = np.hypot(*np.maximum(q, 0).T) + np.minimum(q.max(axis=1), 0)
    assert clearance.min() > CFG["avoidance"]["robot_radius"], "robot body hit the box"