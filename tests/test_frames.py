"""Frame management tests (Part B).

1. The SE(2) helpers obey the algebra (inverse, composition, point transforms).
2. Each sensor transform puts a known point where it physically is.
3. The config and the generated MuJoCo scene agree on where the sensors are mounted.
4. End to end: a real LiDAR scan, taken from a known pose and pushed through
   lidar -> base_link -> world, lands on the walls and obstacles of the room.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.frames.se2 import (camera_point_to_base, compose, inverse, lidar_mount,
                            lidar_points_to_base, transform_points, wrap)
from src.sensing.lidar import scan_to_base
from src.sim.turtlebot_sim import TurtleBotSim

with open(ROOT / "configs/nominal.yaml") as f:
    CFG = yaml.safe_load(f)


# algebra
def test_inverse_composes_to_identity():
    for a in ([1.0, 2.0, 0.3], [-0.5, 4.0, 3.0], [0.0, 0.0, -2.5]):
        assert np.allclose(compose(a, inverse(a)), 0, atol=1e-12)
        assert np.allclose(compose(inverse(a), a), 0, atol=1e-12)


def test_transform_points_matches_compose():
    pose, p = np.array([1.0, -2.0, 0.8]), np.array([0.4, 0.3])
    assert np.allclose(transform_points(pose, p)[0], compose(pose, [*p, 0.0])[:2])


def test_wrap_range():
    a = wrap(np.linspace(-20, 20, 1001))
    assert a.min() >= -np.pi and a.max() < np.pi


# sensor mounts
def test_camera_point_straight_ahead():
    cam = CFG["camera"]
    p = camera_point_to_base([0.0, 0.0, 1.0], cam)      # 1 m along the optical axis
    assert np.allclose(p, [1.0 + cam["x"], 0.0, cam["z"]])


def test_camera_axes():
    cam, d = CFG["camera"], 1.0
    right = camera_point_to_base([0.2, 0.0, d], cam)    # camera +x is image right
    down = camera_point_to_base([0.0, 0.2, d], cam)     # camera +y is image down
    assert right[1] < 0                                  # robot's right is base -y
    assert down[2] < cam["z"]                            # image down is lower in the world


def test_lidar_point_straight_ahead():
    lid = CFG["lidar"]
    p = lidar_points_to_base([[1.0, 0.0]], lid)[0]
    assert np.allclose(p, [1.0 + lid["x"], 0.0])


# config agrees with the scene
def test_scene_matches_config():
    model = mujoco.MjModel.from_xml_path(str(ROOT / CFG["sim"]["model"]))
    lidar_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lidar")
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
    assert np.allclose(model.body_pos[lidar_body], [CFG["lidar"]["x"], 0, CFG["lidar"]["z"]]), \
        "lidar mount differs from config: run scripts/build_scene.py"
    assert np.allclose(model.cam_pos[cam], [CFG["camera"]["x"], 0, CFG["camera"]["z"]]), \
        "camera mount differs from config: run scripts/build_scene.py"


# end to end with a real scan
def _distance_to_room(pts):
    """Distance from each world point to the nearest wall, obstacle or dock surface."""
    def box(p, cx, cy, hx, hy):
        q = np.abs(p - [cx, cy]) - [hx, hy]
        return np.hypot(*np.maximum(q, 0).T) + np.minimum(q.max(axis=1), 0)
    d = [np.abs(pts[:, 0]), np.abs(pts[:, 0] - 5), np.abs(pts[:, 1]), np.abs(pts[:, 1] - 4)]
    for ob in CFG["obstacles"]:
        if ob["type"] == "box":
            d.append(np.abs(box(pts, ob["x"], ob["y"], ob["hx"], ob["hy"])))
        else:
            d.append(np.abs(np.hypot(pts[:, 0] - ob["x"], pts[:, 1] - ob["y"]) - ob["r"]))
    mx, my, _ = CFG["marker"]["pose_world"]
    d.append(np.abs(box(pts, mx + 0.08, my, 0.07, 0.2)))          # dock
    return np.min(d, axis=0)


def test_scan_lands_on_room_geometry():
    pose = np.array([1.2, 2.6, 0.9])          # open floor, rotated so a mount error would show
    sim = TurtleBotSim(CFG)
    sim.reset(pose)
    m = sim.step(0.0, 0.0)
    sim.close()

    finite = m.lidar_ranges[np.isfinite(m.lidar_ranges)]
    assert np.all(finite < CFG["lidar"]["max_range"] + 0.05), "returns at max range: phantom obstacles"

    world = transform_points(pose, scan_to_base(m.lidar_ranges, m.lidar_angles, CFG["lidar"]))
    err = _distance_to_room(world)
    assert len(err) > 50, "too few LiDAR returns"
    assert np.max(err) < 0.05, f"a return landed {np.max(err):.3f} m away from any surface"
    assert np.median(err) < 0.015, f"scan does not land on the room: median {np.median(err):.3f} m"

    # The same scan with the mount offset ignored must fit worse, otherwise this test
    # could not detect a frame error in the first place.
    no_mount = dict(CFG["lidar"], x=0.0)
    err_wrong = _distance_to_room(transform_points(pose, scan_to_base(m.lidar_ranges, m.lidar_angles, no_mount)))
    assert np.mean(err) < np.mean(err_wrong)