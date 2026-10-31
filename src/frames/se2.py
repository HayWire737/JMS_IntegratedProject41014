"""Planar (SE(2)) helpers and the fixed sensor mounts.

Frame conventions (all right-handed, z up):
    world      fixed room frame, origin at the south-west corner
    base_link  robot centre between the wheels, x forward, y left
    lidar      scan centre, axes parallel to base_link, mounted at (lidar.x, 0, lidar.z)
    camera     OpenCV optical frame: z forward, x right, y down
    marker     ArUco frame: origin at the marker centre, z out of the marker face
"""
import numpy as np


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    """Pose a ⊕ b: express pose b (given in frame a) in a's parent frame."""
    x, y, th = a
    c, s = np.cos(th), np.sin(th)
    return np.array([x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], wrap(th + b[2])])


def camera_point_to_base(p_cam, cam_cfg):
    """Convert a 3D point from the OpenCV camera frame to base_link.

    camera z (forward) -> base +x, camera x (right) -> base -y, camera y (down) -> base -z.
    """
    xc, yc, zc = p_cam
    return np.array([zc + cam_cfg["x"], -xc, -yc + cam_cfg["z"]])


def inverse(a):
    """Inverse pose: if a is frame B expressed in A, inverse(a) is A expressed in B."""
    x, y, th = a
    c, s = np.cos(th), np.sin(th)
    return np.array([-c * x - s * y, s * x - c * y, wrap(-th)])


def transform_points(pose, pts):
    """Express (N, 2) points given in the frame `pose` in that frame's parent."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.asarray(pose[:2], dtype=float)


def lidar_mount(lidar_cfg):
    """Pose of the lidar frame in base_link, [x, y, yaw]."""
    return np.array([lidar_cfg["x"], lidar_cfg.get("y", 0.0), lidar_cfg.get("yaw", 0.0)])


def lidar_points_to_base(pts_lidar, lidar_cfg):
    """Convert (N, 2) LiDAR points from the lidar frame to base_link."""
    return transform_points(lidar_mount(lidar_cfg), pts_lidar)
