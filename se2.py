"""Planar (SE(2)) helpers and the fixed sensor mounts.

Frame conventions (all right-handed, z up):
    world      fixed room frame, origin at the south-west corner
    base_link  robot centre between the wheels, x forward, y left
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
