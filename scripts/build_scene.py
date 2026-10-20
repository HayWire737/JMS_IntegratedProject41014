"""Generate the MuJoCo scene (TurtleBot3-style robot, room, obstacles, ArUco dock).

Reads robot, camera, LiDAR and marker settings from the config so the scene and
the code always agree. Writes models/scene.xml and models/aruco_<id>.png.

    python scripts/build_scene.py --config configs/nominal.yaml
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Room is x in [0, 5], y in [0, 4]. Obstacles come from the config.


def make_marker_png(cfg, out_path):
    """ArUco marker with a white quiet zone, as it would be printed."""
    m = cfg["marker"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, m["dictionary"]))
    inner = 400
    border = inner // 6           # quiet zone so the detector can find the edge
    img = cv2.aruco.generateImageMarker(dictionary, m["id"], inner)
    img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(str(out_path), img)
    # Physical size of the textured plate including the white border
    return m["size"] * (inner + 2 * border) / inner


def lidar_sites(cfg):
    n = cfg["lidar"]["n_beams"]
    lines = []
    for i in range(n):
        a = 2 * math.pi * i / n
        lines.append(
            f'          <site name="lidar_{i}" pos="0 0 0" size="0.002" '
            f'zaxis="{math.cos(a):.6f} {math.sin(a):.6f} 0" rgba="1 0 0 0"/>'
        )
    return "\n".join(lines)


def lidar_sensors(cfg):
    n, rmax = cfg["lidar"]["n_beams"], cfg["lidar"]["max_range"]
    return "\n".join(
        f'    <rangefinder name="lidar_{i}" site="lidar_{i}" cutoff="{rmax}"/>' for i in range(n)
    )


def obstacles_xml(cfg):
    out = []
    for k, ob in enumerate(cfg.get("obstacles", [])):
        x, y = ob["x"], ob["y"]
        if ob["type"] == "box":
            size = f'{ob["hx"]} {ob["hy"]} 0.2'
        else:
            size = f'{ob["r"]} 0.2'
        out.append(f'    <geom name="obstacle_{k}" type="{ob["type"]}" pos="{x} {y} 0.2" '
                   f'size="{size}" material="obstacle"/>')
    return "\n".join(out)


def build(cfg):
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    png = models / f"aruco_{cfg['marker']['id']}.png"
    plate = make_marker_png(cfg, png)

    r = cfg["robot"]["wheel_radius"]
    half_track = cfg["robot"]["wheel_base"] / 2
    w_max = cfg["robot"]["v_max"] / r * 1.2
    cam, lid = cfg["camera"], cfg["lidar"]
    mx, my, myaw = cfg["marker"]["pose_world"]
    sx, sy, _ = cfg["sim"]["start_pose"]

    # Marker plate: thin box whose +x face (the textured face) points along the
    # marker normal. The euler rotation turns the box so +x -> marker yaw.
    xml = f"""<mujoco model="turtlebot3_track_a">
  <compiler angle="radian"/>
  <option timestep="{cfg['sim']['physics_dt']}" integrator="implicitfast"/>

  <visual>
    <global offwidth="{cam['width']}" offheight="{cam['height']}"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.8 0.85 0.9" rgb2="0.95 0.95 0.95" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.75 0.75 0.75" rgb2="0.65 0.65 0.65" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="10 10"/>
    <material name="wall" rgba="0.85 0.85 0.8 1"/>
    <material name="obstacle" rgba="0.6 0.3 0.2 1"/>
    <material name="body" rgba="0.15 0.15 0.15 1"/>
    <material name="wheel" rgba="0.05 0.05 0.05 1"/>
    <material name="dock" rgba="0.2 0.4 0.8 1"/>
    <texture name="aruco" type="2d" file="{png.name}"/>
    <material name="aruco" texture="aruco"/>
  </asset>

  <worldbody>
    <light pos="2.5 2 3" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="3 2.5 0.1" pos="2.5 2 0" material="floor"/>

    <!-- Walls -->
    <geom name="wall_s" type="box" pos="2.5 -0.05 0.25" size="2.6 0.05 0.25" material="wall"/>
    <geom name="wall_n" type="box" pos="2.5 4.05 0.25" size="2.6 0.05 0.25" material="wall"/>
    <geom name="wall_w" type="box" pos="-0.05 2 0.25" size="0.05 2.1 0.25" material="wall"/>
    <geom name="wall_e" type="box" pos="5.05 2 0.25" size="0.05 2.1 0.25" material="wall"/>

    <!-- Obstacles -->
{obstacles_xml(cfg)}

    <!-- Charging dock with ArUco marker -->
    <geom name="dock" type="box" pos="{mx + 0.08} {my} 0.15" size="0.07 0.2 0.15" material="dock"/>
    <body name="marker" pos="{mx} {my} {cam['z']}" euler="0 0 {myaw}">
      <geom name="marker_plate" type="box" size="0.002 {plate / 2} {plate / 2}"
            material="aruco" contype="0" conaffinity="0"/>
    </body>

    <!-- Robot (TurtleBot3 Burger proportions) -->
    <body name="base_link" pos="{sx} {sy} 0.0">
      <freejoint name="base_free"/>
      <inertial pos="-0.02 0 0.06" mass="1.0" diaginertia="0.008 0.008 0.006"/>
      <geom name="chassis" type="box" pos="-0.02 0 0.07" size="0.07 0.07 0.05" material="body"/>
      <geom name="caster" type="sphere" pos="-0.08 0 0.01" size="0.01" condim="1" priority="1" friction="0 0 0"/>

      <body name="wheel_left" pos="0 {half_track} {r}">
        <joint name="wheel_left_joint" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom type="cylinder" size="{r} 0.009" euler="1.5708 0 0" material="wheel" contype="0" conaffinity="0"/>
        <geom type="sphere" size="{r}" friction="1.0 0.005 0.0001" rgba="0 0 0 0" mass="0.05"/>
      </body>
      <body name="wheel_right" pos="0 {-half_track} {r}">
        <joint name="wheel_right_joint" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom type="cylinder" size="{r} 0.009" euler="1.5708 0 0" material="wheel" contype="0" conaffinity="0"/>
        <geom type="sphere" size="{r}" friction="1.0 0.005 0.0001" rgba="0 0 0 0" mass="0.05"/>
      </body>

      <body name="lidar" pos="-0.02 0 {lid['z']}">
        <geom name="lidar_housing" type="cylinder" size="0.035 0.02" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
{lidar_sites(cfg)}
      </body>

      <!-- MuJoCo cameras look along -z; this xyaxes makes it look along base +x -->
      <camera name="front_camera" pos="{cam['x']} 0 {cam['z']}" xyaxes="0 -1 0 0 0 1" fovy="{cam['fovy_deg']}"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="motor_left" joint="wheel_left_joint" kv="0.2" forcelimited="true" forcerange="-0.05 0.05" ctrlrange="{-w_max:.3f} {w_max:.3f}"/>
    <velocity name="motor_right" joint="wheel_right_joint" kv="0.2" forcelimited="true" forcerange="-0.05 0.05" ctrlrange="{-w_max:.3f} {w_max:.3f}"/>
  </actuator>

  <sensor>
    <jointpos name="encoder_left" joint="wheel_left_joint"/>
    <jointpos name="encoder_right" joint="wheel_right_joint"/>
{lidar_sensors(cfg)}
  </sensor>
</mujoco>
"""
    out = models / "scene.xml"
    out.write_text(xml)
    print(f"Wrote {out.relative_to(ROOT)} and {png.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    args = ap.parse_args()
    with open(ROOT / args.config) as f:
        build(yaml.safe_load(f))
