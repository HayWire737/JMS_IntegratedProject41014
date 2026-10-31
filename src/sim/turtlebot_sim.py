"""MuJoCo interface for the TurtleBot3.

This class exposes SENSOR READINGS ONLY: wheel encoder ticks, LiDAR ranges and
camera images. It never returns the robot's true pose. Ground truth is read
exclusively by evaluation/ground_truth.py, so the estimator and controller
cannot use it by accident.
"""
from dataclasses import dataclass
from pathlib import Path
import time

import cv2

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Measurements:
    t: float
    enc_ticks: np.ndarray          # [left, right] cumulative integer ticks
    lidar_ranges: np.ndarray       # metres; np.inf where nothing was hit
    lidar_angles: np.ndarray       # beam angles in the lidar frame (rad)
    image: np.ndarray | None       # RGB uint8, or None on steps without a frame


class TurtleBotSim:
    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self._model = mujoco.MjModel.from_xml_path(str(ROOT / cfg["sim"]["model"]))
        self._data = mujoco.MjData(self._model)
        seed = cfg["sim"]["seed"] if seed is None else seed
        self.rng = np.random.default_rng(seed)            # encoders
        self._cam_rng = np.random.default_rng([seed, 1])
        self._lidar_rng = np.random.default_rng([seed, 2])

        self.r = cfg["robot"]["wheel_radius"]
        self.b = cfg["robot"]["wheel_base"]
        self.ticks_per_rev = cfg["robot"]["encoder_ticks_per_rev"]
        self.n_sub = round(cfg["sim"]["control_dt"] / self._model.opt.timestep)
        self.dt = self.n_sub * self._model.opt.timestep

        n = cfg["lidar"]["n_beams"]
        self.lidar_angles = 2 * np.pi * np.arange(n) / n
        self._enc_adr = [self._sensor_adr("encoder_left"), self._sensor_adr("encoder_right")]
        self._lidar_adr = np.array([self._sensor_adr(f"lidar_{i}") for i in range(n)])

        cam = cfg["camera"]
        self._renderer = None
        self._cam_size = (cam["height"], cam["width"])
        self._scene_opt = mujoco.MjvOption()
        self._scene_opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

        self._viewer = None
        self._step_count = 0
        self.reset(cfg["sim"]["start_pose"])

    # ------------------------------------------------------------------ setup
    def _sensor_adr(self, name):
        return self._model.sensor_adr[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, name)]

    def reset(self, pose):
        """Place the robot at pose = [x, y, yaw]. Writing the start pose is the
        only place the simulator state is touched directly."""
        mujoco.mj_resetData(self._model, self._data)
        x, y, yaw = pose
        self._data.qpos[0:3] = [x, y, 0.0]
        self._data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        self._tick_noise = np.zeros(2)

    def open_viewer(self):
        """Interactive 3D viewer. On macOS run scripts with `mjpython` instead of `python`."""
        import mujoco.viewer
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    # ------------------------------------------------------------------ step
    def step(self, v, w):
        """Apply body velocity command (v, w), advance one control period and
        return the sensor measurements."""
        wl = (v - w * self.b / 2) / self.r
        wr = (v + w * self.b / 2) / self.r
        self._data.ctrl[:] = [wl, wr]
        t0 = time.perf_counter()
        for _ in range(self.n_sub):
            mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        if self._viewer is not None:
            self._viewer.sync()
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t0)))   # real-time pacing

        return self._measure()

    def _measure(self):
        noise = self.cfg["noise"]
        angles = np.array([self._data.sensordata[a] for a in self._enc_adr])
        # A wheel whose real radius differs from nominal reports a scaled rotation
        scale = 1.0 + np.asarray(noise.get("wheel_scale_error", [0.0, 0.0]))
        if noise["encoder_tick_sigma"] > 0:
            self._tick_noise += self.rng.normal(0, noise["encoder_tick_sigma"], 2)
        ticks = np.round(angles * scale / (2 * np.pi) * self.ticks_per_rev + self._tick_noise)

        ranges = self._data.sensordata[self._lidar_adr].copy()
        hit = (ranges >= 0) & (ranges < self.cfg["lidar"]["max_range"])
        ranges[hit] += self._lidar_rng.normal(0, noise["lidar_sigma"], ranges.size)[hit]
        ranges[~hit] = np.inf

        image = None
        if self._step_count % self.cfg["sim"]["camera_every"] == 0:
            image = self.render_camera()

        return Measurements(self._data.time, ticks.astype(int), ranges, self.lidar_angles, image)

    def render_camera(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, *self._cam_size)
        self._renderer.update_scene(self._data, camera="front_camera", scene_option=self._scene_opt)
        return self._corrupt_image(self._renderer.render())

    def _corrupt_image(self, img):
        """Camera imperfections the detector does not know about: lens blur, then pixel noise."""
        noise = self.cfg["noise"]
        blur = noise.get("camera_blur_px", 0)
        sigma = noise.get("camera_pixel_sigma", 0.0)
        if blur > 0:
            k = 2 * blur + 1                              # Gaussian kernel size must be odd
            img = cv2.GaussianBlur(img, (k, k), 0)
        if sigma > 0:   # sensor noise
            img = img.astype(np.float32) + self._cam_rng.normal(0.0, sigma, img.shape)
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
