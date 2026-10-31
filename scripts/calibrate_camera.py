"""Part B: calibrate the robot camera's intrinsics with a checkerboard.

Builds a small calibration rig in MuJoCo: a camera identical to the robot's
front camera (same field of view and resolution, from the config) and a
printed checkerboard on a stand that can be moved anywhere. The board is shown
to the camera in many positions and tilts, exactly as you would wave a board in
front of a real camera, with the configured camera noise and blur applied.
cv2.calibrateCamera then estimates the focal lengths, principal point and lens
distortion, which are compared with the nominal pinhole values the detector
uses by default.

    python scripts/calibrate_camera.py
"""
import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

from _common import load_config, results_dir
from src.sensing.aruco import nominal_intrinsics

COLS, ROWS = 9, 6          # inner corners of the checkerboard
SQUARE = 0.04              # square edge, m  -> board 0.40 m x 0.28 m
SQUARE_PX = 60             # texture resolution per square


def checkerboard_png():
    """Checkerboard image with a white margin, as PNG bytes."""
    sq = np.indices((ROWS + 1, COLS + 1)).sum(axis=0) % 2          # 7 x 10 squares
    img = np.kron(sq, np.ones((SQUARE_PX, SQUARE_PX))) * 255
    img = cv2.copyMakeBorder(img.astype(np.uint8), SQUARE_PX, SQUARE_PX, SQUARE_PX, SQUARE_PX,
                             cv2.BORDER_CONSTANT, value=255)
    return cv2.imencode(".png", img)[1].tobytes(), img.shape


def rig_xml(cfg, shape):
    """Camera at the origin looking along +x, and a movable (mocap) board."""
    cam = cfg["camera"]
    h_px, w_px = shape
    w, h = w_px / SQUARE_PX * SQUARE, h_px / SQUARE_PX * SQUARE     # board size including margin
    return f"""<mujoco model="camera_calibration_rig">
  <visual>
    <global offwidth="{cam['width']}" offheight="{cam['height']}"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="board" type="2d" file="checker.png"/>
    <material name="board" texture="board"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <camera name="cam" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="{cam['fovy_deg']}"/>
    <body name="board" mocap="true">
      <geom type="box" size="0.002 {w / 2} {h / 2}" material="board" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""


def board_quat(yaw, pitch, roll):
    """Board facing the camera (textured +x face towards -x), then tilted."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, np.pi + yaw]), "xyz")
    return q


def board_poses(n, rng):
    """Random but well-spread board placements: distance, position in the image, tilt."""
    poses = []
    for _ in range(n):
        d = rng.uniform(0.55, 1.3)
        y = rng.uniform(-0.6, 0.6) * d            # spread over the whole image, edges included;
        z = rng.uniform(-0.45, 0.45) * d          # views that leave the frame are simply skipped
        tilt = np.radians(rng.uniform(-35, 35, 3) * [1, 1, 0.3])
        poses.append((np.array([d, y, z]), board_quat(*tilt)))
    return poses


def corrupt(img, noise, rng):
    """Same camera imperfections as the simulator: blur, then pixel noise."""
    blur = noise.get("camera_blur_px", 0)
    sigma = noise.get("camera_pixel_sigma", 0.0)
    if blur > 0:
        img = cv2.GaussianBlur(img, (2 * blur + 1, 2 * blur + 1), 0)
    if sigma > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--views", type=int, default=160, help="board placements to try")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cam = cfg["camera"]
    rng = np.random.default_rng(cfg["sim"]["seed"])

    png, shape = checkerboard_png()
    model = mujoco.MjModel.from_xml_string(rig_xml(cfg, shape), {"checker.png": png})
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, cam["height"], cam["width"])

    # Board corners in the board's own frame (z = 0 plane), in metres
    obj = np.zeros((ROWS * COLS, 3), np.float32)
    obj[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE

    obj_pts, img_pts, used = [], [], 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    for pos, quat in board_poses(args.views, rng):
        data.mocap_pos[0], data.mocap_quat[0] = pos, quat
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="cam")
        gray = cv2.cvtColor(corrupt(renderer.render(), cfg["noise"], rng), cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS))
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        obj_pts.append(obj)
        img_pts.append(corners.reshape(-1, 1, 2).astype(np.float32))   # shape differs between OpenCV versions
        used += 1
    renderer.close()

    size = (cam["width"], cam["height"])
    rms, K, dist, rvecs, tvecs, std_in, _, per_view = cv2.calibrateCameraExtended(
        obj_pts, img_pts, size, None, None)

    # Reject views that fit far worse than the rest (a mis-detected corner drags the
    # whole solution), then calibrate again on the clean set.
    keep = per_view.ravel() < 3 * np.median(per_view)
    rejected = int((~keep).sum())
    if rejected:
        obj_pts = [o for o, k in zip(obj_pts, keep) if k]
        img_pts = [p for p, k in zip(img_pts, keep) if k]
        rms, K, dist, rvecs, tvecs, std_in, _, per_view = cv2.calibrateCameraExtended(
            obj_pts, img_pts, size, None, None)
        used = len(img_pts)
    K_nom, _ = nominal_intrinsics(cam)

    names = ["fx", "fy", "cx", "cy"]
    calib = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    nominal = [K_nom[0, 0], K_nom[1, 1], K_nom[0, 2], K_nom[1, 2]]
    report = {
        "views_tried": args.views,
        "views_used": used,
        "views_rejected_as_outliers": rejected,
        "reprojection_rms_px": float(rms),
        "intrinsics": {n: {"calibrated": float(c), "std": float(s), "nominal": float(m),
                           "difference": float(c - m)}
                       for n, c, s, m in zip(names, calib, std_in.ravel()[:4], nominal)},
        "distortion_k1_k2_p1_p2_k3": [float(v) for v in dist.ravel()[:5]],
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
    }

    out = results_dir()
    with open(out / "camera_calibration.yaml", "w") as f:
        yaml.safe_dump(report, f, sort_keys=False)

    # Figure: where the corners fell in the image (coverage), and per-view error
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for c in img_pts:
        a1.plot(c[:, 0, 0], c[:, 0, 1], ".", ms=2)
    a1.set(xlim=(0, cam["width"]), ylim=(cam["height"], 0), aspect="equal",
           title=f"Corner coverage, {used} views", xlabel="u (px)", ylabel="v (px)")
    a2.bar(range(used), per_view.ravel(), color="#4c72b0")
    a2.axhline(rms, color="k", ls="--", lw=1, label=f"overall RMS {rms:.3f} px")
    a2.set(title="Reprojection error per view", xlabel="view", ylabel="RMS error (px)")
    a2.legend()
    fig.tight_layout()
    fig.savefig(out / "camera_calibration.png", dpi=130)

    print(yaml.safe_dump({k: v for k, v in report.items() if k not in ("K", "dist")}, sort_keys=False))
    print("Paste into configs/nominal.yaml under camera:")
    print("  intrinsics:               # from scripts/calibrate_camera.py")
    print(f"    fx: {K[0, 0]:.2f}")
    print(f"    fy: {K[1, 1]:.2f}")
    print(f"    cx: {K[0, 2]:.2f}")
    print(f"    cy: {K[1, 2]:.2f}")
    print(f"    dist: [{', '.join(f'{v:.4f}' for v in dist.ravel()[:5])}]")
    print(f"Saved {out / 'camera_calibration.yaml'} and {out / 'camera_calibration.png'}")


if __name__ == "__main__":
    main()