"""Record a run as an MP4 for the demo video.

Left:  top-down map. True path (white) and EKF estimate (red), the EKF's 3-sigma
       position ellipse, LiDAR points placed with the ESTIMATED pose, the current
       goal and any LiDAR detour.
Right: the robot's camera, with the detected marker outlined and its measured range.
Text:  time, phase, commands, EKF sigma, and the true estimation error.

Ground truth is drawn for the viewer only (this is an evaluation script); the
robot itself still runs on the estimate.

    python scripts/record_demo.py                                   # nominal run
    python scripts/record_demo.py --condition push --tag push       # a Part E condition
    python scripts/record_demo.py --condition occluded --speed 2    # 2x speed
    python scripts/record_demo.py --start "0.5,1.5,0" --waypoints "4.3,2.0" --tag obstacle
"""
import argparse
import copy

import cv2
import numpy as np

from _common import load_config, results_dir
from robustness_study import CONDITIONS, random_start
from run import run_trial
from src.frames.se2 import transform_points
from src.sensing.lidar import scan_to_base

W_MAP, H = 660, 560                      # left panel
W_CAM = 640                              # right panel (camera 640 x 480 + text strip)
SCALE = 120                              # pixels per metre on the map
ORIGIN = np.array([30, H - 30])          # map pixel of world (0, 0); legend sits above the room
ELLIPSE_GAIN = 5                         # ellipse drawn 5x larger so it is visible; labelled


def px(p):
    p = np.asarray(p, dtype=float)
    return (ORIGIN + SCALE * np.array([p[..., 0], -p[..., 1]]).T).astype(int)


class Recorder:
    def __init__(self, cfg, out_path, speed, title):
        self.cfg, self.speed, self.title = cfg, speed, title
        fps = round(1.0 / cfg["sim"]["control_dt"])
        self.writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_MAP + W_CAM, H))
        self.true_path, self.est_path = [], []
        self.camera = np.zeros((480, 640, 3), np.uint8)
        self.cam_text = "waiting for first frame"
        self.k = 0
        self.background = self._room()

    # ------------------------------------------------------------------ static map
    def _room(self):
        img = np.full((H, W_MAP, 3), 30, np.uint8)
        cv2.rectangle(img, tuple(px([0, 0])), tuple(px([5, 4])), (200, 200, 200), 2)
        for ob in self.cfg["obstacles"]:
            if ob["type"] == "box":
                a, b = px([ob["x"] - ob["hx"], ob["y"] - ob["hy"]]), px([ob["x"] + ob["hx"], ob["y"] + ob["hy"]])
                cv2.rectangle(img, tuple(a), tuple(b), (60, 90, 160), -1)
            else:
                cv2.circle(img, tuple(px([ob["x"], ob["y"]])), int(ob["r"] * SCALE), (60, 90, 160), -1)
        mx, my, _ = self.cfg["marker"]["pose_world"]
        cv2.rectangle(img, tuple(px([mx, my - 0.2])), tuple(px([mx + 0.14, my + 0.2])), (200, 120, 40), -1)
        cv2.putText(img, "dock", tuple(px([mx - 0.25, my + 0.3])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 160, 90), 1)
        for w in self.cfg["waypoints"]:
            cv2.circle(img, tuple(px(w)), 6, (90, 200, 90), 2)
        return img

    # ------------------------------------------------------------------ per step
    def __call__(self, s):
        self.true_path.append(s["x_true"][:2])
        self.est_path.append(s["x_est"][:2])
        if s["meas"].image is not None:                 # camera runs at 5 Hz: hold the last frame
            self.camera = cv2.cvtColor(s["meas"].image, cv2.COLOR_RGB2BGR)
            mk = s["marker"]
            if mk is not None:
                cv2.polylines(self.camera, [mk.corners.astype(np.int32)], True, (0, 255, 0), 2)
                self.cam_text = f"marker: {mk.range:.2f} m, bearing {np.degrees(mk.bearing):+.1f} deg"
            else:
                self.cam_text = "marker not detected"
        self.k += 1
        if self.k % self.speed:
            return
        self.writer.write(self._frame(s))

    def _frame(self, s):
        m = self.background.copy()
        x_est, x_true, P = s["x_est"], s["x_true"], s["P"]

        # LiDAR points, placed with the ESTIMATED pose (what the robot believes it sees)
        pts = scan_to_base(s["meas"].lidar_ranges, s["meas"].lidar_angles, self.cfg["lidar"])
        for q in px(transform_points(x_est, pts)) if len(pts) else []:
            cv2.circle(m, tuple(q), 2, (80, 220, 220), -1)

        if len(self.true_path) > 1:
            cv2.polylines(m, [px(np.array(self.true_path))], False, (235, 235, 235), 2)
            cv2.polylines(m, [px(np.array(self.est_path))], False, (60, 60, 230), 2)

        if s["goal"] is not None:
            cv2.drawMarker(m, tuple(px(s["goal"])), (90, 200, 90), cv2.MARKER_CROSS, 16, 2)
        if s["target"] is not None and s["goal"] is not None and not np.array_equal(s["target"], s["goal"]):
            cv2.drawMarker(m, tuple(px(s["target"])), (0, 165, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(m, "detour", tuple(px(s["target"]) + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        # EKF 3-sigma position ellipse, magnified so it is visible
        if P is not None and np.all(np.isfinite(P)):
            vals, vecs = np.linalg.eigh(P[:2, :2])
            axes = tuple(int(max(1, ELLIPSE_GAIN * 3 * np.sqrt(max(v, 0)) * SCALE)) for v in vals)
            ang = -np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            cv2.ellipse(m, tuple(px(x_est[:2])), axes, ang, 0, 360, (60, 60, 230), 1)

        for x, col in ((x_true, (235, 235, 235)), (x_est, (60, 60, 230))):
            c = px(x[:2])
            tip = px(x[:2] + 0.18 * np.array([np.cos(x[2]), np.sin(x[2])]))
            cv2.circle(m, tuple(c), int(0.1 * SCALE), col, 2)
            cv2.line(m, tuple(c), tuple(tip), col, 2)

        legend = [("true (sim, evaluation only)", (235, 235, 235)), ("EKF estimate", (60, 60, 230)),
                  (f"3-sigma ellipse x{ELLIPSE_GAIN}", (60, 60, 230)), ("LiDAR points", (80, 220, 220))]
        for i, (txt, col) in enumerate(legend):
            cv2.putText(m, txt, (30 + 300 * (i % 2), 18 + 20 * (i // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Right panel: camera and text strip
        right = np.full((H, W_CAM, 3), 20, np.uint8)
        right[:480] = self.camera
        cv2.putText(right, self.cam_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        err = np.hypot(*(x_est[:2] - x_true[:2]))
        sig = np.sqrt(np.clip(np.diag(P), 0, None)) if P is not None else np.full(3, np.nan)
        v, w = s["cmd"]
        lines = [f"t {s['t']:5.1f} s   phase: {s['phase']}   v {v:.2f} m/s  w {w:+.2f} rad/s",
                 f"EKF sigma x {100 * sig[0]:.1f} cm  y {100 * sig[1]:.1f} cm  th {np.degrees(sig[2]):.1f} deg"
                 f"   true error {100 * err:.1f} cm"]
        for i, txt in enumerate(lines):
            cv2.putText(right, txt, (10, 505 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        banner = self._banner(s["t"])
        if banner:
            cv2.putText(right, banner, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        frame = np.hstack([m, right])
        cv2.putText(frame, self.title, (W_MAP + 10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def _banner(self, t):
        d = self.cfg.get("disturbances") or {}
        for key, label in (("push", "PUSH"), ("slip", "WHEEL SLIP")):
            e = d.get(key)
            if e and e["t"] <= t < e["t"] + e["duration"]:
                return label
        if any(a <= t < b for a, b in d.get("occlusion", [])):
            return "MARKER COVERED"
        return ""

    def close(self):
        self.writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--condition", help="a Part E condition from robustness_study.py (any part of its name)")
    ap.add_argument("--controller", choices=["p", "lqr"])
    ap.add_argument("--start", help="override start pose, 'x,y,yaw'")
    ap.add_argument("--waypoints", help="override waypoints, 'x,y;x,y;...'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=int, default=1, help="1 = real time, 2 = twice as fast, ...")
    ap.add_argument("--tag", default="nominal", help="output name: results/demo_<tag>.mp4")
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    cfg["sim"]["seed"] = args.seed
    title = "nominal"
    if args.condition:
        name = next((n for n in CONDITIONS if args.condition.lower() in n.lower()), None)
        if name is None:
            raise SystemExit(f"no condition matching '{args.condition}'; choose from: {list(CONDITIONS)}")
        change = CONDITIONS[name]
        if change == "random_start":
            cfg["sim"]["start_pose"] = random_start(cfg, np.random.default_rng(100 + args.seed))
        else:
            change(cfg)
        title = name
    if args.controller:
        cfg["controller"]["type"] = args.controller
    if args.start:
        cfg["sim"]["start_pose"] = [float(v) for v in args.start.split(",")]
    if args.waypoints:
        cfg["waypoints"] = [[float(v) for v in p.split(",")] for p in args.waypoints.split(";")]
    title = f"{title}, {cfg['controller'].get('type', 'p').upper()} controller, seed {args.seed}"

    out = results_dir() / f"demo_{args.tag}.mp4"
    rec = Recorder(cfg, out, max(1, args.speed), title)
    log, est, planner = run_trial(cfg, "ekf", args.seed, on_step=rec)
    rec.close()
    state = str(log["phase"][-1]) if "phase" in log else ("done" if planner.done else "stopped")
    print(f"{title}: {state} after {log['t'][-1]:.1f} s")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()