"""Step 4: prove every sensor works.

Places the robot facing the dock, drives forward slowly, prints encoder ticks,
LiDAR and the ArUco measurement, and saves an annotated camera frame.

    python scripts/sensor_check.py
"""
import argparse

import cv2
import numpy as np

from _common import load_config, results_dir
from src.sensing.aruco import ArucoDetector
from src.sensing.lidar import min_range_in_sector
from src.sim.turtlebot_sim import TurtleBotSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--pose", default="3.0,2.0,0.0", help="start x,y,yaw (facing the dock by default)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sim = TurtleBotSim(cfg)
    sim.reset([float(v) for v in args.pose.split(",")])
    aruco = ArucoDetector(cfg["marker"], cfg["camera"])

    print(" step | enc L  enc R | lidar front  hits | marker range  bearing")
    last_image, last_meas = None, None
    for k in range(40):
        m = sim.step(0.1, 0.0)
        front = min_range_in_sector(m.lidar_ranges, m.lidar_angles, 0.0, np.radians(10))
        hits = int(np.isfinite(m.lidar_ranges).sum())
        marker = aruco.detect(m.image) if m.image is not None else None
        if m.image is not None:
            last_image, last_meas = m.image, marker
        if k % 4 == 3:
            mk = (f"{last_meas.range:6.3f} m  {np.degrees(last_meas.bearing):6.2f} deg"
                  if last_meas else "   not seen")
            print(f" {k:4d} | {m.enc_ticks[0]:5d} {m.enc_ticks[1]:6d} | {front:8.3f} m  {hits:4d} | {mk}")

    img = cv2.cvtColor(last_image, cv2.COLOR_RGB2BGR)
    if last_meas is not None:
        cv2.aruco.drawDetectedMarkers(img, [last_meas.corners.reshape(1, 4, 2).astype(np.float32)],
                                      np.array([[last_meas.marker_id]]))
        cv2.putText(img, f"range {last_meas.range:.2f} m  bearing {np.degrees(last_meas.bearing):.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out = results_dir() / "sensor_check_camera.png"
    cv2.imwrite(str(out), img)
    print(f"\nSaved annotated camera frame to {out}")
    sim.close()


if __name__ == "__main__":
    main()
