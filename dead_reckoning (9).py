"""Step 5: open-loop driving with odometry-only estimation, compared against truth.

Shows how dead reckoning drifts, which is the motivation for the EKF.

    python scripts/dead_reckoning.py
"""
import argparse

from _common import load_config, results_dir
from evaluation.ground_truth import RunLogger, true_pose          # evaluation only
from evaluation.metrics import print_summary, summary
from evaluation.plots import trajectory_and_errors
from src.estimation.dead_reckoning import DeadReckoning
from src.sensing.odometry import WheelOdometry
from src.sim.turtlebot_sim import TurtleBotSim

# (duration s, v m/s, w rad/s): a rough loop through open floor
SEQUENCE = [(8, 0.2, 0.0), (3, 0.0, 0.5), (6, 0.15, 0.3), (5, 0.2, 0.0), (4, 0.1, -0.6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--view", action="store_true", help="open the 3D viewer")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sim = TurtleBotSim(cfg)
    if args.view:
        sim.open_viewer()
    rob = cfg["robot"]
    odom = WheelOdometry(rob["wheel_radius"], rob["wheel_base"], rob["encoder_ticks_per_rev"])
    est = DeadReckoning(cfg["sim"]["start_pose"])
    log = RunLogger()

    m = sim.step(0.0, 0.0)
    odom.update(m.enc_ticks)
    for duration, v, w in SEQUENCE:
        for _ in range(round(duration / sim.dt)):
            m = sim.step(v, w)
            est.predict(*odom.update(m.enc_ticks))
            log.log(m.t, est.x, true_pose(sim), (v, w), False)

    data = log.arrays()
    print("Dead reckoning, open loop:")
    print_summary(summary(data))
    out = results_dir() / "dead_reckoning.png"
    trajectory_and_errors(data, cfg, out, "Open loop, odometry only")
    log.save(results_dir() / "dead_reckoning.npz")
    print(f"Saved {out}")
    sim.close()


if __name__ == "__main__":
    main()
