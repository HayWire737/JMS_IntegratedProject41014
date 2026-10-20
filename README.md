# TurtleBot3 Visual Waypoint Navigation and ArUco Docking

**41014 Sensors and Control in Mechatronics, Integrated Project, Track A**

| | |
|---|---|
| **Group** | _Group 99_ |
| **Members** | _Muhammad Suhaib Khan (1351525)_, _Jaden Gambrill(26065197)_, _Siale Smythe (24831226), _Name (student ID)_ |
| **Track** | A: mobile robot, visual waypoint navigation (MuJoCo) |
| **Hardware** | _Simulation only_ |
| **Demo video** | _[link, 5 min]_ |

---

## 1. Overview

_The robot navigates a room with obstacles using 2D LiDAR, then visually docks to the ArUco-marked charging station. An EKF fuses wheel-encoder odometry with camera measurements of the marker (range/bearing or pose). A state-feedback/LQR tracker acts on the **estimated** state only._

_Insert pipeline block diagram: sensors → EKF → planner → LQR → MuJoCo, with ground truth branching to the evaluator only._

## 2. Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip  install -r requirements.txt

python scripts/build_scene.py      # generate models/scene.xml from the config
python -m pytest                   # model checks + ground-truth policy
python scripts/sensor_check.py     # encoders, LiDAR, camera and ArUco working
python scripts/dead_reckoning.py   # open-loop odometry drift vs truth
python scripts/run.py              # closed loop: estimate -> waypoints -> dock
python scripts/run.py --view       # same, with the 3D viewer (macOS: use mjpython)
```

Figures are written to `results/`. All tunable values are in `configs/nominal.yaml`.

Tested on: _OS, Python version, MuJoCo version_.

## 3. Repository layout

```
configs/          YAML configs: robot, sensors, noise, obstacles, waypoints, gains
models/           Generated MuJoCo scene and ArUco texture (rebuild with build_scene.py)
src/
  model/          Differential-drive model, linearisation, observability/controllability
  sensing/        Wheel odometry, LiDAR, ArUco detection and range/bearing
  frames/         SE(2) helpers and sensor mounts (world, base_link, camera, marker)
  estimation/     Dead-reckoning baseline, EKF
  planning/       Waypoint follower, obstacle avoidance from LiDAR
  control/        Go-to-goal baseline, LQR tracker
  sim/            MuJoCo interface. Exposes sensor readings ONLY.
evaluation/       Ground truth access, metrics, plots. NEVER imported by src/
scripts/          Entry points: build_scene, sensor_check, dead_reckoning, run
tests/            Unit tests, including the ground-truth policy check
results/          Logged runs and generated figures
docs/             Images, frame diagram, derivations
```

> **Ground truth policy.** Simulator ground truth is read only in `evaluation/ground_truth.py`. The planner and controller receive the estimate only. `tests/test_no_ground_truth.py` fails if any module under `src/` imports from `evaluation/` or reads the simulator's true pose or velocity.

### Current status

| Component | Status | Lead |
|---|---|---|
| Simulation scene, sensors | Working | _name_ |
| Wheel odometry, dead reckoning | Working (baseline) | _name_ |
| ArUco detection, range/bearing | Working, nominal intrinsics (calibration TODO) | _name_ |
| Go-to-goal controller, waypoints | Working (baseline) | _name_ |
| EKF | TODO: `src/estimation/ekf.py` | _name_ |
| LQR tracker | TODO: `src/control/lqr.py` | _name_ |
| LiDAR obstacle avoidance | TODO: `src/sensing/lidar.py` | _name_ |
| Visual docking, evaluation matrix | TODO | _name_ |

## 4. Modelling (Part A)

- **State:** _x = [x, y, θ]ᵀ (add v, ω if used)_
- **Input:** _u = [v, ω]ᵀ or wheel speeds_
- **Continuous model:** _differential-drive kinematics, derived for TurtleBot3 (wheel radius r = ___ m, track width b = ___ m)_
- **Discretisation:** _method, Δt = ___ s_
- **Linearisation:** _about the reference trajectory; error dynamics used for LQR_
- **Assumptions:** _no wheel slip, planar motion, ..._
- **Controllability:** _rank of controllability matrix; why it fails at v = 0 and how this is handled_
- **Observability:** _which states are observable from odometry alone vs. with marker measurements; what happens when the marker leaves view_

## 5. Sensing and calibration (Part B)

- **Sensors:** _encoders (rate, resolution), LiDAR (range, FOV, rate), camera (resolution, FOV, rate)_
- **Camera intrinsics:** _how obtained, values_
- **Noise characterisation:** _method (e.g. static logging), measured σ for each sensor_
- **ArUco detection:** _dictionary, marker size, detection range limits_
- **Frame diagram:** _figure of world, odom, base_link, camera, marker frames, with transform values_

## 6. Estimation (Part C)

- **Filter:** EKF
- **Process model and Jacobian F:** _..._
- **Measurement model and Jacobian H:** _range/bearing or pose, with Jacobian_
- **Q:** _values and justification (from encoder noise characterisation)_
- **R:** _values and justification (from camera noise characterisation)_
- **Consistency:** _NEES/NIS results with χ² bounds_
- **Marker loss:** _behaviour and covariance growth when no measurement is available_

## 7. Control (Part D)

- **Controller:** _LQR on tracking error / state feedback_
- **Weights / gains:** _Q_lqr, R_lqr values and how they were chosen_
- **Limits:** _velocity and acceleration saturation, effect on tracking_
- **Docking approach:** _final-approach strategy, tolerance_

## 8. Results (Parts C, D, E)

### Nominal case
| Metric | Value |
|---|---|
| Position RMSE (estimate vs truth) | _m_ |
| Heading RMSE | _rad_ |
| Tracking error (RMS) | _m_ |
| Docking final error | _m, rad_ |
| Docking success rate | _x / N_ |

### Robustness (repeated trials, varied conditions)
| Condition | Trials | Success | Est. RMSE | Track RMSE | Notes |
|---|---|---|---|---|---|
| Nominal | | | | | |
| 2× sensor noise | | | | | |
| Marker occlusion | | | | | |
| Wheel slip / bias | | | | | |
| Varied start pose | | | | | |
| External push | | | | | |

### Trade-offs
_e.g. R scaling vs. NEES and responsiveness; LQR weight ratio vs. control effort and tracking error._

### Failure analysis
_Each failure traced to model / sensing / estimator / controller, with evidence._

## 9. Hardware (optional)

_Setup, differences from simulation, quantitative sim-vs-hardware comparison._

## 10. Limitations and future work

_What the results do and do not support._

## 11. Individual contributions

| Member | Led | Contributed to |
|---|---|---|
| _Name_ | _Muhammad Suhaib Khan_ | _Part A & C_ |
| _Name_ | _Jaden Gambrill_ | _Part B & sim enviroment_ |
| _Name_ | _Siale Smythe_ | _Part D_ |
| _Name_ | | |

## 12. Generative AI declaration

| Tool | Purpose | Where used |
|---|---|---|
| _e.g. Claude_ | _e.g. README structure, debugging help_ | _e.g. README, src/estimation_ |

## 13. References

_Tutorials, papers, libraries used._
