# JMS_IntegratedProject41014
High Distinctions all around! 
Lets Gooooo
yay
# TurtleBot3 Visual Waypoint Navigation and ArUco Docking

**41014 Sensors and Control in Mechatronics, Integrated Project, Track A**

| | |
|---|---|
| **Group** | _Group 99_ |
| **Members** | _Muhammad Suhaib Khan (1351525)_, _Name (student ID)_, _Name (student ID)_, _Name (student ID)_ |
| **Track** | A: mobile robot, visual waypoint navigation (MuJoCo) |
| **Hardware** | _Simulation only / Simulation + TurtleBot3 hardware_ |
| **Demo video** | _[link, 5 min]_ |

---

## 1. Overview

_The robot navigates a room with obstacles using 2D LiDAR, then visually docks to the ArUco-marked charging station. An EKF fuses wheel-encoder odometry with camera measurements of the marker (range/bearing or pose). A state-feedback/LQR tracker acts on the **estimated** state only._

_Insert pipeline block diagram: sensors → EKF → planner → LQR → MuJoCo, with ground truth branching to the evaluator only._

## 2. Quick start

```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Nominal run
python scripts/run.py --config configs/nominal.yaml

# Full evaluation (repeated trials, varied conditions)
python scripts/evaluate.py --config configs/eval_matrix.yaml --trials 20
```

Tested on: _OS, Python version, MuJoCo version_.

## 3. Repository layout

```
configs/          YAML configs: noise, gains, Q/R, scenarios
models/           MuJoCo XML for TurtleBot3, room, docking station
src/
  model/          State-space model, linearisation, observability/controllability
  sensing/        Encoders, LiDAR, camera, ArUco detection, calibration
  frames/         Transforms (world, odom, base_link, camera, marker)
  estimation/     EKF: predict/update, Jacobians, consistency checks
  planning/       Waypoints, obstacle avoidance from LiDAR
  control/        LQR / state-feedback tracker
  sim/            MuJoCo interface. Exposes sensor readings ONLY.
evaluation/       Ground truth access, metrics, plots. NEVER imported by src/
scripts/          Entry points: run, evaluate, calibrate, make_figures
tests/            Unit tests, including a check that src/ never imports evaluation/
results/          Logged runs and generated figures
docs/             Images, frame diagram, derivations
```

> **Ground truth policy.** Simulator ground truth is read only inside `evaluation/`. The controller receives the EKF estimate. `tests/test_no_ground_truth.py` fails if any module under `src/` imports from `evaluation/` or reads MuJoCo body poses directly.

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
| _Name_ | _e.g. EKF design and tuning_ | _e.g. evaluation scripts_ |
| _Name_ | | |
| _Name_ | | |
| _Name_ | | |

## 12. Generative AI declaration

| Tool | Purpose | Where used |
|---|---|---|
| _e.g. Claude_ | _e.g. README structure, debugging help_ | _e.g. README, src/estimation_ |

## 13. References

_Tutorials, papers, libraries used._
