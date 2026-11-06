# TurtleBot3 Visual Waypoint Navigation and ArUco Docking

**41014 Sensors and Control in Mechatronics, Integrated Project, Track A**

| | |
|---|---|
| **Group** | _Group 99_ |
| **Members** | _Muhammad Suhaib Khan (1351525)_, _Jaden Gambrill(26065197)_, _Siale Smythe (24831226)_ |
| **Track** | A: mobile robot, visual waypoint navigation (MuJoCo) |
| **Hardware** | _Simulation only_ |
| **Demo video** | _[link, 5 min]_ |

---

## 1. Overview

_The robot navigates a room with obstacles using 2D LiDAR, then visually docks to the ArUco-marked charging station. An EKF fuses wheel-encoder odometry with camera measurements of the marker (range/bearing or pose). A state-feedback/LQR tracker acts on the **estimated** state only._

   ![Pipeline](docs/figures/pipeline_diagram.png)

## 2. Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/HayWire737/JMS_IntegratedProject41014.git
cd JMS_IntegratedProject41014
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_scene.py      # generate models/scene.xml from the config
python -m pytest                   # model checks + ground-truth policy
python scripts/sensor_check.py     # encoders, LiDAR, camera and ArUco working
python scripts/dead_reckoning.py   # open-loop odometry drift vs truth
python scripts/run.py              # closed loop with the EKF, then visual docking (--no-dock to stop at the last waypoint)
python scripts/dock_study.py       # docking over 5 seeds, scored against ground truth
python scripts/compare_controllers.py   # P vs LQR on the route, 5 seeds
python scripts/sweep_lqr_weights.py     # LQR weight and v_ref sweep (about 15 min)
python scripts/lqr_design_report.py     # LQR gains and closed-loop poles
python scripts/analyse_model.py    # Part A controllability/observability
python scripts/ekf_study.py        # Part C repeated trials and R trade-off
python scripts/run.py --view       # same, with the 3D viewer (macOS: use mjpython)
```

Figures are written to `results/`. All tunable values are in `configs/nominal.yaml`.

**Windows PowerShell:** activate with `.venv\Scripts\Activate.ps1`. On Python 3.15, PyYAML has no prebuilt package; use Python 3.13, or set `$env:PYYAML_FORCE_LIBYAML = "0"` before `pip install`.

Tested on: Windows 11 Home (build 26200), Python 3.15.0, MuJoCo 3.17.0, NumPy 2.5.4, SciPy 1.18.1, opencv-contrib-python 5.0.0.93.

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
  control/        Go-to-goal baseline, LQR tracker, visual docking
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
| Simulation scene, sensors | Working: independent noise stream per sensor, LiDAR max-range fix, optional camera blur | Jaden |
| ArUco detection, range/bearing | Working: calibrated intrinsics, range-bias correction, edge-dropout fix | Jaden |
| Sensor noise characterisation, frames | Done | Jaden |
| Wheel odometry, dead reckoning | Working (baseline) | Muhammad |
| Model, Jacobians, controllability/observability | Done | Muhammad |
| Go-to-goal controller, waypoints | Working (baseline) | Muhammad |
| EKF (pose model, calibrated Q/R, gating) | Done | Muhammad |
| LQR tracker | Done, default controller: `src/control/lqr.py`; stable at every speed, weight sweep and P comparison in Section 7 | Siale |
| LiDAR obstacle avoidance | Working with both controllers: `src/planning/avoidance.py` | Jaden |
| Visual docking | Done: `src/control/docking.py`, 5/5 seeds with LQR and with P | Siale |
| Evaluation matrix | Done: `scripts/robustness_study.py`, 8 conditions × 5 seeds with P and LQR, 80/80 docked (Section 8) | Jaden |

## 4. Modelling (Part A)

Full derivations and discussion: [`docs/part_A_C_notes.md`](docs/part_A_C_notes.md). Code: `src/model/diff_drive.py`, analysis: `scripts/analyse_model.py`.

- **State** x = [x, y, θ]ᵀ (base_link in world). **Input** u = [v, ω]ᵀ.
- **Continuous model** (differential drive): ẋ = v cos θ, ẏ = v sin θ, θ̇ = ω, with v = r(ω_R+ω_L)/2, ω = r(ω_R−ω_L)/b, **r = 0.033 m, b = 0.160 m** (TurtleBot3 Burger).
- **Discrete odometry model** (Δt = 0.05 s, midpoint rule, m = θ + Δθ/2): x⁺ = x + Δs cos m, y⁺ = y + Δs sin m, θ⁺ = θ + Δθ, driven by encoder increments Δs, Δθ.
- **Jacobians** F = ∂f/∂x and G = ∂f/∂[Δs, Δθ], derived by hand and verified against finite differences in `tests/test_model.py`.
- **Assumptions:** planar motion, no slip, known r and b, kinematic (velocity-controlled wheels settle in ~0.3 s). Slip and radius error appear as odometry drift and are absorbed in Q.
- **Controllability** of the linearised tracking-error model (A(v_r, ω_r), B): rank 3 whenever the reference moves, **rank 2 at rest** (no sideways motion, nonholonomic); σ_min falls as v_r → 0.
- **Observability:**

| Measurements | Rank of observability matrix |
|---|---|
| Odometry only | 0: absolute pose unobservable, drift unbounded |
| Range + bearing to one marker (any motion) | **2**: rotation of the robot about the marker, null vector [dy, −dx, 1], is never observable |
| Marker pose (x, y, yaw) | **3** from a single measurement |

The pose model is therefore used. Observability weakens with range (yaw noise ∝ r²): one measurement constrains the weakest direction to ~1.4 cm at 0.6 m but ~80 cm at 3 m.

![Observability vs range](docs/figures/part_A_observability.png)

Verified experimentally: starting 0.15 rad wrong along the rotation direction, the range+bearing EKF never corrects it (+0.17 rad after 12 s, 3 seeds); the pose EKF corrects it to < 0.004 rad within ~2 s.

![Observability experiment](docs/figures/observability_experiment.png)

## 5. Sensing and calibration (Part B)

Scripts in Quick start; results in `results/`. Ground truth is used only offline, as a surveyed reference.

### Sensors

| Sensor | Simulated as | Rate | Resolution |
|---|---|---|---|
| Wheel encoders | Tick random walk (σ = 1 tick/step), wheel radius error ±0.2 % | 20 Hz | 4096 ticks/rev (0.051 mm) |
| 2D LiDAR | 72 rangefinders, σ = 1 cm | 20 Hz | 5°, 3.5 m max |
| Camera | Pixel noise σ = 8 grey levels | 5 Hz | 640 × 480, 60° × 75° FOV |
| ArUco marker | `DICT_4X4_50`, 0.15 m, centre 0.12 m high | | |

### Frames

![Frame diagram](docs/figures/frame_diagram.png)

camera → base_link (0.07 m forward, 0.12 m up) → world (EKF estimate) ↔ marker (4.84, 2.0 m, facing 180°). LiDAR → base_link (0.02 m behind) via `scan_to_base()`. No separate odom frame: the start pose is known. `tests/test_frames.py` pushes a real scan into the world frame: median 5.3 mm from the walls (12.2 mm if the mount offset is ignored).

### Noise (`characterise_noise.py`)

| | Measured | Predicted |
|---|---|---|
| Encoder distance / step | σ = 0.044 mm | 1 tick = 0.051 mm |
| Encoder heading / step | σ = 4.8 × 10⁻⁴ rad | 4.5 × 10⁻⁴ rad |
| Heading drift (10 seeds) | **−0.024 ± 0.003 rad/m** | **−0.025 rad/m** (radius error) |
| LiDAR | σ = 9.98 mm, bias 0.2 mm | 10 mm |
| Camera range (corrected) | RMS 1.5 mm < 1 m, 3–5 mm at 1–2 m, 17 mm at 2–3 m | |
| Camera bearing | RMS 0.02–0.06° | 1 px = 0.14° |

A single run can be ±13 % off the drift, so encoder figures are averaged over 10 seeds. The drift is a bias, so Q must be inflated above the per-step σ (Part C).

### Camera calibration (`calibrate_camera.py`)

Checkerboard (9 × 6, 40 mm) shown in 160 random poses with camera noise; outlier views rejected.

| | Calibrated (83 views, 0.093 px RMS) | Ideal | Old nominal |
|---|---|---|---|
| f_x, f_y | 415.31, 415.33 (± 0.21) | 415.69 | 415.69 |
| c_x, c_y | **319.30, 239.54** (± 0.2) | 319.5, 239.5 | **320, 240** |
| Distortion | < 0.003 | 0 | 0 |

The old nominal K put the image centre at (320, 240); OpenCV's pixel-centre convention gives (319.5, 239.5). The half-pixel error biased every bearing by 0.069°. One calibration pins c_x to about ±0.25 px, so this halves the bias rather than removing it.

### Marker detection (`detection_envelope.py`)

| Reliable detection (≥ 95 %) | Value |
|---|---|
| Range, head-on | 0.25–4.4 m |
| Bearing off image centre | 15° at 0.3 m, 30° at 1 m, 34° at 2 m |
| View angle off marker normal | 70° at 0.5 m, 60° at 1–2 m |

These set the docking stop distance (0.30 m) and bearing tolerance (Part D).

### Range bias correction (`calibrate_marker_range.py`)

The marker read too far away, growing with distance. Not the intrinsics (identical with either K): ArUco's "L" corners refine inward; checkerboard "X" corners don't. Fitted bias(d) = a₂d² + a₃d³, tested on 120 **held-out** random poses:

| Distance | Bias before → after | RMS before → after |
|---|---|---|
| 1–2 m | +4.9 → +0.3 mm | 7.0 → 4.5 mm |
| 2–3 m | +33.9 → +7.7 mm | 39.2 → 16.4 mm |
| 3–4.5 m | +67.6 → −26.7 mm | 98.3 → 63.7 mm |

Within the EKF's 3 m cut-off the bias is essentially gone; beyond it, pose-to-pose scatter dominates.

### LiDAR obstacle avoidance (`src/planning/avoidance.py`)

Checks a corridor (robot half-width + 12 cm) towards the goal in the scan; if blocked, places a temporary waypoint 0.5 m along the closest clear heading and holds it until reached. Works with both controllers. Routes straight through an obstacle: 0/5 without avoidance (stranded by the safety stop), 5/5 with it, ≥ 10 cm body clearance. Start → dock needs no hand-placed waypoints.

### Limitations

- The simulated camera is an exact pinhole, so calibration validates the method but had no distortion to correct.
- The range correction ignores view angle; residual bias is 1.5–2× larger at 20° off the marker normal.
- Results are for nominal noise without blur; blur is a Part E condition.

## 6. Estimation (Part C)

Code: `src/estimation/ekf.py`. Discussion of every choice: [`docs/part_A_C_notes.md`](docs/part_A_C_notes.md).

- **Filter:** EKF on [x, y, θ]. Predict at 20 Hz from encoder increments (P ← FPFᵀ + GQGᵀ); update at up to 5 Hz from the ArUco marker pose in base_link; Joseph-form covariance update; angle innovations wrapped.
- **Measurement model:** h(x) = [Rᵀ(θ)(m − p); θ_m − θ], H derived by hand, finite-difference tested.
- **Q (process noise):** per-wheel variance q0 + k|d|, propagated to [Δs, Δθ]. Fitted to step residuals, then scaled ×10.3 so predict-only NEES = 3 (the per-step fit misses time-correlated errors such as wheel-radius bias). **q0 = 8.6e-8 m², k = 4.5e-7 m.** (`scripts/calibrate_odometry_noise.py`)
- **R (measurement noise):** calibrated from 3147 detections at 135 known poses; σ(r) = a + b r² per component, e.g. depth 5.6 mm at 1 m → 5.0 cm at 3 m, yaw 0.03 → 0.25 rad. Readings beyond the calibrated 3 m are ignored. (`scripts/calibrate_marker_noise.py`)

![R calibration](docs/figures/marker_noise_calibration.png)

- **Outliers:** χ² gate at 99 % (NIS > 11.34 rejected); after 5 consecutive rejections P is inflated ×100 (divergence recovery, see failure analysis).
- **Marker loss:** predict-only; covariance grows with distance travelled and collapses at the next reading:

![EKF error vs 3-sigma](docs/figures/run_ekf_bounds.png)

**Results** (5 seeds, closed loop): see Section 8.

## 7. Control (Part D)

Code: `src/control/lqr.py` (tracker), `src/control/docking.py` (docking). Full design notes and every number: [`docs/lqr_design.md`](docs/lqr_design.md). Scripts: `compare_controllers.py` (P vs LQR), `sweep_lqr_weights.py` (weights, v_ref), `lqr_design_report.py` (gains, poles), `dock_study.py` (docking). All runs use the EKF estimate in the loop and are scored against ground truth in `evaluation/metrics.py`. **Default controller: LQR** (`controller.type: lqr`); the P go-to-goal controller is kept as the baseline.

### Controller design

- **Reference.** Each waypoint pair defines a segment. A reference point moves along it at v_r: cruise v_ref = 0.15 m/s, slowing into the waypoint (v_r ≤ 1.0 s⁻¹ × remaining length, floor 0.03 m/s). It slows down when the robot lags (it stops at 0.30 m position error or 0.80 rad heading error), so it cannot run away from a held or pushed robot.
- **Error model** (robot frame, Part A, `error_dynamics`): e = Rᵀ(θ)(p_ref − p), e_θ = θ_ref − θ, linearised about (v_r, ω_r = 0) with δu = u_ref − u:
  ė = A e + B δu,  A = [[0, ω_r, 0], [−ω_r, 0, v_r], [0, 0, 0]],  B = [[1, 0], [0, 0], [0, 1]].
- **Discrete LQR.** Zero-order hold at Δt = 0.05 s, discrete Riccati equation, K = (R + B_dᵀPB_d)⁻¹B_dᵀPA_d. Gains are computed at max(v_r, v_lin_min = 0.05 m/s) because the lateral error is uncontrollable at rest (rank 2, Part A), and cached on a 1 cm/s grid.
- **Control law.** δu = −K e, so v = v_r cos e_θ + (Ke)₁ and ω = (Ke)₂. For heading errors above 0.60 rad the robot turns in place (heading gain only) until the error is below 0.15 rad; the two thresholds stop it switching back and forth.
- **Limits.** 0 ≤ v ≤ 0.22 m/s (never reverse: the LiDAR safety stop only looks ahead), |ω| ≤ 2.84 rad/s (1.5 rad/s when turning in place), |a| ≤ 0.5 m/s², |α| ≤ 4 rad/s². At the nominal setting ω never reaches its limit (0 % of steps) and the robot turns in place for 5.8 % of the route (5 seeds), so saturation does not shape the nominal tracking result.

### Weights, gains and stability

**Q_lqr = diag(30, 60, 8)** on [e_x (m), e_y (m), e_θ (rad)], **R_lqr = diag(4, 1)** on [δv (m/s), δω (rad/s)]. At v_ref:

K = [[2.558, 0, 0], [0, 7.149, 2.993]]   (rows δv, δω; columns e_x, e_y, e_θ)

- **Along-track** (e_x → δv) is decoupled: Q_x and R_v only set one gain and one pole (0.872, τ = 0.37 s).
- **Cross-track and heading** (e_y, e_θ → δω) are coupled. Their slow mode is path convergence, with time constant 7.29 s at 0.05 m/s, 2.41 s at v_ref and 1.62 s at v_max: **an offset decays over a fixed ~0.36 m of travel at any speed**.
- **Only the ratio Q/R sets the gain**: Q ×4 and R ×0.25 give the same K to 8 × 10⁻³¹.
- **Stable at every speed.** All closed-loop poles of A_d − B_dK are real and inside the unit circle from v = 0 (computed at v_lin_min) to v_max; the largest is 0.9932 at v_lin_min.

![LQR poles](docs/figures/lqr_poles.png)

### Q/R trade-off (`sweep_lqr_weights.py`)

One-at-a-time scaling of each weight by 0.25, 0.5, 2 and 4, all of Q scaled together, and v_ref from 0.10 to 0.22 m/s; 5 seeds per point. Every setting finished every run.

![LQR weight sweep](docs/figures/lqr_weight_sweep.png)

| Setting | True cross-track RMS [cm] | ω jitter [rad/s/step] | Route time [s] |
|---|---|---|---|
| **Current** | **2.95 ± 0.44** | 0.044 | 38.5 |
| Q_y ×0.25 / ×4 | 4.44 ± 0.44 / 3.63 ± 0.56 | 0.036 / 0.043 | 38.2 / 37.7 |
| Q_θ ×0.25 / ×4 | 2.21 ± 0.33 / 6.27 ± 0.44 | 0.034 / 0.041 | 38.3 / 37.9 |
| R_ω ×4 / ×0.25 | 2.41 ± 0.19 / 5.68 ± 0.42 | 0.032 / 0.044 | 38.7 / 37.5 |
| v_ref 0.10 / 0.22 m/s | 3.20 ± 0.48 / 3.07 ± 0.36 | 0.033 / 0.048 | 54.7 / 28.1 |

**Justification and where the trade-off sits.** The weights were set before the sweep; the sweep shows:
- **Q_y = 60** is at or near the best value: lower or higher is worse.
- **Q_x, R_v** have no measurable effect on tracking (2.92–3.07 cm across ×0.25 to ×4).
- **Q_θ = 8, R_ω = 1** sit on the stiff side. A softer heading loop gave lower true cross-track *and* lower jitter on this route (Q_θ ×0.25: 2.21 cm; R_ω ×4: 2.41 cm), at the cost of more turning in place (8 % vs 6 % of the route). These are 5-seed results and were not adopted: they would also change docking (which inherits these weights) and slow convergence after a push or a detour, which has not been re-tested.
- **v_ref = 0.15 m/s** is a speed/smoothness compromise. Accuracy is flat from 0.10 to 0.22 m/s while route time halves and jitter rises. At v_ref = v_max the forward command has no headroom to catch up with the reference.

**Heavier weights are worse, and not because of tracking.** Over 20 seeds the current setting gives 3.06 ± 0.37 cm, all Q ×3 gives 5.63 ± 0.47 cm and all R ×0.25 gives 6.19 ± 0.48 cm. The controller follows its estimate equally well in all three (estimated-path cross-track ≈ 2 cm); what grows is the EKF error (2.97 → 5.42 / 6.05 cm). On the first segment no marker reading is usable, so the estimate there is pure odometry (EKF and dead reckoning give identical error at 12 s). With dead reckoning in the loop (10 seeds), straight-line heading drift is the same (−1.37 °/m) but the stiff controller rotates more (217° vs 178° over the route) and picks up 0.75° more heading error at the start and 2.7° more through the first corner, which is then carried for ~2 m before the marker is usable. Why odometry error during a rotation depends on how the turn is driven was not isolated.

### LQR vs P (`compare_controllers.py`, 5 seeds, nominal route, docking off)

| Controller | Finished | Route time [s] | True cross-track RMS [cm] | Max [cm] | Estimated-path cross-track [cm] | RMS ω [rad/s] | ω jitter [rad/s/step] |
|---|---|---|---|---|---|---|---|
| P | 5/5 | **25.3 ± 0.1** | **2.73 ± 0.21** | **6.4 ± 0.8** | 2.79 ± 0.17 | 0.595 | 0.182 |
| LQR | 5/5 | 38.5 ± 0.2 | 2.95 ± 0.44 | 8.2 ± 0.3 | **2.17 ± 0.31** | **0.302** | **0.044** |

**LQR is smoother, not more accurate, and slower.** It uses half the yaw rate and a quarter of the yaw-command jitter, and follows the route more closely *as the estimator sees it*, but its true cross-track is the same as P's within noise and its worst deviation is larger. True cross-track on this route is dominated by estimation error (above), not by either controller. It takes 13 s longer because it cruises at 0.15 m/s while P runs faster (RMS speed 0.135 vs 0.206 m/s). LQR is the default for its smoothness, its explicit effort/accuracy weighting and its behaviour under odometry faults (Section 8: smaller worst-case error under wheel slip).

### Visual docking (`src/control/docking.py`, settings under `docking:`)

Takes over after the last waypoint and works from marker detections and wheel odometry only, not the EKF or the marker's map pose.
- *Relative pose.* Each detection gives the marker pose in base_link; inverting it gives the robot pose in the marker frame (x out of the marker face). Between the 5 Hz camera frames this pose is carried forward with the odometry increments.
- *Approach.* A second LQR tracker (slower: v_ref 0.08 m/s) follows the marker normal from the robot's projection on it to the stop point 0.30 m from the marker. Ending on the normal, facing the marker, centres the marker in the image.
- *Decision.* Made on fresh detections only: the robot stops on the first frame that reads within 1 cm of the stop distance, and is docked if the marker is then within 10° of the image centre, otherwise the attempt fails (the controller never reverses). If odometry says the stop point is reached, it holds still until a frame confirms it.
- *Marker loss.* After 2 s without a detection it rotates in place towards the last known bearing; docking gives up after 30 s.
- *Tolerance.* Success is 0.25–0.35 m from the marker (base_link to marker centre) with the marker within ±15° of the image centre, judged against ground truth. The 0.30 m target sits above the closest reliable detection range (0.25 m, Part B), and 15° is the bearing limit measured at 0.3 m.

**Docking results** (`dock_study.py`, 5 seeds, full mission, mean ± std against ground truth):

| Route controller | Success | Final range [m] | Image bearing [°] | Lateral offset [cm] | Heading error [°] | Docking time [s] | Mission time [s] |
|---|---|---|---|---|---|---|---|
| **LQR (default)** | **5/5** | 0.310 ± 0.001 | −0.03 ± 0.05 | −0.1 ± 0.1 | −0.09 ± 0.12 | 5.7 ± 0.0 | 44.1 ± 0.2 |
| P | 5/5 | 0.311 ± 0.001 | +0.09 ± 0.07 | 0.0 ± 0.1 | −0.01 ± 0.07 | 5.6 ± 0.1 | 30.9 ± 0.1 |

The robot stops about 1 cm beyond the target because it stops on the first frame inside the 1 cm tolerance, well inside the allowed band; its own camera reading at the stop agrees with truth to 2–3 mm. Cross-track offsets decay over ~0.36 m and the final approach is only 0.24 m long, so docking relies on the route ending on the marker normal; an offset at the last waypoint is only partly removed (unit tests: 6 cm → about 3.5 cm).

## 8. Results (Parts C, D, E)

### Estimation (Part C), 5 seeds, mean ± std

| Condition | Pos. RMSE [m] | Final pos. error [m] | Mean NEES (ideal 3) | Rejected / run |
|---|---|---|---|---|
| Dead reckoning | 0.056 ± 0.004 | 0.149 ± 0.008 | – | – |
| EKF range+bearing | 0.023 ± 0.002 | 0.005 ± 0.000 | 2.25 ± 0.13 | 0.0 |
| EKF pose, R ×0.1 | 0.028 ± 0.004 | 0.010 ± 0.006 | 10.68 ± 2.73 | 13.0 |
| **EKF pose, calibrated R** | **0.023 ± 0.002** | **0.007 ± 0.002** | **2.45 ± 0.13** | **1.4** |
| EKF pose, R ×10 | 0.023 ± 0.002 | 0.006 ± 0.000 | 1.37 ± 0.09 | 0.0 |

The EKF reduces final position error ~20× vs dead reckoning. Underestimating R makes the filter overconfident (NEES 10.7) and it rejects genuine readings; overestimating makes it underconfident.

![R trade-off](docs/figures/ekf_study_tradeoff.png)

### Nominal case (full pipeline)
| Metric | Value |
|---|---|
| Position RMSE (estimate vs truth) | 0.023 m |
| Heading RMSE | 0.022 rad |
| True cross-track RMS (route) | 2.95 ± 0.44 cm (LQR), 2.73 ± 0.21 cm (P) |
| Docking final range | 0.310 ± 0.001 m (LQR; target 0.30 m, allowed 0.25–0.35 m) |
| Docking final image bearing | −0.03 ± 0.05° (LQR; allowed ±15°) |
| Docking success rate | 5 / 5 (LQR), 5 / 5 (P) |

Position and heading RMSE are for the route only with the P controller, as in the Part C study (`ekf_study.py` pins docking off and `controller.type: p`). Tracking and docking: Section 7 (5 seeds each).

With the P controller on seed 0, during docking the EKF's χ² gate rejected 5 of the close-range (≈0.3 m) marker readings (NIS 15–58, none consecutive enough to trigger recovery) and mean NEES over the full run rose from 2.4 to 5.6: R appears too small at this range. Docking does not use the EKF, and the estimate stayed within 7 mm of truth.

### Robustness (Part E)

Full mission (LiDAR navigation, then visual docking), nominal config plus **one** change per row, 5 seeds each. `python scripts/robustness_study.py --controller p` and `--controller lqr`.

| Condition | Docked (P / LQR) | Est. RMSE P / LQR [cm] | Worst est. error P / LQR [cm] | NEES P / LQR |
|---|---|---|---|---|
| Nominal | 5/5 · 5/5 | 2.0 / 3.0 | 5 / 18 | 8.0 / 6.3 |
| 2× sensor noise | 5/5 · 5/5 | 2.2 / 3.1 | 17 / 24 | 10.5 / 7.2 |
| Camera blur (1 px) | 5/5 · 5/5 | 2.1 / 3.5 | 16 / 31 | 7.1 / 9.0 |
| Marker occluded 22–28 s | 5/5 · 5/5 | 2.0 / 2.7 | 5 / 13 | 9.8 / 3.5 |
| Wheel radius bias ×5 | 5/5 · 5/5 | 22.1 / 20.5 | 58 / 51 | 36.3 / 21.2 |
| Wheel slip (left +20 %, 1 s) | 5/5 · 5/5 | 21.2 / 16.3 | 77 / 46 | 37.2 / 13.2 |
| External push (8 N, 0.3 s) | 5/5 · 5/5 | 4.6 / 3.0 | 16 / 12 | 9.8 / 7.8 |
| Varied start pose | 5/5 · 5/5 | 5.3 / 3.0 | 33 / 18 | 8.2 / 5.0 |

![Robustness, P](docs/figures/robustness_p.png)

![Robustness, LQR](docs/figures/robustness_lqr.png)

*The same study with the LQR controller: docking success and EKF position error (RMSE and worst case) per condition.*

**80/80 docked.** Noise, blur, occlusion and pushes barely affect the mission. Odometry faults (wheel bias, slip) are the weak point: see failure analysis.

### Trade-offs

**LQR weights:** a stiffer heading loop (Q_θ ×4 or R_ω ×0.25) leaves ω jitter about the same (0.041 / 0.044 vs 0.044 rad/s/step) and roughly doubles true cross-track (6.27 / 5.68 vs 2.95 cm); a softer one (Q_θ ×0.25 or R_ω ×4) lowers both (2.21 / 2.41 cm, jitter 0.034 / 0.032) but turns in place more (Section 7).

**P vs LQR:** P finishes the full mission in 31 s vs 44 s. True cross-track on the route is the same within noise (2.73 ± 0.21 vs 2.95 ± 0.44 cm, Section 7); nominal EKF position RMSE in the table above is lower with P (2.0 vs 3.0 cm). LQR degrades more gracefully under odometry faults (worst EKF position error under slip 46 vs 77 cm, NEES 13 vs 37).

### Failure analysis

**Gate lockout (estimator).** The first EKF was worse than dead reckoning (final error 18 cm vs 15 cm, 37 of 62 updates rejected). Logging every innovation showed that marker readings from 4.3–4.8 m, beyond the calibrated range, were biased ~15 cm in depth but accepted; they shifted the estimate ~10 cm and shrank P, after which correct close-range readings had NIS 30–45 and were all gated out. Fix: ignore readings beyond the calibrated range and add divergence recovery. Result: final error 0.5 cm, NEES 2.5.

**Unobservable direction (model).** Range+bearing cannot correct initial error along the rotation about the marker (Part A); the pose measurement model fixes this.

**Overconfident EKF after odometry faults (integration).** With wheel bias ×5 or slip the EKF error reaches 50–77 cm (NEES ≈ 35), yet every trial still docks. Cause: from ~3 s to 19 s no marker reading is usable (beyond the 3 m cut-off, then out of view), so the EKF runs on odometry alone; Q only covers the nominal ±0.2 % wheel error, so P stays small while the error grows. When the marker returns, the χ² gate rejects the correct readings (19 rejections). The divergence recovery (P inflated after 5 rejections) then accepts them and the error falls below 1 cm within 6 s. Weak points: single-marker coverage, and Q tuned for nominal conditions only.

**Coupled sensor noise (simulation).** Enabling camera noise changed odometry drift by 20 %: all sensors shared one random generator. Fixed with one stream per sensor (`tests/test_sim_noise.py`).

**Phantom LiDAR obstacles (simulation).** MuJoCo clamps long beams to 3.5 m instead of returning nothing, making a ring of fake obstacles (13 of 72 beams). Max-range returns are now dropped; worst scan point went from 607 mm to 23 mm off a surface (`tests/test_frames.py`).

**Marker dropouts at the image edge (sensing).** OpenCV's default `minDistanceToBorder = 3 px` rejected fully visible markers. Setting it to 0 widened the usable bearing at 0.3 m from 10° to 15° (`tests/test_aruco.py`).

**Half-pixel principal point and range bias (sensing).** Bearings were offset 0.069° and long-range readings up to 7 cm long; both diagnosed and corrected in Part B.

## 9. Hardware (optional)

_Setup, differences from simulation, quantitative sim-vs-hardware comparison._

## 10. Limitations and future work

_What the results do and do not support._

### Control

- **LQR vs P.** LQR is smoother (a quarter of P's yaw-command jitter) but not more accurate against truth (2.95 ± 0.44 vs 2.73 ± 0.21 cm true cross-track, the same within noise) and 13 s slower on the route. The results support LQR as the smoother controller and the more graceful one under odometry faults (Section 8); they do not show that it tracks better.
- **Tracking error mostly reflects estimation error.** With every tested weight the controller follows its estimate to about 2 cm; true cross-track grows with the odometry error picked up before the marker is usable. The metric does not separate controller quality from estimation quality.
- **Heavier-weight result only partly explained.** Stiffer weights rotate more and add odometry heading error in the turns (measured), but why the way a turn is driven changes odometry error was not isolated.
- **Docking depends on the route ending on the marker normal.** Offsets decay over about 0.36 m and the final approach is 0.24 m, so an offset at the last waypoint is only partly removed (6 cm → about 3.5 cm in unit tests). 5/5 docking is shown for the nominal route end, not for arbitrary approach poses.
- **Narrow evidence base.** The weight and v_ref sweeps use one route and 5 seeds per point; only the heavier-weight check used 20 seeds.
- **Q/R tuned for nominal conditions.** The weights were chosen and swept on the nominal route; the robustness study used them unchanged.
- **Future work.** Check the softer heading weights (Q_θ ×0.25, R_ω ×4) over 20 seeds, then re-run docking, the avoidance test and the robustness study before adopting them; isolate the rotation–odometry effect; test docking from offset last-waypoint poses.

## 11. Individual contributions

| Member | Led | Contributed to |
|---|---|---|
| Muhammad Suhaib Khan (1351525) | Part A (modelling), Part C (EKF) | Project page, repository setup |
| Jaden Gambrill (26065197) | Part B (sensing) and the simulation enviroment | Repository workflow (branch protection, reviews) |
| Siale Smythe (24831226) | Part D (control) | LQR tracker, visual docking, P vs LQR comparison, Q/R/v_ref sweep and design report, control tests |

## 12. Generative AI declaration

| Tool | Purpose | Where used |
|---|---|---|
| _e.g. Claude_ | _e.g. README structure, debugging help_ | _e.g. README, src/estimation_ |

## 13. References

_Tutorials, papers, libraries used._
