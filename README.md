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

_Insert pipeline block diagram: sensors → EKF → planner → LQR → MuJoCo, with ground truth branching to the evaluator only._

## 2. Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_scene.py      # generate models/scene.xml from the config
python -m pytest                   # model checks + ground-truth policy
python scripts/sensor_check.py     # encoders, LiDAR, camera and ArUco working
python scripts/dead_reckoning.py   # open-loop odometry drift vs truth
python scripts/run.py              # closed loop with the EKF, then visual docking (--no-dock to stop at the last waypoint)
python scripts/dock_study.py       # docking over 5 seeds, scored against ground truth
python scripts/analyse_model.py    # Part A controllability/observability
python scripts/ekf_study.py        # Part C repeated trials and R trade-off
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
| Wheel odometry, dead reckoning | Working (baseline) | _name_ |
| Model, Jacobians, controllability/observability | Done | _name_ |
| Go-to-goal controller, waypoints | Working (baseline) | _name_ |
| EKF (pose model, calibrated Q/R, gating) | Done | _name_ |
| LQR tracker | Working: `src/control/lqr.py` | Siale |
| LiDAR obstacle avoidance | Working: `src/planning/avoidance.py` | _name_ |
| Visual docking | Working: `src/control/docking.py`, 5/5 seeds | Siale |
| Evaluation matrix | TODO | _name_ |

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

All Part B numbers below are produced by the scripts listed in Quick start and committed in `results/`. Ground truth is used only in these offline calibration scripts, the way a real lab would survey a calibration target; nothing under `src/` reads it.

### Sensors

| Sensor | Model in simulation | Rate | Resolution / range |
|---|---|---|---|
| Wheel encoders | Joint angle → ticks, random-walk tick noise (σ = 1 tick/step), wheel radius error ±0.2 % (unknown to the estimator) | 20 Hz | 4096 ticks/rev = 0.051 mm of wheel travel per tick |
| 2D LiDAR | 72 MuJoCo rangefinders in the scan plane, Gaussian range noise σ = 1 cm | 20 Hz | 5° beam spacing, 3.5 m max range |
| Camera | MuJoCo render, Gaussian pixel noise σ = 8 grey levels (optional blur) | 5 Hz | 640 × 480, 60° vertical / 75.2° horizontal FOV |
| ArUco marker | `DICT_4X4_50`, id 0, 0.15 m black square on the dock, centre 0.12 m high | | |

### Frames

![Frame diagram](docs/figures/frame_diagram.png)

The chain from measurement to control is: **camera** (OpenCV optical frame: z forward, x right, y down) → **base_link** (fixed mount: 0.07 m forward, 0.12 m up; z_cam = x_base, x_cam = −y_base, y_cam = −z_base) → **world** (through the EKF estimate [x, y, θ]), compared with the known **marker** pose (4.84, 2.0) m facing 180°. LiDAR points go **lidar** → **base_link** (0.02 m behind, axes parallel) with `scan_to_base()`. There is no separate odom frame: the start pose is known, so odometry is integrated directly in the world frame and the EKF corrects that estimate.

All mounts live in `configs/nominal.yaml`; `tests/test_scene_matches_config` fails if the MuJoCo scene and the config disagree, and `tests/test_frames.py` checks the transforms end to end: a real LiDAR scan taken from a known pose and pushed through lidar → base_link → world lands a median **5.3 mm** from the walls, against 12.2 mm if the 2 cm mount offset is ignored.

### Noise characterisation (`scripts/characterise_noise.py`)

| Sensor | Measured | Predicted from first principles |
|---|---|---|
| Encoder distance, per 0.05 s step | σ = 0.044 mm | one tick of wheel travel = 0.051 mm |
| Encoder heading, per step | σ = 4.8 × 10⁻⁴ rad | √2 × 0.051 mm / 0.16 m = 4.5 × 10⁻⁴ rad |
| Heading drift driving straight | **−0.024 ± 0.003 rad/m** (10 seeds) | **−0.025 rad/m** from the ±0.2 % wheel radius error |
| Turn scale error (spin on the spot) | +0.40 ± 0.06 % | not in the config: a contact effect of the simulated wheels |
| LiDAR range | σ = 9.98 mm, bias 0.2 mm, every beam within 8.8–11.1 mm | 10 mm configured |
| Camera range (after correction) | RMS 1.5 mm below 1 m, 3–5 mm at 1–2 m, 17–18 mm at 2–3 m | grows with distance (see below) |
| Camera bearing | RMS 0.02–0.06° from 1 m out (0.09° at 0.5–1 m) | 1 px = 0.14° at f = 415 px |

Encoder drift is a random walk, so a single 12 s run can land anywhere in roughly ±13 % of the true value; every encoder figure is therefore the mean ± spread over 10 seeds. The heading drift is a **bias**, not random noise, so the EKF's Q has to be inflated above the per-step σ to cover it (Part C).

### Camera intrinsics (`scripts/calibrate_camera.py`)

The camera is calibrated the way a real one would be: a 9 × 6 checkerboard (40 mm squares) is shown to a camera identical to the robot's in 160 random positions and tilts, with the camera noise applied. Views where the board is not fully visible are skipped, views whose fit is worse than 3× the median are rejected, and `cv2.calibrateCamera` is run on the rest.

| Parameter | Calibrated (83 views, 0.093 px RMS) | Ideal pinhole | Previous nominal K |
|---|---|---|---|
| f_x, f_y (px) | 415.31 ± 0.21, 415.33 ± 0.21 | 415.69 | 415.69 |
| c_x (px) | **319.30 ± 0.21** | 319.5 | **320.0** |
| c_y (px) | **239.54 ± 0.19** | 239.5 | **240.0** |
| Distortion k₁, k₂, k₃ | −0.001, 0.003, −0.002 | 0 | 0 |

Focal length agrees with the ideal pinhole to 0.09 % and distortion is negligible, which confirms the pinhole model. The calibration also exposed a bug in the nominal intrinsics: OpenCV puts pixel centres at integer coordinates, so the centre of a 640 × 480 image is (319.5, 239.5), not (320, 240). The half-pixel error offset every bearing by **0.069°** (the constant offset `sensor_check.py` showed with the marker dead ahead). The detector now uses the calibrated values from the config.

**Limit:** five independent calibrations gave c_x between 319.07 and 319.61, so a single calibration pins the principal point to about ±0.25 px, the same size as the error it corrected. The calibrated K halves the bearing bias rather than eliminating it.

### Marker detection (`scripts/detection_envelope.py`)

Detection rate over 5 noisy frames per pose ("reliable" = ≥ 95 %):

| | Value | Used by |
|---|---|---|
| Closest reliable range, head-on | **0.25 m** | Docking stop distance (Part D) |
| Furthest reliable range | 4.4 m (the whole room) | EKF cut-off at 3 m is set by accuracy, not detection |
| Max bearing off image centre | 15° at 0.3 m, 24° at 0.5 m, 30° at 1 m, 34° at 2 m | Final-approach heading tolerance |
| Max view angle off the marker normal | 70° at 0.5 m, 60° at 1–2 m | Approach directions that keep the dock visible |

### Range bias correction (`scripts/calibrate_marker_range.py`)

The marker always read too far away: ~1 mm at 1 m, +34 mm at 2–3 m, +68 mm beyond 3 m. The cause is not the intrinsics (it is identical with nominal or calibrated K) but where the marker's outer corners are located: the detected outline sits slightly inside the true edge, so the marker looks smaller and further away. Checkerboard corners are symmetric "X" corners and refine without bias; ArUco outer corners are "L" corners and do not. All four OpenCV corner-refinement methods were compared and none is unbiased (sub-pixel is the best; AprilTag over-corrects and loses 9 % of detections).

The bias is fitted as **bias(d) = a₂d² + a₃d³** (d = camera-to-marker distance; d² from pinhole geometry, d³ because the corner offset itself grows with distance) on a grid of poses, then tested on **120 different random poses with different noise**:

| Distance | Bias before → after | RMS before → after |
|---|---|---|
| 0.3–1 m | −0.6 → −1.0 mm | 1.2 → 1.2 mm |
| 1–2 m | +4.9 → **+0.3 mm** | 7.0 → **4.5 mm** |
| 2–3 m | +33.9 → **+7.7 mm** | 39.2 → **16.4 mm** |
| 3–4.5 m | +67.6 → −26.7 mm | 98.3 → 63.7 mm |

The correction scales the camera-to-marker vector, so bearings are unchanged. Within the EKF's 3 m cut-off the bias is essentially removed and what remains is frame-to-frame noise, which is what R should model. Beyond 3 m the scatter between poses dominates, which independently supports the 3 m cut-off. It does not change the nominal run, because the waypoint route only sees the marker at 0.5–2.1 m where the bias was already small; its value is in letting R shrink at 2–3 m.

### Limitations

- The simulated camera is an exact pinhole by construction, so the calibration validates the **method** and found a real convention bug, but there was no lens distortion to correct. A real camera would also need a distortion check.
- The range correction depends on range only. The residual bias depends on view angle too (roughly 1.5–2× larger at 20° off the marker normal than head-on, beyond 2 m).
- Characterisation and detection limits are for the nominal noise (σ = 8 grey levels, no blur). Blur (`camera_blur_px`) cuts long-range detection sharply; that is a Part E condition.

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

- **Controller:** _LQR on tracking error / state feedback_
- **Weights / gains:** _Q_lqr, R_lqr values and how they were chosen_
- **Limits:** _velocity and acceleration saturation, effect on tracking_
- **Docking approach** (`src/control/docking.py`, settings under `docking:` in the config): takes over after the last waypoint and works from marker detections and wheel odometry only, not the EKF or the marker's map pose.
  - *Relative pose.* Each detection gives the marker pose in base_link; inverting it gives the robot pose in the marker frame (x out of the marker face). Between the 5 Hz camera frames this pose is carried forward with the odometry increments.
  - *Approach.* A second LQR tracker (slower: v_ref 0.08 m/s) follows the marker normal from the robot's projection on it to the stop point 0.30 m from the marker. Ending on the normal, facing the marker, centres the marker in the image.
  - *Decision.* Made on fresh detections only: the robot stops on the first frame that reads within 1 cm of the stop distance, and is docked if the marker is then within 10° of the image centre, otherwise the attempt fails (the controller never reverses). If odometry says the stop point is reached, it holds still until a frame confirms it.
  - *Marker loss.* After 2 s without a detection it rotates in place towards the last known bearing; docking gives up after 30 s.
  - *Tolerance.* Success is 0.25–0.35 m from the marker (base_link to marker centre) with the marker within ±15° of the image centre, judged against ground truth in `evaluation/metrics.py`. The 0.30 m target sits above the closest reliable detection range (0.25 m, Part B), and 15° is the bearing limit measured at 0.3 m.

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
| Tracking error (RMS) | _m (Part D)_ |
| Docking final range | 0.311 ± 0.001 m (target 0.30 m, allowed 0.25–0.35 m) |
| Docking final image bearing | 0.09 ± 0.07° (allowed ±15°) |
| Docking success rate | 5 / 5 |

Position and heading RMSE are for the route only, as in the Part C study (`ekf_study.py` runs with docking off). Docking figures are from `python scripts/dock_study.py` (5 seeds, P controller on the route):

| Route controller | Success | Final range [m] | Image bearing [°] | Lateral offset [cm] | Heading error [°] | Docking time [s] |
|---|---|---|---|---|---|---|
| P | 5/5 | 0.311 ± 0.001 | +0.09 ± 0.07 | 0.0 ± 0.1 | −0.01 ± 0.07 | 5.6 ± 0.1 |
| LQR | 5/5 | 0.310 ± 0.001 | −0.03 ± 0.05 | −0.1 ± 0.1 | −0.09 ± 0.12 | 5.7 ± 0.0 |

The robot stops about 1 cm beyond the target because it stops on the first frame inside the 1 cm tolerance; this is well inside the allowed band. Its own camera reading at the stop agrees with truth to 2–3 mm. During docking the EKF's χ² gate rejected 5 of the close-range (≈0.3 m) marker readings on seed 0 (NIS 15–58, none consecutive enough to trigger recovery) and mean NEES over the full run rose from 2.4 to 5.6: R appears too small at this range. Docking does not use the EKF, and the estimate stayed within 7 mm of truth.

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

**Gate lockout (estimator).** The first EKF was worse than dead reckoning (final error 18 cm vs 15 cm, 37 of 62 updates rejected). Logging every innovation showed that marker readings from 4.3–4.8 m, beyond the calibrated range, were biased ~15 cm in depth but accepted; they shifted the estimate ~10 cm and shrank P, after which correct close-range readings had NIS 30–45 and were all gated out. Fix: ignore readings beyond the calibrated range and add divergence recovery. Result: final error 0.5 cm, NEES 2.5.

**Unobservable direction (model).** Range+bearing cannot correct initial error along the rotation about the marker (Part A); the pose measurement model fixes this.

**Coupled sensor noise (simulation).** Turning on camera noise changed the odometry: heading drift went from −0.0286 to −0.0227 rad/m and turn scale error from 0.0031 to 0.0043, with nothing about the wheels changed. All sensors drew from one random generator, so every rendered frame (and later, every LiDAR beam that hit something) shifted the encoder noise; changing the controller would have changed the odometry too, confounding any comparison. Fix: one random stream per sensor. `tests/test_sim_noise.py` fails on the old code and passes now.

**Phantom LiDAR obstacles (simulation).** MuJoCo's rangefinder `cutoff` clamps long beams to exactly 3.5 m instead of reporting no return, creating a ring of fake obstacles (13 of 72 beams at the start pose). Found by pushing a real scan through the frame chain: one point landed 607 mm from any surface. After dropping max-range returns the worst point is 23 mm away. Caught by `tests/test_frames.py`.

**Marker dropouts near the image edge (sensing).** At close range the marker vanished for single degrees of rotation (e.g. 11° at 0.3 m) while fully visible mid-frame. OpenCV's default `minDistanceToBorder = 3 px` rejected candidates whose white border neared the image edge. Setting it to 0 removed every dropout: the usable bearing at 0.3 m rose from 10° to 15° and the closest reliable range fell from 0.30 to 0.25 m. Caught by `tests/test_aruco.py`.

**Half-pixel principal point (sensing).** Every bearing was offset by 0.069°, traced by the checkerboard calibration to the nominal image centre (320, 240) instead of (319.5, 239.5). See Part B.

**Marker range bias (sensing).** Range read up to 7 cm long beyond 3 m; traced to corner localisation rather than intrinsics, and corrected with a calibration validated on held-out poses. See Part B.

## 9. Hardware (optional)

_Setup, differences from simulation, quantitative sim-vs-hardware comparison._

## 10. Limitations and future work

_What the results do and do not support._

## 11. Individual contributions

| Member | Led | Contributed to |
|---|---|---|
| Muhammad Suhaib Khan (1351525) | Part A (modelling), Part C (EKF) | Project page, repository setup |
| Jaden Gambrill (26065197) | Part B (sensing): noise characterisation, camera calibration, range-bias correction, detection envelope, frames and frame diagram, sensing tests; simulation environment: independent sensor noise, LiDAR max-range fix, camera blur | Repository workflow (branch protection, reviews) |
| Siale Smythe (24831226) | Part D (control) | |

## 12. Generative AI declaration

| Tool | Purpose | Where used |
|---|---|---|
| _e.g. Claude_ | _e.g. README structure, debugging help_ | _e.g. README, src/estimation_ |

## 13. References

_Tutorials, papers, libraries used._
