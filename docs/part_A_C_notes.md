# Parts A and C: design notes and viva guide

These notes explain every design choice in the modelling (Part A) and estimation (Part C) code, with the numbers behind them. Read them alongside the code. The last section lists likely viva questions with model answers.

**File map**

| What | File |
|---|---|
| Model, Jacobians, controllability/observability functions | `src/model/diff_drive.py` |
| EKF | `src/estimation/ekf.py` |
| Marker measurement (position + yaw in base_link) | `src/sensing/aruco.py` |
| Rank tables and information plot (Part A) | `scripts/analyse_model.py` |
| R calibration | `scripts/calibrate_marker_noise.py` |
| Q calibration | `scripts/calibrate_odometry_noise.py` |
| Repeated trials and R trade-off | `scripts/ekf_study.py` |
| Wrong-initial-pose experiment | `scripts/observability_experiment.py` |
| Tests | `tests/test_model.py`, `tests/test_ekf.py` |
| Parameters | `configs/nominal.yaml` → `robot:` and `ekf:` |

---

## Part A: Modelling

### A1. State, input, continuous model

- State **x = [x, y, θ]ᵀ**: pose of `base_link` (midpoint of the wheel axle) in the world frame.
- Input **u = [v, ω]ᵀ**: forward speed and yaw rate.
- Kinematics of a differential drive (unicycle):

```
ẋ = v cos θ      ẏ = v sin θ      θ̇ = ω
```

- Wheel relation, with wheel radius **r = 0.033 m** and track width **b = 0.160 m** (TurtleBot3 Burger datasheet):

```
v = r(ω_R + ω_L)/2        ω = r(ω_R − ω_L)/b
```

Derivation in one line: each wheel's contact point moves at r·ω_wheel. The robot centre is the average of the two wheels; the difference divided by the distance between them is the turning rate.

**Assumptions**, which go on the project page: planar motion; no wheel slip; rigid wheels with known r and b; point contact; the caster carries no load sideways. **Where they break:** slip during fast turns and acceleration, and a wheel radius that differs from nominal. Both appear as odometry drift, which is why Q grows with distance travelled (see C2).

This is a **kinematic** model, not dynamic: inputs are velocities, not torques. That's justified because the wheel motors are velocity controlled and reach the commanded speed in about 0.3 s, much faster than the navigation dynamics.

### A2. Discrete odometry model (used by the EKF)

The encoders give, per step, the distance each wheel travelled: d_L, d_R. From these:

```
Δs = (d_R + d_L)/2        Δθ = (d_R − d_L)/b
```

Discrete motion model with the **midpoint heading** m = θ + Δθ/2:

```
x⁺ = x + Δs cos m
y⁺ = y + Δs sin m
θ⁺ = θ + Δθ
```

**Why the midpoint?** Over one step the robot follows an arc. Using the start heading (Euler) cuts the corner of every arc; using the heading halfway along is exact to second order. `test_quarter_circle_ends_on_arc` checks it: 1000 small steps around a quarter circle end within 0.1 mm of the exact point.

**Why drive the model with odometry rather than commanded v, ω?** The encoders measure what the wheels actually did, which is closer to the truth than what was commanded. This is the standard "odometry as control input" EKF.

### A3. Jacobians

Differentiate the discrete model:

```
F = ∂f/∂x = ⎡1  0  −Δs sin m⎤        G = ∂f/∂[Δs, Δθ] = ⎡cos m   −½Δs sin m⎤
            ⎢0  1   Δs cos m⎥                           ⎢sin m    ½Δs cos m⎥
            ⎣0  0   1       ⎦                           ⎣0        1        ⎦
```

How to explain them:
- **F**: a heading error δθ swings the predicted position sideways by Δs·δθ. That's the third column, and it's why heading errors are the main cause of position drift.
- **G**: noise in Δs moves the robot along its heading, the first column. Noise in Δθ rotates it, and because m uses Δθ/2 it also shifts position by ½Δs, the second column.

`test_motion_jacobians_match_finite_differences` checks F and G against numerical derivatives at 20 random points. Always verify hand-derived Jacobians like this: an EKF with a wrong Jacobian still runs, but it's quietly inconsistent.

### A4. Controllability (for the controller, Part D)

The pose model itself is driftless and nonholonomic, so its linearisation about a stationary point is **not** controllable. You can't push a differential drive sideways. The useful model is the **tracking error** in the robot frame about a reference (v_r, ω_r):

```
e = Rᵀ(θ)(p_ref − p),  e_θ = θ_ref − θ,  du = u_ref − u

ė = A e + B du,   A = ⎡ 0    ω_r   0 ⎤    B = ⎡1 0⎤
                      ⎢−ω_r  0    v_r⎥        ⎢0 0⎥
                      ⎣ 0    0     0 ⎦        ⎣0 1⎦
```

(From the Kanayama error dynamics ė₁ = ωe₂ − v + v_r cos e₃, ė₂ = −ωe₁ + v_r sin e₃, ė₃ = ω_r − ω, linearised at e = 0.)

Controllability matrix 𝒞 = [B, AB, A²B]. Results from `analyse_model.py`:

| Reference (v_r, ω_r) | rank 𝒞 | σ_min (discrete) |
|---|---|---|
| (0.20, 0.00) | 3 | 7.1e-4 |
| (0.20, 0.80) | 3 | 2.9e-3 |
| (0.05, 0.00) | 3 | 1.8e-4 |
| (0.00, 0.80) | 3 | 2.8e-3 |
| (0.00, 0.00) | **2** | 0 |

**What this means:** while the reference moves, the rotating error frame couples sideways error into the inputs, so all three errors are controllable. At rest the lateral error is uncontrollable. σ_min also shrinks as the reference slows, so control authority over the lateral error fades near the dock. That's why LQR needs a moving reference, and why the final docking approach is the hardest part for Part D.

The discretisation uses zero-order hold through the matrix exponential (`discretise`). The discrete ranks match the continuous ones, as they should for a non-pathological sample time.

### A5. Observability: the key result

Three measurement options:

**1. Odometry only:** there's no measurement of absolute pose (C = 0, rank 0). Any constant offset in the pose is invisible, so the error grows without bound. Result: dead reckoning ends **0.149 m** off (5 seeds).

**2. Range and bearing to one marker:** h(x) = [√(dx²+dy²), atan2(dy,dx) − θ], with H a 2×3 matrix.

```
H = ⎡−dx/ρ   −dy/ρ    0⎤          (dx = m_x − x, dy = m_y − y, ρ = range)
    ⎣ dy/ρ²  −dx/ρ²  −1⎦
```

One measurement gives rank 2. The surprising part is that **it stays rank 2 however much the robot moves.** The null vector is

```
n = [dy, −dx, 1]ᵀ        H n = 0
```

That's the direction of **rotating the robot about the marker**: move it around a circle centred on the marker and turn it by the same angle. Range doesn't change, and the bearing relative to the robot doesn't change. Odometry doesn't help either, because it measures motion in the robot's own frame, which is identical in the rotated world. So no motion and no number of measurements can recover this direction.

**3. Marker pose (position + orientation of the marker in base_link):**

```
h(x) = ⎡ cos θ·dx + sin θ·dy ⎤      H = ⎡−cos θ  −sin θ   h_y⎤
       ⎢−sin θ·dx + cos θ·dy ⎥          ⎢ sin θ  −cos θ  −h_x⎥
       ⎣ θ_m − θ             ⎦          ⎣ 0       0       −1 ⎦
```

The third row has the marker's orientation as seen by the camera, and that pins down the rotation about the marker. H is rank 3 from a single measurement. **This is why the EKF uses the pose model by default.** The ArUco detector gives orientation for free through `solvePnP`.

**Weak observability at range.** The third row is the only thing separating "sideways offset" from "rotated about the marker", and the yaw measurement gets noisy with distance: σ_yaw ≈ 0.002 + 0.028 r² rad. The smallest eigenvalue of HᵀR⁻¹H (information in the weakest direction) falls from 5310 at 0.6 m to 1.55 at 3 m. Put differently, one measurement constrains that direction to about 1.4 cm at 0.6 m but only 80 cm at 3 m (`part_A_observability.png`). So the filter only becomes really confident when it gets close to the dock, which suits a docking task well.

**Experimental check** (`observability_experiment.py`): both filters start 0.15 rad (~34 cm) wrong along the rotation direction, with the marker in view throughout.

| Model | Final error along rotation direction (3 seeds) |
|---|---|
| range + bearing | +0.168, +0.171, +0.162 rad: **never corrected** |
| pose | −0.003, −0.004, −0.003 rad: corrected within about 2 s |

**Why range+bearing still looked fine in the closed-loop trials:** the start pose is known to 2 cm, so the unobservable direction starts with almost no error and only drifts slowly. Unobservability matters when the prior is poor: an unknown start, a kidnapped robot, or a long time out of view.

---

## Part C: Estimation

### C1. Filter structure

EKF over x = [x, y, θ].

**Predict** every control step (20 Hz) from encoder increments:
```
x ← f(x, Δs, Δθ)
P ← F P Fᵀ + G Q Gᵀ
```

**Update** whenever the marker is detected (camera at 5 Hz):
```
y = z − h(x)              (wrap the angle component!)
S = H P Hᵀ + R
K = P Hᵀ S⁻¹
x ← x + K y
P ← (I − KH) P (I − KH)ᵀ + K R Kᵀ      (Joseph form)
```

**Why the Joseph form** instead of P = (I−KH)P? It's algebraically the same for the optimal K, but it stays symmetric and positive definite under rounding errors and slightly-off gains. It's cheap insurance.

**Why wrap the angle innovation?** If the measured yaw is 179° and the prediction is −179°, the real difference is 2°, not 358°. Without wrapping, one such update throws the heading off completely.

### C2. Process noise Q: from encoder physics, calibrated

Model: each wheel's measured travel per step has variance

```
var(d_wheel) = q0 + k·|d_wheel|
```

- q0 is a floor covering encoder quantisation and jitter, present even when barely moving.
- k·|d| is error that grows with distance, from slip and radius error. It's linear in distance because independent errors per unit distance add variances, like a random walk.

The two wheels are independent, and propagating through Δs = (d_R+d_L)/2, Δθ = (d_R−d_L)/b gives Q = J diag(var_L, var_R) Jᵀ with J = [[½, ½], [−1/b, 1/b]] (`odometry_covariance`).

**Calibration**, `calibrate_odometry_noise.py`, two stages:
1. **Per-step fit.** Compare odometry wheel travel with the true wheel travel rebuilt from consecutive true poses over 6 random runs, and fit q0 and k to the binned squared residuals. Result: q0 = 8.3e-9 m², k = 4.4e-8 m.
2. **Consistency scaling.** With these values, predict-only runs give **mean NEES = 22.5** (the ideal is 3): the filter is far too confident. Why? A wheel-radius error pushes the robot the same way *every* step. That error is correlated over time, which a per-step variance can't see, yet it dominates long-run drift. So I scale Q by a factor c, found by bisection, so that mean NEES = 3: **c = 10.3**.

Final values: **q0 = 8.6e-8 m², k = 4.5e-7 m.**

This is a standard and defensible tuning method: tune Q so the filter is *statistically consistent*, and report both the physical model and the scaling. **Limitation:** it uses the simulator's ground truth. On real hardware you'd use a motion-capture system, or drive a measured course (e.g. a 5 m straight and a 10-turn spin) and compare.

### C3. Measurement noise R: calibrated, range dependent

`calibrate_marker_noise.py` places the robot at 135 known poses (ranges 0.6–3.0 m, lateral offsets, ±0.25 rad headings) and takes 25 noisy frames at each, giving 3147 detections. Each measurement is compared with the value predicted from the known pose.

The RMS error grows with range, so I fit σ(r) = a + b·r² per component:

| Component | a | b | σ at 1 m | σ at 3 m |
|---|---|---|---|---|
| x (depth) | 0.00015 m | 0.0055 m/m² | 5.6 mm | 5.0 cm |
| y (lateral) | 0.00018 m | 0.0011 m/m² | 1.3 mm | 1.0 cm |
| yaw | 0.0017 rad | 0.028 rad/m² | 0.03 rad | 0.25 rad |
| range | 0.00016 m | 0.0056 m/m² | 5.8 mm | 5.1 cm |
| bearing | 0.0012 rad | 0 | 0.07° | 0.07° |

Why r²: a fixed pixel error δ at range r gives a depth error ∝ r²·δ/(f·s), because depth comes from the marker's apparent size. Lateral position and bearing come from where the marker sits in the image, which is much better conditioned. Depth is about 5× worse than lateral, and that's physically expected.

**Design choices worth defending:**
- **RMS, not standard deviation.** The errors include a pose-dependent bias (visible as vertical bands in `marker_noise_calibration.png`). The filter can't correct a bias, so R has to account for it.
- **Robust fit:** the worst 2 % per range bin (PnP yaw flips) are excluded. Those are outliers, handled by the gate, not noise.
- **Sub-pixel corner refinement** (`CORNER_REFINE_SUBPIX`). Without it, corners snap to whole pixels, which gave a 14 cm depth bias at 2.6 m and zero repeatability.
- **Range limit 3.0 m:** the R model is only valid where it was calibrated. Readings beyond that are ignored (see C5 for why this mattered).

### C4. Outlier gating

The normalised innovation squared NIS = yᵀS⁻¹y follows a χ² distribution with 3 degrees of freedom if the filter is right. An update is rejected when NIS > χ²₃(0.99) = 11.34, a measurement with less than 1 % probability of being genuine. This catches PnP yaw flips, a known ambiguity of planar targets seen nearly face-on.

### C5. Two failures found during development, and their fixes

This is good material for the viva question "what failed and how did you diagnose it?"

**Failure 1: gate lockout.** The first EKF version was *worse* than dead reckoning (final error 18 cm against 15 cm), with 37 of 62 updates rejected.

*Diagnosis:* I printed every innovation. At the start, the robot sees the marker from 4.3–4.8 m away, outside the calibrated range. Those readings were biased by ~15 cm in depth, and they were accepted because R was extrapolated. They dragged the estimate ~10 cm off and shrank P. Later, close-range readings were correct but inconsistent with the confident wrong estimate (NIS 30–45), so the gate rejected every one. The filter had locked itself out.

*Fixes:*
1. Ignore readings beyond the calibrated range (3 m).
2. Divergence recovery: after 5 consecutive rejections, the readings agree with each other but not with the estimate, so the estimate is assumed wrong. P is inflated ×100 and the update accepted.

*Result:* final error 0.5 cm, mean NEES 2.5, 1 rejection.

**Failure 2: unobservable direction.** See A5. Range+bearing can't fix an initial error along the rotation-about-marker direction, which motivated the pose model.

### C6. Results (5 seeds, closed loop, P controller)

| Condition | Pos. RMSE [m] | Final pos. error [m] | Heading RMSE [rad] | Mean NEES (ideal 3) | Rejected / run |
|---|---|---|---|---|---|
| Dead reckoning | 0.056 ± 0.004 | 0.149 ± 0.008 | 0.054 | – | – |
| EKF range+bearing | 0.023 ± 0.002 | 0.005 ± 0.000 | 0.021 | 2.25 ± 0.13 | 0.0 |
| EKF pose, R ×0.1 | 0.028 ± 0.004 | 0.010 ± 0.006 | 0.025 | 10.68 ± 2.73 | 13.0 |
| EKF pose, R ×0.3 | 0.023 ± 0.002 | 0.008 ± 0.003 | 0.021 | 4.49 ± 1.00 | 4.6 |
| **EKF pose, R ×1 (calibrated)** | **0.023 ± 0.002** | **0.007 ± 0.002** | **0.021** | **2.45 ± 0.13** | **1.4** |
| EKF pose, R ×3 | 0.023 ± 0.002 | 0.006 ± 0.000 | 0.021 | 1.63 ± 0.10 | 0.0 |
| EKF pose, R ×10 | 0.023 ± 0.002 | 0.006 ± 0.000 | 0.021 | 1.37 ± 0.09 | 0.0 |

**How to read it:**
- The EKF cuts the final error **about 20×** compared with dead reckoning (15 cm → 0.7 cm).
- The RMSE is similar across all EKF variants because for about 70 % of the run the marker is out of view or range, so the error there is pure odometry drift. The difference shows up in the final error and in consistency.
- **R trade-off** (`ekf_study_tradeoff.png`). Too small (×0.1): the filter is overconfident, NEES 10.7, and it rejects 13 genuine readings per run, so it ends *less* accurate. Too large (×10): NEES 1.4, underconfident, with ±3σ bounds wider than they need to be. Calibrated R gives NEES ≈ 2.5, closest to 3.
- NEES slightly below 3 means slightly conservative, which is the safe side to be on.

**Covariance when the marker is out of view** (`run_ekf_bounds.png`): from 0–18 s no usable marker reading arrives, and the ±3σ bounds grow steadily from odometry noise alone, to about ±0.2 m in x and y and ±0.16 rad in heading. At the first good reading, they collapse to centimetres. The error stays inside the bounds throughout.

### C7. Honest limitations

These belong in the project page's limitations section and score well at Distinction/HD:
- **Measurement errors are correlated, not white.** The bias depends on the pose, so consecutive readings from similar poses share it. The EKF assumes independence, so it slightly over-trusts repeated readings. Fixes could be inflating R, or decimating updates.
- **Q was tuned against simulator ground truth.** On hardware this needs an external reference.
- **Single landmark.** Accuracy depends on seeing one marker. More markers, or LiDAR scan matching, would bound the drift during the out-of-view middle section.
- **PnP yaw ambiguity** near face-on views is handled by gating, not solved.
- **Kinematic model:** wheel slip in aggressive manoeuvres isn't modelled, only absorbed into Q.

---

## How to reproduce everything

```bash
python -m pytest                              # 15 tests, including Jacobian checks
python scripts/analyse_model.py               # Part A ranks + information plot
python scripts/calibrate_marker_noise.py      # R (about 3 min)
python scripts/calibrate_odometry_noise.py    # Q (about 1 min)
python scripts/run.py                         # one EKF run + 3-sigma plot
python scripts/run.py --estimator dr          # dead-reckoning comparison
python scripts/ekf_study.py --seeds 5         # repeated trials + R trade-off (about 6 min)
python scripts/observability_experiment.py    # wrong initial pose experiment
```

---

## Viva practice: likely questions and answers

**Q: Which states are observable from which sensors?**
Odometry alone: none of the absolute pose. It only measures relative motion, so the error grows without bound. Range+bearing to one marker: rank 2. Rotation of the robot about the marker is unobservable, null vector [dy, −dx, 1], however much it moves. Marker pose: rank 3 from one measurement, because the marker's measured orientation fixes that rotation. It's weakly observable at long range because yaw noise grows as r².

**Q: What happens when the marker leaves the view?**
No updates, so the EKF runs predict-only. The estimate continues on odometry, and P grows each step through F P Fᵀ + G Q Gᵀ, faster with distance travelled. The heading uncertainty grows, and through F's third column it turns into growing sideways position uncertainty. When the marker comes back, the first update collapses P, since the gain is large because P is large. See the 0–18 s region of the bounds plot.

**Q: Where do Q and R come from?**
R comes from calibration: 3147 detections at known poses, RMS error per range bin, fitted as σ = a + b r². Depth noise grows as r² because depth comes from the marker's apparent size. Q comes from a per-wheel model, var = q0 + k|d|, fitted to step residuals, then scaled ×10.3 so predict-only NEES = 3, because the per-step fit misses correlated errors such as wheel-radius bias.

**Q: What would happen if R were ten times smaller?**
The filter would trust each marker reading 10× more. It becomes overconfident: P shrinks too much, NEES rises to ~10.7 instead of 3, and because S is too small, the gate rejects genuine readings (13 per run). The final error gets *worse* (1.0 cm vs 0.7 cm) and becomes more variable between runs. In general, over-trusting a sensor makes the estimate jumpy and lets bias straight through.

**Q: And ten times larger?**
The filter becomes underconfident (NEES 1.4). It corrects more slowly and its uncertainty bounds are wider than reality. Here the accuracy barely changed because there are many close-range readings at the end, but with fewer readings it would converge more slowly.

**Q: Why an EKF and not a KF?**
Both the motion model (cos θ, sin θ) and the measurement model (rotation into the robot frame) are nonlinear in θ. The EKF linearises them around the current estimate using the Jacobians F, G and H.

**Q: How did you check the Jacobians?**
Unit tests compare F, G and both H matrices with finite differences at 20 random points each.

**Q: Why the midpoint rule?**
The robot moves on arcs. Using the heading halfway along the step is exact to second order, while the start heading cuts every corner. A test checks that 1000 steps around a quarter circle end within 0.1 mm.

**Q: Why is the system not controllable at rest?**
It's nonholonomic: the wheels can't move it sideways. The linearised error model at v_r = ω_r = 0 has rank 2. While the reference moves, the rotating frame couples the lateral error to the inputs and the rank is 3. Control authority fades as v_r → 0, which makes the final docking approach hard.

**Q: What is NEES / NIS?**
NEES = eᵀP⁻¹e, using the true error (simulation only). If the filter's covariance is right, it averages to the state dimension, 3. NIS = yᵀS⁻¹y uses the innovation, so it works on real hardware. It's χ² with 3 degrees of freedom, and it's what the gate tests.

**Q: What failed and how did you diagnose it?**
Gate lockout (C5): the first EKF was worse than dead reckoning. Printing innovations showed far-range biased readings pulled the estimate off and made it overconfident, after which every correct reading was rejected. Fixed with a calibrated-range limit and divergence recovery.

**Q: What would you change?**
Add a second marker or LiDAR scan matching to bound the drift mid-run. Model the measurement bias, or inflate R for repeated similar views. Calibrate Q on hardware against an external reference. Try an iterated EKF or UKF for the large initial-error case.
