# Chapter briefs (condensed from the instructions given to the builders)

Common rules: see `agent_brief.md` and `docs/STYLE_GUIDE.md`. Each chapter: 4–7 notebooks, a README, tested modules in `rgkit/`, every number
verified in code, real-robot cross-checks against MuJoCo, only verified links (`docs/VIDEO_INDEX.md`, `docs/REFERENCES.md`).

## MFoR (follows Craig)
* **1 Introduction, DOF.** Tour of real robots (UR5e, Panda, iiwa, Sawyer, OP3, Baxter, Dynamixel 2R) with joint counts read from the MuJoCo models; Grübler–Kutzbach (four-bar, slider-crank, Stewart, failure cases); C-space topology (circle, torus, SO(3)); holonomic vs non-holonomic; workspace sampling vs analytic annulus (2R: both links 0.18 m); task space and redundancy.
* **2 Spatial descriptions.** 2.1–2.5 done. To do: 2.6 screw theory (twists, adjoint, exponential coordinates, wrenches), 2.7 ~15 original practice problems with verified solutions.
* **3 Forward kinematics.** Modified (Craig) vs standard D-H; frame assignment; worked planar/SCARA/articulated/spherical; PUMA 560 (PDF pp. 2–3, 38–40); **derive** UR5e/Panda D-H from the MuJoCo model and verify over ≥200 random configurations; product of exponentials; actuator/joint/Cartesian spaces.
* **4 Inverse kinematics.** Solvability, 2R/3R closed form, atan2, half-angle, Pieper/decoupling (count up to 8 branches), UR5e closed form with branches rendered side by side, numerical IK (Newton–Raphson, gradient/Jacobian-transpose, pseudoinverse, damped least squares), Panda redundancy.
* **5 Jacobians and statics.** Velocity propagation, column construction, frames, analytical vs geometric, singularities and manipulability ellipsoids, τ = JᵀF, force ellipsoids, null space; compare with MuJoCo `mj_jacSite` (world frame).
* **6 Dynamics.** Inertia tensor, Newton–Euler (recursive), Lagrangian (2R/3R), properties (M SPD, Ṁ−2C skew, linear in parameters), friction/gearing, forward dynamics and RK4; compare M(q), bias, `mj_inverse` with MuJoCo (note the `mj_fullM` signature change; see `docs/MUJOCO.md`).
* **7 Trajectories.** Cubic/quintic, via points, LSPB, Cartesian lines and SLERP, time scaling, run-time generation; execute on UR5e/Panda in MuJoCo.
* **8 Case studies.** **8.1 Baxter — complete study**: dual-arm, 7 joints per arm; D-H table from the course past paper (PDF p. 2 item 7: (0,0,0,θ1), (−90°,L1,0,θ2+90°), (90°,0,L2,θ3), (−90°,L3,0,θ4), (90°,0,L4,θ5), (−90°,L5,0,θ6), (90°,0,0,θ7)) and the offsets of arXiv 2409.00867 (0.27035, 0.069, 0.36435, 0.069, 0.37429, 0.01, 0.229525 m); real model `rgkit.robots.load_robot('baxter')` (robosuite, MIT); joint limits (verify before quoting), redundancy, workspace, Jacobian. **8.2 NAO — complete study**: 25-DOF humanoid; dimensions from the Aldebaran NAO docs (H25): NeckOffsetZ 126.5, ShoulderOffsetY 98, UpperArmLength 105, HipOffsetZ 85, HipOffsetY 50, ThighLength 100, TibiaLength 102.9, FootHeight 45.19 mm (v3.2 differs: state the version); no MuJoCo NAO model — draw a skeleton and/or capsule MJCF and verify D-H FK; PDF p. 41 gives the course approach to the past-paper NAO question. Later: industrial six-axis pipeline, OP3 humanoid (COM, support polygon), safety/HRI (qualitative).
* **9 (planned) Mechanism design and closed chains** — Craig Ch. 8; Lynch & Park closed chains: workspace, actuation, stiffness, sensing; four-bar, Stewart platform, delta.
* **10 (planned) Programming languages and off-line programming** — Craig Ch. 12–13: levels of robot programming, task-level, simulation-based programming, Python/MuJoCo as an off-line programming environment, calibration.
* **11 (planned) Motion planning** — configuration-space obstacles, potential fields, PRM/RRT, collision checking with MuJoCo on a real arm.

## RMC
* **1 Foundations.** Transfer functions, DC motor, second-order response (ωₙ, ζ), Routh–Hurwitz, root locus, Bode/Nyquist, PID tuning/anti-windup, state-space, pole placement, observers, LQR intro; identify a joint model from MuJoCo. (Only `scipy` — `control` not installed.)
* **2 Independent-joint control.** J_eff = J_m + J_l/r²; P/PD/PID; gravity error; feedforward; control-law partitioning (α/β); saturation, flexibility, sampling; PD/PID on UR5e/Panda/2R in MuJoCo.
* **3 Lyapunov, model-based.** Lyapunov and LaSalle; PD + gravity; computed torque; robustness; Ṁ−2C skew; computed torque on a real arm using MuJoCo dynamics with an unknown payload.
* **4 Robust, adaptive, operational-space.** Sliding mode/boundary layer, Lyapunov redesign, Slotine–Li adaptive (regressor), operational-space control (Khatib), null-space tasks; Panda posture task.
* **5 Force and interaction.** Constraints, impedance/admittance, hybrid position/force (Raibert–Craig), contact in MuJoCo.
* **6 Teleoperation and master–slave.** P–P, P–F, four-channel; transparency vs stability; delay, passivity, wave variables; teach pendant / teach-and-playback; master–slave pair in MuJoCo. (IIIT-A syllabus lists master–slave explicitly.)
* **7 Optimal, predictive, learning.** LQR, iLQR, MPC, RL basics, policy gradient, residual RL, imitation (numpy only; short runs). (RL appears in the IIIT-A past paper.)
* **8 (stopped) Sensing and visual servoing**, **9 (stopped) Underactuated and mobile** — see `SYLLABUS.md` for notebook plans.
