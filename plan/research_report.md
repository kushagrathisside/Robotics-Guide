# Research report for the main thread: MFoR and RMC syllabus and resource research

**Research environment limits (read first).** Network access was restricted for this run.
- WebFetch could not reach youtube.com or sites.google.com (DNS failure, "outside allowed domains").
- curl got a proxy 403 for youtube.com, aaa.iiita.ac.in and hades.mech.northwestern.edu.
- Only WebSearch (which returns snippets and summaries), pypi.org and the instructor PDF were usable. The GitHub API refused unattached repositories.

As a result:
- No IIIT-A PDF could be opened. All IIIT-A syllabus text below comes from search-engine snippets, which are close to the original but **not guaranteed verbatim**.
- No YouTube page could be opened. Playlist IDs come from the instructor PDF or from search-result URLs.
- Book chapter titles were checked against publisher, library or search listings where possible. Section-level numbering is often from my own knowledge, cross-checked against the instructor's notes (which cite Craig and L&P section numbers). Each such item is flagged **[mem]**: verify against the book before publishing.

Abbreviations:
- **C** = Craig, 3rd ed.
- **LP** = Lynch & Park, *Modern Robotics* (2017)
- **SHV** = Spong, Hutchinson & Vidyasagar, 2nd ed. (2020)
- **SSVO** = Siciliano, Sciavicco, Villani & Oriolo (2009)
- **RVC** = Corke, *Robotics, Vision and Control*, 3rd ed. (Python, 2023)
- **MLS** = Murray, Li & Sastry (1994)
- **SL** = Slotine & Li
- **K** = Khalil, 3rd ed.
- **O** = Ogata, 5th ed.
- **N** = Nise
- **MIT-M** = Tedrake, *Robotic Manipulation*
- **MIT-U** = Tedrake, *Underactuated*
- **GCN** = Prof. Nandi's lecture number (MFoR playlist)
- **TE** = ThatsEngineering

---

## (a) IIIT Allahabad syllabus text found

### Documents identified (URLs seen in search results; none could be opened)

1. **IIITA/Course Syllabus/2023/v.2**, "applicable with effect from 2023 admitted batch and onwards":
   - https://aaa.iiita.ac.in/PDF1/course%20syllabus%20PG.pdf
   - https://aaa.iiita.ac.in/PDF1/M.Tech%20Curriculum%202025.pdf (mirror: https://prc.iiita.ac.in/PDF1/M.Tech%20Curriculum%202025.pdf)
   - UG: https://aaa.iiita.ac.in/PDF1/UG%20B,Tech%20%20Curriculum%202025.pdf
2. **B.Tech 2022 & 2023 Batch Course Curriculum**: https://aaa.iiita.ac.in/PDF1/B.Tch%202022%20&%202023%20Batch%20Curriculum.pdf
3. **Guidelines for Choosing M.Tech(IT) Specialization 2021-22**: https://it.iiita.ac.in/ITSpecializationguidelines.php (and the PDF https://aaa.iiita.ac.in/PDF1/M.Tech%20IT%20for%20uploading.pdf)
4. M.Tech–PhD curriculum 2022: https://it.iiita.ac.in/MTechPhD_Curriculum_2022.html ; Robotics lab M.Tech page: https://robita.iiita.ac.in/mtech.php
5. Prof. Nandi's study-materials page (https://sites.google.com/iiita.ac.in/gcnandi/study-materials) lists "Mathematical Foundation of Robotics (July 2024–December 2024)" and says students should follow materials as instructed in class.
6. No page for Dr. K. R. Kaza was found. The search returned nothing for "Kaza" at IIIT-A. His slides are known only through the instructor notes ("K. R. Kaza, *Mathematical Foundations of Robotics: Forward and Inverse Kinematics*, lecture slides, IIIT Allahabad (MFoR.pdf, 65 slides)").

### Robot Motion Control (RMC)
Extracted from snippets of the 2023 v.2 syllabus. Wording is near-verbatim but unconfirmed.

- **Course:** Robot Motion Control, **IT 4020**, type **PCC**, **4 credits**, L-T-P-? **3-0-2-0**. Listed under **RMI (Robotics and Machine Intelligence) Semester 1** in the PG syllabus. The same code and credits appear in the B.Tech 2022/2023 curriculum.
- **Objective (truncated):** "To be able to obtain a working mathematical model of a system and be able to do a time-domain and [frequency-domain analysis] ..."
- **Topic groups seen:**
  - "Robot Modeling Using Dynamics, Trajectory Planning"
  - "Control Systems for Robotics: Robot Control Principles, Master-Slave Control Architecture, PD-PID Control, Computed Torque/Model-Based Methodology, Nonlinear Control, Sensory Devices for Robots"
- **Books in the same document** (assignment to IT 4020 is uncertain because the extracted text was garbled):
  - Ogata, *Modern Control Engineering* (PHI)
  - Craig, *Introduction to Robotics: Mechanics and Control*
  - Siciliano et al., *Robotics: Modelling, Planning and Control*
  - Lynch & Park, *Modern Robotics*
  - Corke, *Robotics, Vision and Control: Fundamental Algorithms in MATLAB*
  - Franklin, Powell & Emami-Naeini, *Feedback Control of Dynamic Systems*
- **Gaps:** Reinforcement learning, impedance/force control and adaptive/robust control were **not** visible in the snippets. The user's description of RMC (which includes RL and master–slave) is consistent with "Nonlinear Control" plus "Master-Slave Control Architecture", but RL could not be confirmed in the official text.

### Mathematical Foundations of Robotics (MFoR)
- **2021-22 guidelines (RMI specialisation; contact listed as Dr. Rahul Kala):**
  - "Mathematical Foundations of Robotics -1": 4 credits, 2-0-2, HC (core).
  - "Mathematical Foundations of Robotics -2": listed in the Third Semester table, 4 credits, 2-1-1.
- **2023 v.2 syllabus:** the course appears as **"Foundation of Robotics" (IT 4030)**. Objective: "to build a strong mathematical foundation for modeling, analyzing, and designing robotic systems". One topic heading seen: "Mathematical Foundation for Spatial Rigid Body". The full unit list was not retrievable.
- **The 2022 M.Tech–PhD curriculum** lists "Foundations of Robotics", 4 credits, 2-1-1.
- **Actual scope as taught (from the instructor's mid-sem notes, p. 8):**
  - Mid-sem covers Craig Ch. 1–4 (excluding §4.6 Pieper and §4.7 PUMA/Motoman worked IK), plus L&P Ch. 6 numerical IK, plus quaternions (Corke Ch. 2 / GCN L13–14).
  - Lectures 26–29 of the Nandi playlist cover dynamics, so post-mid-sem likely covers Craig Ch. 5–7 (Jacobians/statics, dynamics, trajectories). This inference about end-sem scope is **[inference]**.

**Recommendation:** ask the user to download and share the two IIIT-A PDFs (course syllabus PG; UG B.Tech curriculum 2025), or allow aaa.iiita.ac.in / sites.google.com in network settings, to get verbatim text.

---

## (b) Consolidated topic lists (chapter, then section, then subsection) with sources

Each line gives the sources that cover the topic and the course(s) that teach it. Chapter and section references for each textbook are in (d).

### MFoR: Mathematical Foundations of Robotics

**1. Robots, links, joints and degrees of freedom**
- 1.1 Robot anatomy
  - 1.1.1 Manipulator, links, joints, base, end-effector, tool; serial vs closed chains (C 1; LP 2; notes U1)
  - 1.1.2 Robot classification: articulated, SCARA, spherical, cylindrical, Cartesian, parallel (C 1, 3.7; SSVO 1; NPTEL Pratihar)
  - 1.1.3 Actuators, transmissions and sensors at overview level (SSVO 5; C 8)
- 1.2 Degrees of freedom
  - 1.2.1 DOF of a rigid body in the plane and in space (LP 2.1)
  - 1.2.2 Lower pairs and joint types: R, P, H, C, U, S (LP 2.2.1; notes 1.4)
  - 1.2.3 Grübler's formula; closed chains, four-bar, Stewart platform (LP 2.2.2; notes 1.6)
- 1.3 Configuration space
  - 1.3.1 C-space topology and representation (explicit vs implicit) (LP 2.3)
  - 1.3.2 Holonomic and nonholonomic (Pfaffian) constraints (LP 2.4)
  - 1.3.3 Joint space, Cartesian/task space, workspace, dexterous workspace (LP 2.5; C 3.6, 4.2)

**2. Spatial descriptions and rotations**
- 2.1 Positions, orientations and frames
  - 2.1.1 Craig's leading super/subscript notation (C 2.2)
  - 2.1.2 Rotation matrix as the axes of {B} written in {A}; orthonormality; SO(3) (C 2.2; LP 3.2.1; SHV 2)
  - 2.1.3 Elementary rotations about x, y, z (C 2.2; SSVO 2.2)
- 2.2 Mappings and operators
  - 2.2.1 Changing the frame of description (C 2.3)
  - 2.2.2 Composition: the chain rule; fixed vs current axes (pre- vs post-multiplication) (C 2.3, 2.8; SHV 2; SSVO 2.3)
  - 2.2.3 Rotations as operators; three interpretations of a rotation matrix (C 2.4–2.5)
- 2.3 Angular velocity and skew-symmetric matrices
  - 2.3.1 so(3), the hat/vee map, cross products (LP 3.2.2; MLS 2.2)
  - 2.3.2 $\dot R = [\omega] R$; space vs body angular velocity (LP 3.2.2; C 5.3–5.4)

**3. Homogeneous transformations and rigid-body motion**
- 3.1 Homogeneous transformation matrices (HTM)
  - 3.1.1 The 4×4 HTM, SE(3), homogeneous coordinates (C 2.3; LP 3.3.1; SHV 2)
  - 3.1.2 Inverse using structure: $R^\top$, $-R^\top p$ (C 2.6; LP 3.3.1)
  - 3.1.3 Compound transforms, transform equations, transform graphs (C 2.6–2.7)
- 3.2 Standard frames
  - 3.2.1 Base, station, wrist, tool and goal frames; "where is the tool?" (C 3.8–3.9)
  - 3.2.2 Free vectors and their transformation (C 2.9)
  - 3.2.3 Computational considerations (C 2.10)
- 3.3 Screw theory view *(not in Craig)*
  - 3.3.1 Twists, se(3), the adjoint map (LP 3.3.2; MLS 2.3–2.4)
  - 3.3.2 Exponential coordinates of rigid motion; screw axis; Chasles–Mozzi (LP 3.3.3; MLS 2.3)
  - 3.3.3 Wrenches and their frame changes (LP 3.4; MLS 2.5)

**4. Orientation representations**
- 4.1 Three-angle sets
  - 4.1.1 X-Y-Z fixed angles (roll–pitch–yaw) and Z-Y-X Euler angles (C 2.8; SSVO 2.4)
  - 4.1.2 Z-Y-Z Euler angles; the 24 conventions (C 2.8, App. B)
  - 4.1.3 Inverse problem with atan2; gimbal lock / representation singularities (C 2.8; notes 4.3–4.6)
- 4.2 Angle–axis and exponential coordinates
  - 4.2.1 Rodrigues' formula (C 2.8; LP 3.2.3; SSVO 2.5)
  - 4.2.2 Matrix logarithm of a rotation (LP 3.2.3.3 **[mem]**)
- 4.3 Unit quaternions (Euler parameters)
  - 4.3.1 Algebra: product, conjugate, norm, inverse (RVC 2; SSVO 2.6)
  - 4.3.2 Conversions to and from rotation matrices and angle–axis (C 2.8 "Euler parameters"; RVC 2)
  - 4.3.3 SLERP interpolation; double cover (RVC 2–3; notes 5.6)
- 4.4 Comparison and choice of representation (notes 4.9, 5.7; RVC 2)

**5. Forward kinematics**
- 5.1 Denavit–Hartenberg convention
  - 5.1.1 Link length, twist, offset, joint angle (C 3.2–3.3)
  - 5.1.2 Frame-assignment rules; modified (Craig/proximal) vs standard (distal) D-H (C 3.4; SHV 3; SSVO 2.8)
  - 5.1.3 Link transform ${}^{i-1}_{i}T$ and chaining (C 3.5)
- 5.2 Worked manipulators
  - 5.2.1 Planar 2R/3R (C Ex. 3.3); RPR (C Ex. 3.4) (notes 7.1–7.3)
  - 5.2.2 Articulated (elbow), SCARA, spherical (Stanford), cylindrical arms (SHV 3; SSVO 2.9)
  - 5.2.3 PUMA 560 and Yasukawa Motoman (C 3.7); spherical wrist (SSVO 2.9)
- 5.3 Product of exponentials *(not in Craig)*
  - 5.3.1 Space-frame PoE (LP 4.1.1; MLS 3.2)
  - 5.3.2 Body-frame PoE (LP 4.1.3)
  - 5.3.3 URDF robot descriptions (LP 4.2)
- 5.4 Actuator, joint and Cartesian space; kinematic calibration (C 3.6; SSVO 2.10–2.11)

**6. Inverse kinematics**
- 6.1 Solvability
  - 6.1.1 Existence (workspace) and multiplicity of solutions (C 4.2)
  - 6.1.2 Manipulator subspace when n < 6 (C 4.3)
  - 6.1.3 Redundancy vs dexterity (SSVO 3.4; notes 8.15)
- 6.2 Closed-form methods
  - 6.2.1 Algebraic and geometric solutions; atan2 (C 4.4; SHV 5 **[mem]**)
  - 6.2.2 Reduction to polynomial (tangent half-angle); standard trig equations (C 4.5, App. C)
  - 6.2.3 Pieper's solution and kinematic decoupling with a spherical wrist (C 4.6; SHV 5; LP 6.1)
  - 6.2.4 Paden–Kahan subproblems *(not in Craig)* (MLS 3.3; CMU 16-711)
- 6.3 Numerical methods
  - 6.3.1 Newton–Raphson, full-pose IK with twists (LP 6.2.1–6.2.2)
  - 6.3.2 Jacobian transpose / gradient descent; pseudoinverse; damped least squares (SSVO 3.7; notes 9.6–9.10)
  - 6.3.3 Inverse velocity kinematics; null-space motion (LP 6.3; SSVO 3.5)
- 6.4 Practical matters: repeatability, accuracy, SOLVE-ing (C 4.8–4.11)

**7. Velocity kinematics, Jacobians and statics**
- 7.1 Velocity propagation
  - 7.1.1 Linear and angular velocity of rigid bodies (C 5.2–5.4)
  - 7.1.2 Link-to-link propagation (C 5.5–5.6)
- 7.2 The manipulator Jacobian
  - 7.2.1 Geometric Jacobian by column construction (SSVO 3.1–3.2; SHV 4)
  - 7.2.2 Space and body Jacobians *(PoE)* (LP 5.1)
  - 7.2.3 Analytical Jacobian and representation singularities (SSVO 3.6; SHV 4)
  - 7.2.4 Frame changes of the Jacobian (C 5.7)
- 7.3 Singularities
  - 7.3.1 Boundary and interior singularities; wrist/arm decoupling (C 5.8; SSVO 3.3; LP 5.3)
  - 7.3.2 Manipulability ellipsoids and measures (LP 5.4; SSVO 3.9)
- 7.4 Statics
  - 7.4.1 Static force propagation (C 5.9)
  - 7.4.2 $\tau = J^\top F$; force ellipsoids; kineto-static duality (C 5.10; LP 5.2; SSVO 3.8)
  - 7.4.3 Cartesian transformation of velocities and forces (C 5.11)

**8. Manipulator dynamics**
- 8.1 Rigid-body mass properties
  - 8.1.1 Inertia tensor, parallel-axis theorem (C 6.3; LP 8.2)
  - 8.1.2 Newton–Euler equations of a single body (C 6.4; LP 8.2)
- 8.2 Recursive Newton–Euler
  - 8.2.1 Outward velocity/acceleration and inward force/torque recursions (C 6.5)
  - 8.2.2 Iterative vs closed form (C 6.6–6.7)
  - 8.2.3 Twist/wrench (spatial) form (LP 8.3)
- 8.3 Lagrangian formulation
  - 8.3.1 Kinetic and potential energy; Euler–Lagrange equations (C 6.9; SHV 6; SSVO 7.1; LP 8.1)
  - 8.3.2 Structure $M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau$; Christoffel symbols (C 6.8; SHV 6)
  - 8.3.3 Properties: $M$ symmetric positive definite, $\dot M - 2C$ skew-symmetric, linearity in parameters (SSVO 7.2; SHV 6)
- 8.4 Extensions
  - 8.4.1 Cartesian / operational-space dynamics (C 6.10; LP 8.6; SSVO 7.8)
  - 8.4.2 Friction, gearing, rotor inertia (C 6.11; LP 8.9)
  - 8.4.3 Forward dynamics and simulation (C 6.12; LP 8.5)
  - 8.4.4 Dynamic parameter identification (SSVO 7.4)

**9. Trajectory generation**
- 9.1 Path vs trajectory; time scaling (C 7.2; LP 9.1; SSVO 4.1)
- 9.2 Joint-space schemes
  - 9.2.1 Cubic and quintic polynomials; via points (C 7.3; LP 9.2–9.3; SHV 7)
  - 9.2.2 Linear segments with parabolic blends (LSPB) / trapezoidal velocity (C 7.3; SHV 7)
- 9.3 Cartesian-space schemes
  - 9.3.1 Straight lines; orientation interpolation (angle–axis, SLERP) (C 7.4; SSVO 4.3)
  - 9.3.2 Geometric problems with Cartesian paths (C 7.5)
- 9.4 Time-optimal time scaling (LP 9.4); planning with the dynamic model (C 7.8)

### RMC: Robot Motion Control

**1. Control-systems foundations**
- 1.1 Modelling
  - 1.1.1 Transfer functions and block diagrams; DC-motor model (O 2–3; N 2; SHV 8)
  - 1.1.2 State-space models; linearisation (O 9; N 3)
- 1.2 Time response
  - 1.2.1 First- and second-order systems; $\omega_n$, $\zeta$, overshoot, settling time (C 9.3; O 5; N 4)
  - 1.2.2 Stability, Routh–Hurwitz, steady-state error (O 5; N 6–7)
- 1.3 Classical design
  - 1.3.1 Root locus (O 6; N 8–9)
  - 1.3.2 Frequency response, Bode/Nyquist, margins (O 7; N 10–11)
  - 1.3.3 PID: structure, tuning (Ziegler–Nichols), anti-windup (O 8; MathWorks PID talks)
- 1.4 State-space design: pole placement, observers, LQR (O 10; N 12; Brunton *Control Bootcamp*)
- 1.5 Digital implementation: sampling, servo rates (C 9.8; N 13)

**2. Linear (independent-joint) control**
- 2.1 Single-joint model
  - 2.1.1 Actuator + gear + link; effective inertia and damping; gear ratio (C 9.9; SHV 8; SSVO 8.3)
  - 2.1.2 Disturbance from coupling and gravity (SHV 8; SSVO 8.3.1)
- 2.2 Set-point control
  - 2.2.1 P, PD, PID; critical damping; integral action for disturbance rejection (C 9.4, 9.7; SHV 8; LP 11.3–11.4 **[mem]**)
  - 2.2.2 Feedforward and trajectory-following (C 9.6; SSVO 8.3.2)
  - 2.2.3 Saturation, flexibility and structural resonance limits (C 9.9; SHV 8)
- 2.3 Control-law partitioning: model-based plus servo portion (C 9.5)
- 2.4 Industrial controller architecture (C 9.10, 10.6)

**3. Nonlinear and model-based control of manipulators**
- 3.1 Lyapunov stability theory
  - 3.1.1 Definitions, direct method, LaSalle invariance (SL 3; K 4; SHV App. C; C 10.7)
  - 3.1.2 Passivity and energy shaping (K 6; SL 4 **[mem]**; SHV 9)
- 3.2 PD with gravity compensation (SSVO 8.5.1; SHV 9; Kelly–Santibáñez–Loría)
- 3.3 Computed torque / inverse dynamics control
  - 3.3.1 Feedback linearisation in joint space (C 10.4; SHV 9; SSVO 8.5.2; LP 11.4)
  - 3.3.2 Computed-torque feedforward vs feedback (SSVO 8.4; Lewis et al. 4)
  - 3.3.3 Robustness to model error (C 10.5; SSVO 8.5.3)
- 3.4 Robust control
  - 3.4.1 Sliding-mode control and chattering (SL 7; K 14.1)
  - 3.4.2 Lyapunov redesign / robust inverse dynamics (SHV 9; SSVO 8.5.3; Lewis et al. 5)
- 3.5 Adaptive control
  - 3.5.1 Linear-in-parameters regressor (SSVO 7.2; SHV 6)
  - 3.5.2 Slotine–Li adaptive controller; MRAC basics (SL 8–9; C 10.9; SSVO 8.5.4)
- 3.6 Operational-space (task-space) control
  - 3.6.1 Jacobian-transpose PD + gravity (SSVO 8.6.2)
  - 3.6.2 Operational-space inverse dynamics (Khatib) (SSVO 8.6.3; CS223A/CS327A; C 10.8)
  - 3.6.3 Redundancy resolution and null-space control (CS327A; SSVO 3.4)
- 3.7 Feedback linearisation in general (SL 6; K 13; SHV 12 **[mem]**)

**4. Force and interaction control**
- 4.1 Interaction modelling
  - 4.1.1 Natural and artificial constraints (C 11.3; SSVO 9.5–9.6)
  - 4.1.2 Environment stiffness; mass–spring force control (C 11.5)
- 4.2 Indirect force control
  - 4.2.1 Stiffness and compliance control (SSVO 9.2)
  - 4.2.2 Impedance control (Hogan) (SSVO 9.3; LP 11.7; SHV 10)
  - 4.2.3 Admittance control (De Luca Robotics 2; Polimi CIR)
- 4.3 Direct force control
  - 4.3.1 Force control with inner position or velocity loop (SSVO 9.4; LP 11.5)
  - 4.3.2 Hybrid position/force control (Raibert–Craig) (C 11.4–11.6; SSVO 9.7; LP 11.6)
- 4.4 Low-level joint torque control; series-elastic actuation (LP 11.8; MIT-M)

**5. Teleoperation and master–slave control**
- 5.1 Architectures
  - 5.1.1 Unilateral vs bilateral; position–position, position–force, four-channel (Hokayem & Spong 2006)
  - 5.1.2 Transparency and stability trade-off (Hokayem & Spong 2006)
- 5.2 Time delay
  - 5.2.1 Passivity, scattering/wave variables (Hokayem & Spong 2006; K 6 for passivity)
- 5.3 Haptics and shared autonomy (Siciliano & Khatib *Springer Handbook*, teleoperation chapter **[unverified chapter number]**)

**6. Optimal, predictive and learning-based control**
- 6.1 LQR, iLQR/DDP, trajectory optimisation (CMU 16-745; MIT-U)
- 6.2 Model predictive control (CMU 16-745 "Convex MPC"; MIT-U; MathWorks MPC talks; CasADi/do-mpc)
- 6.3 Reinforcement learning for control
  - 6.3.1 MDPs, value and policy methods (Sutton & Barto)
  - 6.3.2 Policy gradients, actor–critic, model-based RL (Sutton & Barto; MIT-U RL chapters; Kober et al. 2013)
  - 6.3.3 Sim-to-real, residual RL on top of computed torque, imitation learning (Kober et al.; MuJoCo/Gymnasium/SB3)
- 6.4 Neural-network control of manipulators (Lewis et al. 8)

**7. Sensing and implementation**
- 7.1 Encoders, tachometers, force/torque sensors (SSVO 5; C 8; IIT-A "Sensory Devices for Robots")
- 7.2 Control architecture and real-time software (SSVO 6; C 9.10; Lewis et al. 10; ros2_control)
- 7.3 Visual servoing (overview) (SSVO 10; SHV 11; RVC 15)

**8. Underactuated and mobile systems (extensions)**
- 8.1 Pendulum, acrobot, cart-pole; energy shaping, partial feedback linearisation (MIT-U; SHV 13)
- 8.2 Wheeled mobile robot control; driftless systems (SHV 14; LP 13; SSVO 11)

---

## (c) Topics taught elsewhere that a Craig-only syllabus would miss

| Topic | Where it is taught | Book coverage |
|---|---|---|
| Screw theory, twists, wrenches, adjoint, exponential coordinates | Northwestern ME 449; CMU 16-711; Berkeley C106A | LP 3; MLS 2 |
| Product-of-exponentials FK; space/body Jacobians | ME 449; C106A | LP 4–5; MLS 3 |
| Paden–Kahan subproblems for IK | CMU 16-711; C106A | MLS 3 |
| Numerical IK: Newton–Raphson, DLS, null-space projection | ME 449; MIT 6.4210 | LP 6; SSVO 3.5–3.7 |
| Unit-quaternion algebra and SLERP (Craig has Euler parameters only) | Most courses | SSVO 2.6; RVC 2 |
| Manipulability ellipsoids | ME 449; Sapienza | LP 5.4; SSVO 3.9 |
| Linearity-in-parameters, skew-symmetry of $\dot M-2C$, parameter identification | Illinois ME 446; Sapienza Robotics 2 | SHV 6; SSVO 7.2, 7.4 |
| PD + gravity compensation with a Lyapunov proof | Polimi; Sapienza; Illinois | SSVO 8.5.1; SHV 9 |
| Robust (sliding-mode, Lyapunov-redesign) control | Polimi; Sapienza | SL 7; SSVO 8.5.3; SHV 9 |
| Full adaptive control (Slotine–Li) (Craig 10.9 is brief) | Sapienza Robotics 2; Polimi | SL 8–9; SSVO 8.5.4 |
| Operational-space control and redundancy | Stanford CS223A/CS327A; CMU 16-711; TUM | SSVO 8.6; Khatib papers |
| Impedance and admittance control | CMU 16-711; Sapienza; Polimi; Illinois | SSVO 9.3; LP 11.7; SHV 10 |
| Passivity-based control, bilateral teleoperation | Sapienza; specialised | K 6; Hokayem & Spong |
| LQR, MPC, trajectory optimisation | CMU 16-711 and 16-745; MIT 6.8210 | MIT-U; 16-745 notes |
| Learning-based control and RL | CMU 16-745; MIT 6.8210; NPTEL Asokan | Sutton & Barto; Kober et al. |
| Visual servoing | Most second courses | SSVO 10; SHV 11; RVC 15–16 |
| Underactuated systems | MIT 6.8210 | MIT-U; SHV 13 |
| Floating-base / legged dynamics | ETH Robot Dynamics | ETH notes |
| Simulation-first practice (Pinocchio, Drake, MuJoCo) | MIT 6.4210/6.8210; ETH | — |

---

## (d) Chapter and section numbers per textbook, by topic

Confidence key: **V** = verified from a publisher, library or search listing, or from the instructor's notes; **M** = from memory, consistent with V items, verify before publishing; **?** = uncertain.

### Craig, 3rd ed. (13 chapters plus Appendices A–C)

**Chapter titles (V, LOC/Pearson):**
1 Introduction; 2 Spatial descriptions and transformations; 3 Manipulator kinematics; 4 Inverse manipulator kinematics; 5 Jacobians: velocities and static forces (p. 135); 6 Manipulator dynamics (p. 165); 7 Trajectory generation; 8 Manipulator-mechanism design; 9 Linear control of manipulators; 10 Nonlinear control of manipulators; 11 Force control of manipulators; 12 Robot programming languages and systems; 13 Off-line programming systems.

The 4th ed. (Pearson eLibrary) keeps Ch. 7 Trajectory Generation, Ch. 8 Manipulator-Mechanism Design, Ch. 12 Robot Programming Languages and Systems, and Ch. 13 Off-Line Programming Systems (also listed as "Simulation and Off-Line Programming"). One listing names Ch. 9–11 simply "Linear Control", "Non-Linear Control" and "Force Control" (?).

**Sections:**
- Ch. 2: 2.2 Descriptions: positions, orientations and frames (M); 2.3 Mappings (V notes); 2.4 Operators (M); 2.5 Summary of interpretations (V notes); 2.6 Transformation arithmetic (M); 2.7 Transform equations (M); 2.8 More on representation of orientation (V notes); 2.9 Transformation of free vectors (M); 2.10 Computational considerations (V notes). Appendix B: the 24 angle-set conventions (V notes). Appendix C: inverse-kinematic formulas (V notes).
- Ch. 3: 3.2 Link description (V); 3.3 Link-connection description (V); 3.4 Convention for affixing frames to links (V); 3.5 Manipulator kinematics (V); 3.6 Actuator space, joint space and Cartesian space (V); 3.7 Examples: two industrial robots (PUMA 560, Yasukawa Motoman) (V); 3.8 Frames with standard names (M); 3.9 Where is the tool? (M); 3.10 Computational considerations (V notes).
- Ch. 4: 4.2 Solvability (V); 4.3 Manipulator subspace when n < 6 (V); 4.4 Algebraic vs geometric (V); 4.5 Reduction to polynomial (V); 4.6 Pieper's solution (V); 4.7 Examples (V); 4.8–4.11 Standard frames, SOLVE-ing, repeatability and accuracy, computational considerations (V notes).
- Ch. 5 (M): 5.2 Notation for time-varying position/orientation; 5.3 Linear and rotational velocity of rigid bodies; 5.4 More on angular velocity; 5.5 Motion of the links of a robot; 5.6 Velocity propagation from link to link (one lecture outline numbers this 5.5 (?)); 5.7 Jacobians; 5.8 Singularities; 5.9 Static forces; 5.10 Jacobians in the force domain; 5.11 Cartesian transformation of velocities and static forces.
- Ch. 6 (M): 6.2 Acceleration of a rigid body; 6.3 Mass distribution; 6.4 Newton's and Euler's equations; 6.5 Iterative Newton–Euler; 6.6 Iterative vs closed form; 6.7 Closed-form example; 6.8 Structure of the dynamic equations; 6.9 Lagrangian formulation; 6.10 Dynamics in Cartesian space; 6.11 Non-rigid-body effects; 6.12 Dynamic simulation; 6.13 Computational considerations.
- Ch. 7 (M): 7.2 General considerations; 7.3 Joint-space schemes (cubic, via points, higher-order, LSPB); 7.4 Cartesian-space schemes; 7.5 Geometric problems with Cartesian paths; 7.6 Path generation at run time; 7.7 Paths in a robot programming language; 7.8 Planning paths using the dynamic model; 7.9 Collision-free path planning.
- Ch. 9 (M): 9.2 Feedback and closed-loop control; 9.3 Second-order linear systems; 9.4 Control of second-order systems; 9.5 Control-law partitioning; 9.6 Trajectory-following control; 9.7 Disturbance rejection; 9.8 Continuous vs discrete-time control; 9.9 Modelling and control of a single joint; 9.10 Architecture of an industrial-robot controller.
- Ch. 10 (M): 10.2 Nonlinear and time-varying systems; 10.3 MIMO control systems; 10.4 The control problem for manipulators; 10.5 Practical considerations; 10.6 Current industrial-robot control systems; 10.7 Lyapunov stability analysis; 10.8 Cartesian-based control systems; 10.9 Adaptive control.
- Ch. 11 (M): 11.2 Industrial robots in assembly; 11.3 Framework for partially constrained tasks; 11.4 The hybrid position/force control problem; 11.5 Force control of a mass–spring system; 11.6 The hybrid position/force control scheme; 11.7 Current industrial-robot control schemes.

### Lynch & Park, *Modern Robotics* (V chapter list, Northwestern / Cambridge)

**Chapters:** 1 Preview; 2 Configuration Space; 3 Rigid-Body Motions; 4 Forward Kinematics; 5 Velocity Kinematics and Statics; 6 Inverse Kinematics; 7 Kinematics of Closed Chains; 8 Dynamics of Open Chains; 9 Trajectory Generation; 10 Motion Planning; 11 Robot Control; 12 Grasping and Manipulation; 13 Wheeled Mobile Robots.

**Sections** (V where the instructor's notes cite them: 2.1–2.5, 3.2.1, 3.2.3, 3.3.1, 5.1, 5.3, 6.2.1, 6.2.2; the rest M):
- 2.1 DOF of a rigid body; 2.2 DOF of a robot (2.2.1 joints, 2.2.2 Grübler); 2.3 C-space topology and representation; 2.4 Configuration and velocity constraints; 2.5 Task space and workspace.
- 3.1 Planar motions; 3.2 Rotations and angular velocities (3.2.1 rotation matrices, 3.2.2 angular velocities, 3.2.3 exponential coordinates); 3.3 Rigid-body motions and twists (3.3.1 HTM, 3.3.2 twists, 3.3.3 exponential coordinates of rigid motion); 3.4 Wrenches.
- 4.1 PoE (4.1.1 space form, 4.1.3 body form); 4.2 URDF.
- 5.1 Jacobian (space, body); 5.2 Statics; 5.3 Singularity analysis; 5.4 Manipulability.
- 6.1 Analytic IK; 6.2 Numerical IK (6.2.1 Newton–Raphson, 6.2.2 algorithm); 6.3 Inverse velocity kinematics.
- 8.1 Lagrangian formulation; 8.2 Single rigid body; 8.3 Newton–Euler inverse dynamics; 8.4 Closed form; 8.5 Forward dynamics; 8.6 Task-space dynamics; 8.7 Constrained dynamics; 8.9 Actuation, gearing and friction.
- 9.1 Definitions; 9.2 Point-to-point; 9.3 Polynomial via points; 9.4 Time-optimal time scaling.
- 11.1 Control system overview (V slides); 11.2 Error dynamics (V slides); 11.3 Motion control with velocity inputs; 11.4 Motion control with torque or force inputs (PID, computed torque, task space); 11.5 Force control; 11.6 Hybrid motion–force control; 11.7 Impedance control; 11.8 Low-level joint force/torque control. Cambridge confirms that the chapter covers joint- and task-space motion control, force control, hybrid motion–force control and impedance control (V); the section numbers are M.

### Spong, Hutchinson & Vidyasagar, 2nd ed. (Wiley 2020)

- Parts I–IV plus Appendices A Trigonometry, B Linear Algebra, C Lyapunov Stability, D Optimization (V).
- Chapters V: 1 Introduction; 2 Rigid Motions; 3 Forward Kinematics; 8 Independent Joint Control; 9 Nonlinear and Multivariable Control; 10 Force Control; 11 Vision-Based Control; 13 Underactuated Robots (13.2 Modeling, 13.7 Passivity-Based Control); 14 contains 14.7 Feedback Control of Driftless Systems (mobile robots).
- Chapters M: 4 Velocity Kinematics (the Jacobian); 5 Inverse Kinematics; 6 Dynamics; 7 Path and Trajectory Planning; 12 Geometric Nonlinear Control / Feedback Linearization (?); 14 Mobile Robots (?).

### Siciliano, Sciavicco, Villani & Oriolo (Springer 2009)

- **Chapters:** 8 Motion Control (pp. 303–361), 9 Force Control (pp. 363–405) and 10 Visual Servoing (pp. 407–467) are V (Springer's site counts front matter, so it shows them as 9/10/11).
- Others M: 1 Introduction; 2 Kinematics; 3 Differential Kinematics and Statics; 4 Trajectory Planning; 5 Actuators and Sensors; 6 Control Architecture; 7 Dynamics; 11 Mobile Robots; 12 Motion Planning.
- **Sections (M):**
  - Ch. 2: 2.4 Euler angles; 2.5 Angle and axis; 2.6 Unit quaternion; 2.7 Homogeneous transformations; 2.8 Direct kinematics; 2.9 Typical structures; 2.10 Joint vs operational space; 2.11 Kinematic calibration; 2.12 Inverse kinematics.
  - Ch. 3: 3.1 Geometric Jacobian; 3.3 Singularities; 3.4 Redundancy; 3.5 Inverse differential kinematics; 3.6 Analytical Jacobian; 3.7 IK algorithms; 3.8 Statics; 3.9 Manipulability ellipsoids.
  - Ch. 7: 7.1 Lagrange; 7.2 Notable properties; 7.4 Parameter identification; 7.5 Newton–Euler; 7.8 Operational-space dynamic model.
  - Ch. 8: 8.1 The control problem; 8.2 Joint-space control; 8.3 Decentralised control (8.3.1 independent joint control, 8.3.2 decentralised feedforward compensation); 8.4 Computed-torque feedforward; 8.5 Centralised control (8.5.1 PD + gravity compensation, 8.5.2 inverse dynamics, 8.5.3 robust, 8.5.4 adaptive); 8.6 Operational-space control (8.6.2 PD + gravity, 8.6.3 inverse dynamics); 8.7 Comparison. Springer snippets confirm the joint-space control and PD + gravity subsections (V).
  - Ch. 9: 9.1 Interaction with environment; 9.2 Compliance control; 9.3 Impedance control; 9.4 Force control; 9.5 Constrained motion; 9.6 Natural and artificial constraints; 9.7 Hybrid force/motion control.

### Corke, RVC 3rd ed. (Python; 16 chapters, V)

- 2 Representing Position and Orientation; 3 Time and Motion; 4–6 mobile robotics; **7 Robot Arm Kinematics; 8 Manipulator Velocity; 9 Dynamics and Control** (V); 10–14 vision; 15–16 vision-based control (titles ?). Chapters 2 and 7 were extensively rewritten for this edition (V).

### Murray–Li–Sastry (V chapter list, Taylor & Francis)

- 2 Rigid Body Motion; 3 Manipulator Kinematics; 4 Robot Dynamics and Control; 5–9 follow (5 Multifingered hand kinematics, 6 Hand dynamics and control, 7 Nonholonomic behaviour, 8 Nonholonomic motion planning, 9 Future prospects: M).
- Paden–Kahan subproblems are in Ch. 3 (M, §3.3).

### Control texts

- **Slotine & Li (V):** 2 Phase Plane; 3 Fundamentals of Lyapunov Theory; 4 Advanced Stability Theory; 5 Describing Function; 6 Feedback Linearization (p. 207); 7 Sliding Control; 8 Adaptive Control; 9 Control of Multi-Input Physical Systems (includes robot trajectory control and adaptive robot control).
- **Khalil 3rd (partly V):** four parts; 4 Lyapunov Stability (p. 111); 13 Feedback Linearization (13.2 output linearisation, 13.3 full-state, 13.4 state feedback control); 14 Nonlinear Design Tools (p. 551; 14.1 sliding mode). Passivity chapter number: ? (M: Ch. 6).
- **Ogata 5th (M; Pearson confirms root locus in Ch. 6):** 1 Introduction; 2 Mathematical Modelling; 3 Mechanical/Electrical; 4 Fluid/Thermal; 5 Transient and Steady-State Response; 6 Root Locus; 7 Frequency Response; 8 PID and Modified PID; 9 State-Space Analysis; 10 State-Space Design.
  - Note: one Pearson blurb says "Frequency-Response Analysis (Chapter 8)" (?). This conflicts, so verify against the edition the students use.
- **Nise (M):** 1 Introduction; 2 Modelling in the Frequency Domain; 3 Modelling in the Time Domain; 4 Time Response; 5 Reduction of Multiple Subsystems; 6 Stability; 7 Steady-State Errors; 8 Root Locus Techniques; 9 Design via Root Locus; 10 Frequency Response Techniques; 11 Design via Frequency Response; 12 Design via State Space; 13 Digital Control Systems. The 8th ed. adds an online chapter on optimal control and new sections on PID tuning and multivariable systems (V publisher blurb).
- **Lewis, Dawson & Abdallah 2nd ed. (V):** 1 Commercial Robot Manipulators; 2 Introduction to Control Theory; 3 Robot Dynamics; 4 Computed-Torque Control; 5 Robust Control; 6 Adaptive Control; 7 Advanced Control Techniques; 8 Neural Network Control; 9 Force Control; 10 Implementation and Software.

---

## (e) Prof. Nandi playlist findings

| # | Playlist | Result |
|---|---|---|
| 1 | *Mathematical Foundation of Robotics* (29 videos) | URL https://youtube.com/playlist?list=PLFWlHcAOSQboRkC6DtxKIqXmErq0oc7S3 , taken from the instructor's PDF (p. 58). Lecture map from the PDF: L2 two-link; L3 FK/IK concepts; L6 spatial transformation; L7 rotational transformation; L8–L12 orientation, angle–axis, gimbal lock, Euler solution; L13–14 quaternions; L15 spatial transformation explained; L16–17, L19 D-H principles I–II and more structures; L18 IK (D-H principle III); L20–21 PUMA 560 anatomy; L22 inverse pose kinematics; L23–25 Q&A and concluding unit; L26–29 dynamics and ChatGPT. The playlist page itself could not be opened. |
| 2 | Robot Motion Control playlist | **Not found and not verified.** No URL could be confirmed. Search engines (including `allowed_domains: youtube.com` queries for "Robot Motion Control, Lecture", "G C Nandi", "Nandi robotics lecture playlist") returned nothing attributable to him. |
| 3 | Case-studies playlist | **Not found and not verified.** |

**Related videos found** (both use the same "Title, Lecture-N" naming style as playlist 1, which suggests the same uploader, but this is **unverified**):
- "Mathematical Foundation of Robotics, Lecture-4 (Basics of Forward and Inverse Dynamics)": https://www.youtube.com/watch?v=dJA9kR1MVXM . Its description links to a Google Sites course page.
- "Introduction to Humanoid Robots, Lecture-1": https://www.youtube.com/watch?v=dNqMxiBGIio . Its metadata names Prof. G. C. Nandi, and its description mentions "Unit two ... modelling techniques for humanoid robots". This could belong to the case-studies playlist or to a separate humanoid course.
- Other Nandi-group videos (research demos, not lectures):
  - https://www.youtube.com/watch?v=LWRxkcoEAbk (hexapod)
  - https://www.youtube.com/watch?v=X_BYs4kbrQE (SMART robot)
  - https://www.youtube.com/watch?v=WtX9-ueIk4k (deep-learning demo)
  - https://www.youtube.com/watch?v=h-dffwsYIA8 (virtual tour of the Centre of Intelligent Robotics)

**How to resolve this:** ask the user for the links, or allow youtube.com and sites.google.com in the session network settings. Then open the channel's Playlists tab from playlist 1, or from Prof. Nandi's study-materials page (https://sites.google.com/iiita.ac.in/gcnandi/study-materials).

---

## Other facts useful to the main thread

- **Stanford CS223A (Winter 2026)** uses Siciliano et al. as its textbook (V), not Craig. The classic 2008 video series follows the Craig-style sequence: spatial descriptions, then FK, IK, Jacobians, dynamics, trajectory generation, position and force control.
- **The Modern Robotics free PDF:** the current authors' file is MR-v2.pdf (https://hades.mech.northwestern.edu/images/2/25/MR-v2.pdf). The instructor notes cite the older MR.pdf link (https://hades.mech.northwestern.edu/images/7/7f/MR.pdf), which was not confirmed live.
- **The instructor notes link a Craig PDF** at marsuniversity.github.io. It appears to be an unauthorised copy, so I deliberately left it out of REFERENCES.md. Recommend the main thread does not link it either.
- **Pinocchio on PyPI** is `pip install pin`; the PyPI package called `pinocchio` is unrelated (V via PyPI). Current versions (PyPI, Nov 2026): roboticstoolbox-python 1.5.0, modern-robotics 1.1.1, mujoco 3.17.0, drake 1.58.0, pin 4.1.0, control 0.10.2.
- **Sapienza's Robotics 2** now uses Siciliano, Villani, Oriolo & De Luca, *Foundations of Robotics* (Springer, 2025): https://link.springer.com/book/10.1007/978-3-031-85523-8
