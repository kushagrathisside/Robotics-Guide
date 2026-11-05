# Syllabus and chapter plan

This file is the master plan of the repository. It merges three sources: the topics of the two IIIT Allahabad courses
(see [§1](#1-the-two-courses)), the chapter structure of the standard textbooks, and the content of comparable courses at
other leading robotics programmes (see [docs/REFERENCES.md](docs/REFERENCES.md)). Each part below is organised
**chapter → notebook → topics**, i.e. three levels of depth. Status symbols: ✅ written and verified · 🚧 in progress · ⬜ planned.

## 1. The two courses

| | **MFoR** — Mathematical Foundations of Robotics | **RMC** — Robot Motion Control |
|:--|:--|:--|
| Nature | Introduction to robotics: spatial mathematics, kinematics, Jacobians, dynamics, trajectories | Control systems applied to robot motion: linear, model-based, robust, adaptive, force and learning-based control |
| Principal text | J. J. Craig, *Introduction to Robotics: Mechanics and Control* | Craig Ch. 9–11, with Spong et al. and Siciliano et al. for the nonlinear theory; Ogata/Nise for classical control |
| Parallel texts | Lynch & Park, *Modern Robotics*; Spong, Hutchinson & Vidyasagar; Siciliano et al.; Corke | Slotine & Li; Khalil; Lewis et al.; Sutton & Barto (learning) |
| Lectures | Prof. G. C. Nandi, playlist 1 (29 lessons); ThatsEngineering; Northwestern *Modern Robotics* | Prof. Nandi's playlist 2 (*Execution Level …*, 12 videos); Sapienza *Robotics 2*; Khatib (Stanford CS223A) |
| Case studies | Prof. Nandi's playlist 3, *Introduction to Humanoid and Collaborative Robots* (16 videos) | |

*IIIT Allahabad details (from public search results; the official PDFs could not be opened, so please verify):* RMC appears as
course code IT 4020 (4 credits) with topics "Robot Modeling Using Dynamics, Trajectory Planning; Control Systems for Robotics:
Robot Control Principles, Master–Slave Control Architecture, PD–PID Control, Computed Torque/Model-Based Methodology, Nonlinear
Control, Sensory Devices for Robots". The foundation course appears as IT 4030 *Foundation of Robotics* (2023 syllabus) and
as *Mathematical Foundations of Robotics* (earlier guidelines). The repository deliberately goes beyond both: it is meant to be
usable by a student at **any** college, and it is not organised around examinations.

## 2. Reading guide

* Do the **[Mathematics Primer](math-primer/README.md)** first if linear algebra, trigonometry, ODEs or Laplace transforms feel rusty.
* Follow MFoR chapters in order. RMC assumes MFoR Chapters 2, 3, 5 and 6.
* Every notebook runs in Google Colab (badge at the top) or locally ([installation guide](docs/INSTALLATION.md)).
* Simulations use [MuJoCo](docs/MUJOCO.md) and real robot models (UR5e, Franka Panda, KUKA iiwa, Dynamixel 2R, …).

---

## Part A — MFoR: Mathematical Foundations of Robotics

### Chapter 1 — Introduction, Joints and Degrees of Freedom ⬜
Folder `MFoR/01-introduction-and-degrees-of-freedom/`
| Notebook | Topics |
|:--|:--|
| 1.1 Robots, links and joints | manipulator anatomy; serial vs closed chains; the six lower pairs; naming robots (RRR, SCARA, …); classification: articulated, spherical, cylindrical, Cartesian, parallel; tour of real robot models |
| 1.2 Degrees of freedom | DOF of a rigid body (plane, space); Grübler–Kutzbach formula; four-bar, slider–crank, Stewart platform; redundancy |
| 1.3 Configuration space | explicit vs implicit representation; topology (circle, torus, SO(3)); holonomic and non-holonomic constraints |
| 1.4 Spaces of a robot | joint, actuator, Cartesian/task space; workspace, reachable vs dexterous; the FK/IK maps |

### Chapter 2 — Spatial Descriptions and Transformations 🚧
Folder `MFoR/02-spatial-descriptions-and-transformations/`
| Notebook | Topics |
|:--|:--|
| 2.1 Positions, orientations and frames ✅ | notation; position vectors; rotation matrices; SO(3); elementary rotations; frames; real robot link frames |
| 2.2 Mappings and operators ✅ | mapping between frames; chain rule; fixed vs current axes; non-commutativity; operators; skew-symmetric matrices |
| 2.3 Homogeneous transformations ✅ | 4×4 transform; SE(3); inverse; compound transforms; transform equations; free vectors; operators; UR5e workcell |
| 2.4 Fixed angles and Euler angles ⬜ | X-Y-Z fixed (roll–pitch–yaw); Z-Y-X and Z-Y-Z Euler angles; the 24 conventions; inverse problem with atan2; gimbal lock; teach-pendant orientation of a real arm |
| 2.5 Angle–axis and quaternions ⬜ | Rodrigues' formula; exponential coordinates; unit quaternions; conversions; SLERP; choice of representation |
| 2.6 Screw theory (beyond Craig) ⬜ | twists, se(3), adjoint, exponential coordinates of rigid motion, wrenches |
| 2.7 Practice problems ⬜ | graded numerical problems with full solutions |

### Chapter 3 — Forward Kinematics ⬜
Folder `MFoR/03-forward-kinematics/`
| Notebook | Topics |
|:--|:--|
| 3.1 Link parameters and frame assignment | link length, twist, offset, joint angle; modified (Craig) vs standard (Spong) D-H; frame-assignment recipe; link transform |
| 3.2 Worked manipulators | planar 2R/3R, RPR, articulated, SCARA, spherical, cylindrical, with D-H tables and symbolic FK |
| 3.3 Industrial arms | PUMA 560; UR5e and Franka Panda D-H tables validated against MuJoCo poses |
| 3.4 Product of exponentials | screw axes, space and body forms, comparison with D-H |
| 3.5 Joint, actuator and Cartesian space; practice | actuator–joint coupling; tool and station frames; practice problems |

### Chapter 4 — Inverse Kinematics ⬜
Folder `MFoR/04-inverse-kinematics/`
| Notebook | Topics |
|:--|:--|
| 4.1 Solvability | existence, workspace, multiple solutions, manipulator subspace, redundancy vs dexterity |
| 4.2 Closed-form solutions | algebraic and geometric methods; 2R, 3R planar; atan2; trigonometric toolbox |
| 4.3 Arms with spherical wrists | kinematic decoupling, Pieper's condition, PUMA-style and UR5e solutions, counting branches |
| 4.4 Numerical inverse kinematics | Newton–Raphson, gradient descent, pseudoinverse, damped least squares, convergence experiments |
| 4.5 Practice | graded problems with full solutions |

### Chapter 5 — Velocity Kinematics, Jacobians and Statics ⬜
Folder `MFoR/05-velocity-kinematics-and-statics/`
| Notebook | Topics |
|:--|:--|
| 5.1 Velocity of rigid bodies | linear and angular velocity; velocity propagation along links |
| 5.2 The manipulator Jacobian | column construction; frames; analytical vs geometric; Jacobian from MuJoCo for UR5e |
| 5.3 Singularities and manipulability | boundary/interior singularities; manipulability ellipsoids |
| 5.4 Statics | force propagation; τ = Jᵀ F; force ellipsoids; kineto-static duality |
| 5.5 Redundancy | pseudoinverse; null-space motion; Cartesian transformation of velocities and forces; practice |

### Chapter 6 — Manipulator Dynamics ⬜
Folder `MFoR/06-manipulator-dynamics/`
| Notebook | Topics |
|:--|:--|
| 6.1 Mass distribution | inertia tensor, parallel-axis theorem, principal axes |
| 6.2 Newton–Euler | recursive formulation, outward/inward iterations |
| 6.3 Lagrangian dynamics | kinetic and potential energy; M(q) q̈ + C(q,q̇) q̇ + g(q) = τ; 2R worked example |
| 6.4 Structure and properties | symmetric positive-definite M; skew-symmetry of Ṁ − 2C; linearity in parameters; friction, gearing, rotor inertia |
| 6.5 Forward dynamics and simulation | integration schemes; comparison of hand-derived M(q), g(q) with MuJoCo for a real arm; practice |

### Chapter 7 — Trajectory Generation ⬜
Folder `MFoR/07-trajectory-generation/`
| Notebook | Topics |
|:--|:--|
| 7.1 Joint-space trajectories | path vs trajectory; cubic and quintic polynomials; via points |
| 7.2 Trapezoidal profiles | linear segments with parabolic blends; bang-bang |
| 7.3 Cartesian trajectories | straight lines; orientation interpolation (angle–axis, SLERP); Cartesian-path pitfalls |
| 7.4 Time scaling and execution | time-optimal scaling; run-time trajectory generation; tracking in MuJoCo; practice |

### Chapter 8 — Case Studies: Collaborative Arms, Industrial Arms and Humanoids ⬜
Folder `MFoR/08-case-studies/` (companion to Prof. Nandi's playlist 3, *Introduction to Humanoid and Collaborative Robots*)
| Notebook | Topics |
|:--|:--|
| 8.1 A seven-joint collaborative arm | Sawyer/Panda/iiwa: D-H modelling, redundancy, worked D-H table from the course papers |
| 8.2 An industrial six-axis arm | UR5e and PUMA 560 pipeline: frames → D-H → FK → IK → Jacobian |
| 8.3 A humanoid | OP3 humanoid: kinematic tree, limb chains, centre of mass, support polygon |
| 8.4 Safety and human–robot interaction | collaborative operation, speed and separation, force limits (qualitative, with simulation) |

---

## Part B — RMC: Robot Motion Control

### Chapter 1 — Control-Systems Foundations ⬜
Folder `RMC/01-control-systems-foundations/`
| Notebook | Topics |
|:--|:--|
| 1.1 Modelling | transfer functions, block diagrams, DC-motor model, state-space, linearisation |
| 1.2 Time response and stability | first/second-order systems, ωₙ, ζ, overshoot, settling time; poles; Routh–Hurwitz; steady-state error |
| 1.3 Root locus and frequency response | root-locus rules, Bode and Nyquist, gain and phase margins |
| 1.4 PID control | structure, tuning (Ziegler–Nichols), anti-windup, discrete implementation |
| 1.5 State-space design | pole placement, observers, controllability; introduction to LQR |

### Chapter 2 — Independent-Joint Control ⬜
Folder `RMC/02-independent-joint-control/`
| Notebook | Topics |
|:--|:--|
| 2.1 The single-joint model | actuator, gearing, effective inertia and damping; coupling and gravity as disturbances |
| 2.2 Set-point control | P, PD, PID; critical damping; integral action; steady-state error under gravity |
| 2.3 Trajectory following | feedforward; control-law partitioning (model-based + servo portion) |
| 2.4 Practical limits | saturation, flexibility, resonance, sampling; industrial controller architecture |
| 2.5 Joint control in simulation | PD/PID on a MuJoCo UR5e and 2R arm; tuning experiments |

### Chapter 3 — Lyapunov Stability and Model-Based Control ⬜
Folder `RMC/03-lyapunov-and-model-based-control/`
| Notebook | Topics |
|:--|:--|
| 3.1 Lyapunov stability | definitions, direct method, LaSalle, passivity |
| 3.2 PD control with gravity compensation | energy-based proof; regulation of a manipulator |
| 3.3 Computed-torque control | inverse-dynamics (feedback linearising) control; tracking; gain choice |
| 3.4 Robustness to model error | parameter errors, unmodelled dynamics, performance comparison in MuJoCo |

### Chapter 4 — Robust, Adaptive and Operational-Space Control ⬜
Folder `RMC/04-robust-adaptive-and-operational-space-control/`
| Notebook | Topics |
|:--|:--|
| 4.1 Sliding-mode and robust control | sliding surfaces, chattering, boundary layers, Lyapunov redesign |
| 4.2 Adaptive control | linear-in-parameters regressor, Slotine–Li controller, parameter convergence |
| 4.3 Operational-space control | Jacobian-transpose PD, operational-space inverse dynamics (Khatib) |
| 4.4 Redundancy and null-space control | task priorities, secondary objectives |

### Chapter 5 — Force and Interaction Control ⬜
Folder `RMC/05-force-and-interaction-control/`
| Notebook | Topics |
|:--|:--|
| 5.1 Constrained motion | natural and artificial constraints, environment models |
| 5.2 Impedance and admittance control | stiffness/compliance control; Hogan's impedance; admittance |
| 5.3 Direct and hybrid force control | force control with inner loops; Raibert–Craig hybrid control |
| 5.4 Contact in simulation | contact experiments with MuJoCo |

### Chapter 6 — Teleoperation and Master–Slave Control ⬜
Folder `RMC/06-teleoperation-and-master-slave-control/`
| Notebook | Topics |
|:--|:--|
| 6.1 Architectures | unilateral vs bilateral; position–position, position–force, four-channel |
| 6.2 Transparency and stability | trade-offs; two-port models |
| 6.3 Time delay | passivity, scattering and wave variables |
| 6.4 Haptics and shared autonomy | force reflection, virtual fixtures; simulated master–slave pair |

### Chapter 7 — Optimal, Predictive and Learning-Based Control ⬜
Folder `RMC/07-optimal-predictive-and-learning-control/`
| Notebook | Topics |
|:--|:--|
| 7.1 Linear-quadratic regulation | LQR, Riccati equation, tracking |
| 7.2 Trajectory optimisation | iLQR/DDP, direct methods |
| 7.3 Model predictive control | receding horizon, constraints, simulation |
| 7.4 Reinforcement learning basics | MDPs, value methods, policy gradients |
| 7.5 Learning for robot control | RL on a MuJoCo arm, residual RL over computed torque, imitation, sim-to-real |

### Chapter 8 — Sensing, Implementation and Visual Servoing ⬜
Folder `RMC/08-sensing-and-implementation/`
| Notebook | Topics |
|:--|:--|
| 8.1 Sensors for robot control | encoders, tachometers, force/torque sensors, IMUs; noise and filtering |
| 8.2 Control architecture | real-time loops, sampling, latency; control software structure |
| 8.3 Visual servoing | image-based and position-based servoing |

### Chapter 9 — Extensions: Underactuated and Mobile Systems ⬜
Folder `RMC/09-underactuated-and-mobile-systems/`
| Notebook | Topics |
|:--|:--|
| 9.1 Underactuated systems | pendulum, acrobot, cart-pole; energy shaping; partial feedback linearisation |
| 9.2 Wheeled mobile robots | kinematic models, non-holonomic constraints, control of unicycle and differential-drive robots |

---

## Appendix — Mathematics Primer ✅
[`math-primer/`](math-primer/README.md): vectors; matrices; eigenvalues, SVD and least squares; trigonometry and atan2;
calculus and differential equations; Laplace transforms and linear systems; numerical methods; Lagrangian mechanics.

## How this plan was checked
The topic lists were cross-checked against the course structures and textbooks of the programmes listed in
[docs/REFERENCES.md](docs/REFERENCES.md); topics that a Craig-only course would miss (screw theory, numerical IK,
operational-space control, impedance control, MPC, reinforcement learning) are included deliberately.
