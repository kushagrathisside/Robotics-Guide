# Mathematics Primer

> A self-contained mathematical foundation for **MFoR** (*Mathematical Foundations of Robotics*) and **RMC** (*Robot Motion Control*).

## Purpose

The two courses in this guide assume a working command of linear algebra, trigonometry, calculus, differential equations, elementary control theory, numerical methods and classical mechanics. Students arrive from many colleges and programmes, and they rarely share the same background: one has met the singular value decomposition but never the Laplace transform; another can tune a PID loop but has not seen an inertia tensor. This primer fills those gaps. It is not a substitute for a full mathematics course. It presents exactly the mathematics that Craig's *Introduction to Robotics*, Lynch and Park's *Modern Robotics*, Spong, Hutchinson and Vidyasagar's *Robot Modeling and Control* and Siciliano et al.'s *Robotics: Modelling, Planning and Control* use, in the notation those books use, with the robotics application stated alongside each idea.

Every chapter follows the same pattern:

1. **Why this matters in robotics**: a short motivation;
2. **Learning objectives** and a linked list of contents;
3. **Sections** that build intuition first, then give definitions, results and short derivations, each with a "where this is used" pointer into the robotics texts;
4. at least three **worked examples** with full solutions;
5. at least five **exercises**, with answers hidden in collapsible blocks;
6. **Further reading**, with chapter-level pointers to standard texts.

Every number in every worked example and exercise answer has been checked in Python (NumPy, SciPy and SymPy) by direct computation, symbolic algebra or simulation. The figures are reproducible: run `python math-primer/figures/make_figures.py` (requires `numpy`, `scipy` and `matplotlib`).

## Chapters

| Chapter | Topics |
|---|---|
| [M1 — Vectors and Geometry](M1-vectors-and-geometry.md) | vectors and frames, dot and cross products, projections, right-hand rule, triple products, lines and planes, the skew-symmetric matrix $[\boldsymbol{\omega}]_\times$ |
| [M2 — Matrices and Linear Maps](M2-matrices-and-linear-maps.md) | matrices as linear maps, products and composition, transpose, inverse, determinant as signed volume, rank and null space, orthogonal matrices and $SO(3)$, change of basis, homogeneous transforms |
| [M3 — Eigenvalues, the SVD and Least Squares](M3-eigen-svd-least-squares.md) | eigenvalues and eigenvectors, the rotation axis as an eigenvector, symmetric and positive-definite matrices, the SVD and the manipulability ellipsoid, pseudoinverse, least squares, damped least squares, condition number |
| [M4 — Trigonometry and the Two-Argument Arctangent](M4-trigonometry-and-atan2.md) | identities used in kinematics, $\operatorname{atan2}$ and quadrants, solving $a\cos\theta + b\sin\theta = c$, tangent half-angle substitution, law of cosines and the 2R inverse kinematics |
| [M5 — Calculus and Differential Equations](M5-calculus-and-differential-equations.md) | derivatives of vector and matrix functions, angular velocity, gradient and Jacobian, chain rule, Taylor linearisation, first- and second-order ODEs, $\omega_n$ and $\zeta$, PD-control preview, state-space form |
| [M6 — Laplace Transforms and Linear Systems](M6-laplace-and-linear-systems.md) | transform table and properties, partial fractions, transfer functions, poles and zeros, first- and second-order step responses, final value theorem, Routh–Hurwitz, frequency response |
| [M7 — Numerical Methods](M7-numerical-methods.md) | floating point, Newton–Raphson (scalar and vector, numerical IK), fixed-point iteration, gradient descent, Euler, semi-implicit Euler and RK4, re-orthonormalising rotation matrices |
| [M8 — Lagrangian Mechanics Primer](M8-lagrangian-mechanics-primer.md) | generalised coordinates, kinetic and potential energy, Euler–Lagrange equations, inertia tensor and parallel-axis theorem, structure of manipulator dynamics, planar two-link arm |

## How the primer maps to the courses

The MFoR column uses the chapter numbering of Craig, *Introduction to Robotics* (3rd/4th ed.), which the [MFoR notes](../MFoR/) follow. The RMC column names the control topics (and Craig's control chapters) in which each primer chapter is used. See [docs/REFERENCES.md](../docs/REFERENCES.md) for full bibliographic details.

| Primer chapter | Used in MFoR (Craig chapter) | Used in RMC |
|---|---|---|
| M1 Vectors and Geometry | Ch. 2 (frames, position vectors, direction cosines); Ch. 3 (joint axes, common normals in the Denavit–Hartenberg convention); Ch. 5 (velocity $\boldsymbol{\omega}\times\mathbf{r}$, moments $\mathbf{r}\times\mathbf{f}$, Jacobian columns); Ch. 6 (Newton–Euler recursion) | force and impedance control (wrenches); the skew-symmetry property $\dot M - 2C$ in passivity-based proofs |
| M2 Matrices and Linear Maps | Ch. 2 (rotation matrices, homogeneous transforms, transform equations); Ch. 3 (chains of link transforms); Ch. 5 (Jacobian rank, singularities, statics $\boldsymbol{\tau} = J^{T}\mathbf{F}$) | multivariable and operational-space control; coordinate changes of gains and errors |
| M3 Eigenvalues, SVD, Least Squares | Ch. 2 (equivalent angle–axis); Ch. 5 (singularities, manipulability); Ch. 6 (principal axes of inertia); Ch. 8 (well-conditioned designs) | Lyapunov stability ($P \succ 0$), positive-definite gain matrices, redundancy resolution, damped least squares, parameter identification |
| M4 Trigonometry and atan2 | Ch. 3 (forward kinematics with $c_{12}$, $s_{12}$); Ch. 4 (closed-form inverse kinematics) | workspace limits and joint-space set-points |
| M5 Calculus and ODEs | Ch. 5 (Jacobians, angular velocity); Ch. 6 (equations of motion in state-space form); Ch. 7 (derivatives of trajectory polynomials) | Ch. 9 (second-order error dynamics, PD gain design); Ch. 10 (linearisation, nonlinear control); state-space models |
| M6 Laplace and Linear Systems | Ch. 9 preview only | Ch. 9 (transfer functions, steady-state error, disturbance rejection, structural resonance); independent-joint PID control; frequency-domain tuning |
| M7 Numerical Methods | Ch. 2 (keeping rotations orthonormal); Ch. 4 (numerical inverse kinematics); Ch. 6 (simulation of dynamics) | simulation of closed-loop systems; discrete-time implementation; iterative IK in task-space controllers |
| M8 Lagrangian Mechanics | Ch. 6 (Lagrangian dynamics, inertia tensor, parallel-axis theorem) | Ch. 9–10 (model-based and computed-torque control); passivity and PD + gravity compensation; adaptive control via linearity in parameters |

## Suggested order

The chapters can be read in numerical order, but each course needs only some of them at first.

| If you are about to study... | Read first | Then, when you reach it |
|---|---|---|
| MFoR: spatial descriptions and kinematics (Craig Ch. 2–4) | M1, M2, M4 | M7 §7 (orthonormalisation) |
| MFoR: Jacobians and statics (Craig Ch. 5) | M3, M5 §1–3 | M7 §3 (numerical IK) |
| MFoR: dynamics and trajectories (Craig Ch. 6–7) | M8, M5 §4–6 | M7 §6 (simulation) |
| RMC: linear and independent-joint control (Craig Ch. 9) | M5 §5–6, M6 | M3 §3 (definiteness) |
| RMC: nonlinear, adaptive and force control (Craig Ch. 10–11) | M3 §3, M8 §5 | M1 §7 (skew-symmetry), M7 |

A student with a solid first course in linear algebra can skim M1–M2 and start at M3. A student with a control background can skim M6 but should still read M5 §5, which connects $\omega_n$ and $\zeta$ to PD gains.

## Notation conventions

The primer follows Craig's notation for frames and transforms and the common notation of the other texts elsewhere. Where the books differ, the chapter says so.

| Object | Notation | Example |
|---|---|---|
| Scalars | italic lower case or Greek | $a$, $\theta_1$, $\lambda$ |
| Vectors | bold lower case (bold capital for Craig's position vectors) | $\mathbf{p}$, $\boldsymbol{\omega}$, $\mathbf{P}$ |
| Unit vectors | hat | $\hat{\mathbf{x}}$, $\hat{\mathbf{k}}$, ${}^{A}\hat{X}_{B}$ |
| Matrices | italic capitals | $R$, $J$, $M(\mathbf{q})$; identity $I$ or $I_3$ |
| Frames | braces | $\lbrace A\rbrace$, $\lbrace B\rbrace$ |
| Vector written in frame $\lbrace A\rbrace$ | leading superscript | ${}^{A}\mathbf{P}$ |
| Orientation of $\lbrace B\rbrace$ relative to $\lbrace A\rbrace$ | leading super- and subscripts | ${}^{A}_{B}R$, so that ${}^{A}\mathbf{P} = {}^{A}_{B}R\,{}^{B}\mathbf{P}$ |
| Homogeneous transform | as above, $4\times4$ | ${}^{A}_{B}T$ |
| Origin of $\lbrace B\rbrace$ in $\lbrace A\rbrace$ | Craig's subscript | ${}^{A}\mathbf{P}_{B\,\text{org}}$ |
| Skew-symmetric (cross-product) matrix | square brackets | $[\boldsymbol{\omega}]_\times\mathbf{v} = \boldsymbol{\omega}\times\mathbf{v}$ (also written $S(\boldsymbol{\omega})$ or $[\boldsymbol{\omega}]$ in other texts) |
| Trigonometric shorthand | subscripts | $c_1 = \cos\theta_1$, $s_{12} = \sin(\theta_1 + \theta_2)$ |
| Euclidean norm | double bars | $\lVert\mathbf{p}\rVert$ |
| Transpose, inverse, pseudoinverse | superscripts | $A^{T}$, $A^{-1}$, $A^{+}$ |
| Time derivative | dot | $\dot{\mathbf{q}}$, $\ddot{\theta}$ |
| Positive-definite | curly inequality | $P \succ 0$ |
| Two-argument arctangent | argument order $(y, x)$ | $\operatorname{atan2}(y, x)$ |

Angles are in **radians** in every formula; degrees appear only in worked examples, for readability. Frames are always **right-handed**. In figures, coordinate axes are coloured $x$ red, $y$ green and $z$ blue, and are also labelled with text.

## Figures

All figures live in [`figures/`](figures/) and are generated by [`figures/make_figures.py`](figures/make_figures.py). To regenerate them after an edit:

```bash
python math-primer/figures/make_figures.py
```

## Further reading (whole primer)

* G. Strang, *Introduction to Linear Algebra* (M1–M3).
* E. Kreyszig, *Advanced Engineering Mathematics* (M1–M7).
* K. Ogata, *Modern Control Engineering* (M5–M6).
* B. Siciliano et al., *Robotics: Modelling, Planning and Control*, Appendices A (Linear Algebra), B (Rigid-Body Mechanics) and C (Feedback Control).
* M. W. Spong et al., *Robot Modeling and Control*, the appendices on trigonometry, linear algebra and Lyapunov stability.
* K. M. Lynch and F. C. Park, *Modern Robotics*, Appendix A (summary of useful formulas) and Appendix D (optimisation and Lagrange multipliers).
