# Chapter 6 — Manipulator Dynamics

*Mathematical Foundations of Robotics · follows Craig, Ch. 6*

**Status: partly written.** Only notebook 6.1 and the tested library module `rgkit/dynamics.py` exist; notebooks 6.2–6.6 are planned.

## Overview and learning path

Dynamics relates joint torques to joint motion. The chapter starts from how mass is distributed in a link (inertia tensor), derives the
equations of motion by Newton–Euler and by Lagrange, studies the structure of `M(q) q̈ + C(q, q̇) q̇ + g(q) = τ`, adds joint-level effects
(gearing, friction), and ends with forward dynamics and simulation. Real robot models (Dynamixel 2R, UR5e, Panda) are cross-checked
against MuJoCo (`rgkit.dynamics.chain_from_mujoco` reproduces `M(q)`, `qfrc_bias` and `mj_inverse` to about 1e-13).

## Notebooks

| Notebook | Topics | Time | Prerequisites | Status |
|:--|:--|:--|:--|:--|
| [6.1 Mass Distribution and the Inertia Tensor](6.1-mass-distribution-and-the-inertia-tensor.ipynb) | centre of mass, inertia tensor, principal axes, parallel-axis theorem, composite bodies, MuJoCo inertia of UR5e/Panda/2R | 2.5 h | Ch. 2 | written |
| 6.2 Newton–Euler Dynamics | Euler's equation, torque-free motion, recursive outward/inward algorithm | — | 6.1 | planned |
| 6.3 Lagrangian Dynamics | energies, Christoffel symbols, 2R/3R/spatial closed forms (SymPy) | — | 6.2 | planned |
| 6.4 Structure and Properties | SPD `M`, skew-symmetry of `Ṁ − 2C`, regressor / linearity in parameters | — | 6.3 | planned |
| 6.5 Actuators, Gearing, Friction, Cartesian Dynamics | reflected inertia, friction models, `Λ(q)` | — | 6.4 | planned |
| 6.6 Forward Dynamics, Simulation and Practice | integrators, energy drift, MuJoCo comparison, practice problems | — | 6.5 | planned |

## Library support (already tested)

`rgkit/dynamics.py` (tests in `tests/test_dynamics.py`): inertia tensors of solids, parallel axis, principal axes, rigid-body equations,
recursive Newton–Euler (`rnea`), mass matrix, Coriolis matrix (Christoffel, complex-step), regressor, integrators, SymPy Lagrangian helpers,
and the MuJoCo bridge `chain_from_mujoco`.

## Where the textbooks cover this

| Topic | Craig | Lynch & Park | Siciliano et al. |
|:--|:--|:--|:--|
| Mass distribution, Newton–Euler, Lagrange, simulation | Ch. 6 | Ch. 8 | Ch. 7 |

## Not covered (yet)

Notebooks 6.2–6.6 above; flexible-link effects; closed-chain and floating-base dynamics.
