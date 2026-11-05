# Robotics Guide

**Open, notebook-based course material for *Mathematical Foundations of Robotics* (MFoR) and *Robot Motion Control* (RMC)**,
prepared for the courses at IIIT Allahabad and written so that a student at **any** college can use it to build a sound,
working understanding of robotics.

Every topic is a Jupyter notebook that reads like a section of a formal textbook — complete paragraphs, definitions,
derivations, tables, worked examples and exercises with answers — and that you can *run*: figures you can rotate, sliders
that change a robot's configuration, animations, and simulations of **real robot models** (UR5e, Franka Panda, KUKA iiwa,
Sawyer, ROBOTIS OP3, Dynamixel 2R) in the [MuJoCo](https://mujoco.org) physics engine. Hand calculations are checked against the
simulator, so you can trust what you read and see why it is true.

> **Status.** The repository is being built chapter by chapter; see the status column below and [SYLLABUS.md](SYLLABUS.md).
> Chapter 2 (spatial descriptions and transformations) is the reference chapter for format and depth.

## How to use it

| If you want to… | Do this |
|:--|:--|
| read it on GitHub | open any `.ipynb` file: text, equations, figures and outputs are rendered; animations appear as key-frame strips and as GIFs in each chapter's `README.md` |
| run it without installing anything | click the **Open in Colab** badge at the top of a notebook |
| run it locally | follow [docs/INSTALLATION.md](docs/INSTALLATION.md) (about ten minutes) |
| refresh the mathematics | start with the [Mathematics Primer](math-primer/README.md) |
| find textbooks and lectures | see [docs/REFERENCES.md](docs/REFERENCES.md) and the verified [video index](docs/VIDEO_INDEX.md) |

## Contents

### MFoR — Mathematical Foundations of Robotics (follows J. J. Craig, *Introduction to Robotics*)

| # | Chapter | Folder |
|--:|:--|:--|
| 1 | Introduction, joints and degrees of freedom | [`MFoR/01-…`](MFoR/01-introduction-and-degrees-of-freedom/) |
| 2 | Spatial descriptions and transformations | [`MFoR/02-…`](MFoR/02-spatial-descriptions-and-transformations/) |
| 3 | Forward kinematics (Denavit–Hartenberg, product of exponentials) | [`MFoR/03-…`](MFoR/03-forward-kinematics/) |
| 4 | Inverse kinematics | [`MFoR/04-…`](MFoR/04-inverse-kinematics/) |
| 5 | Velocity kinematics, Jacobians, singularities and statics | [`MFoR/05-…`](MFoR/05-velocity-kinematics-and-statics/) |
| 6 | Manipulator dynamics | [`MFoR/06-…`](MFoR/06-manipulator-dynamics/) |
| 7 | Trajectory generation | [`MFoR/07-…`](MFoR/07-trajectory-generation/) |
| 8 | Case studies: collaborative arms, industrial arms and humanoids | [`MFoR/08-…`](MFoR/08-case-studies/) |

### RMC — Robot Motion Control

| # | Chapter | Folder |
|--:|:--|:--|
| 1 | Control-systems foundations | [`RMC/01-…`](RMC/01-control-systems-foundations/) |
| 2 | Independent-joint control | [`RMC/02-…`](RMC/02-independent-joint-control/) |
| 3 | Lyapunov stability and model-based control | [`RMC/03-…`](RMC/03-lyapunov-and-model-based-control/) |
| 4 | Robust, adaptive and operational-space control | [`RMC/04-…`](RMC/04-robust-adaptive-and-operational-space-control/) |
| 5 | Force and interaction control | [`RMC/05-…`](RMC/05-force-and-interaction-control/) |
| 6 | Teleoperation and master–slave control | [`RMC/06-…`](RMC/06-teleoperation-and-master-slave-control/) |
| 7 | Optimal, predictive and learning-based control | [`RMC/07-…`](RMC/07-optimal-predictive-and-learning-control/) |
| 8 | Sensing, implementation and visual servoing | [`RMC/08-…`](RMC/08-sensing-and-implementation/) |
| 9 | Extensions: underactuated and mobile systems | [`RMC/09-…`](RMC/09-underactuated-and-mobile-systems/) |

## Repository layout

```
MFoR/, RMC/        the notebooks, one folder per chapter (each with its own README.md and assets/)
math-primer/       mathematics prerequisites as Markdown
rgkit/             shared, tested Python library used by every notebook (transforms, kinematics, plotting, robots, …)
tools/             notebook builder, figure extractor and quality checker; tools/sources/ holds the notebook sources
tests/             unit tests for rgkit
simulations/       stand-alone MuJoCo scripts and minimal models
docs/              installation, MuJoCo guide, references, video index, style guide
```

Real robot models are **not** stored here: `rgkit.robots` downloads them on first use from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (pinned to a fixed commit). Each model keeps its own
licence; see [docs/MUJOCO.md](docs/MUJOCO.md).

## Quality assurance

* `pytest` runs the unit tests of `rgkit` (rotation, transform, kinematics and every later module).
* `python tools/check_notebooks.py` verifies that each notebook is fully executed, has the required sections, has no broken
  links, stays within its size budget and cites only verified lecture links.
* Worked examples and exercise answers are recomputed in code inside the notebooks; wherever a quantity exists in MuJoCo
  (poses, Jacobians, mass matrices, control response) the hand-written result is compared with the simulator.

## Contributing

Read [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) first; it defines the structure, writing standards and verification rules for a
notebook. Errors in the mathematics are the most valuable thing you can report — please open an issue naming the notebook, the
cell and the correct result.

## AI assistance

This repository was developed with the help of AI. The notebooks, library code, tests and documentation were drafted with
**Claude** (Anthropic, via Claude Code) working under the direction of the repository owner, who set the scope, the structure and
the standards. The content is checked by code: worked examples and exercise answers are recomputed in the notebooks, and
hand-derived results are compared with the MuJoCo simulator. Nevertheless, treat the material as you would any textbook draft —
errors are possible, and corrections are welcome (see *Contributing*).

## Acknowledgements and sources

The syllabus draws on the courses taught at IIIT Allahabad (Prof. G. C. Nandi, Dr. K. R. Kaza) and on the standard textbooks and
open courses listed in [docs/REFERENCES.md](docs/REFERENCES.md). Notation follows J. J. Craig. The notebooks are original
expositions; they cite, and do not reproduce, the books and lectures they are based on.
