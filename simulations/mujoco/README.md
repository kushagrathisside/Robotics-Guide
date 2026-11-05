# MuJoCo companion scripts

This directory contains small, self-contained MuJoCo programmes that check the
mathematics of the course against an independent physics engine. Each script
performs a numerical verification with explicit assertions, so that a script
which finishes without error *is* a successful experiment. Rendering of images
and animations is optional and is skipped gracefully when no OpenGL context is
available.

The background, including the anatomy of an MJCF model and the correspondence
between MuJoCo quantities and the notation of Craig's *Introduction to
Robotics*, is given in [`docs/MUJOCO.md`](../../docs/MUJOCO.md). Installation is
described in [`docs/INSTALLATION.md`](../../docs/INSTALLATION.md).

## Contents

| File | Course link | What it demonstrates |
|------|-------------|----------------------|
| `models/frames_demo.xml` | MFoR Ch. 2 | A free-floating body drawn as a red–green–blue coordinate frame, beside a translucent world frame. |
| `models/planar_2r.xml` | MFoR Ch. 3–6, RMC | A planar two-link arm with $L_1 = 1.0$ m, $L_2 = 0.8$ m, hinges about $z$, a tool `site` at the tip and two position servos. |
| `01_frames_and_quaternions.py` | MFoR Ch. 2 | Sets the body's orientation from Z-Y-X Euler angles and asserts that `data.xmat` and `data.xquat` equal the rotation matrix and quaternion computed by hand. |
| `02_planar_2r_forward_kinematics.py` | MFoR Ch. 3 and 5 | Sweeps 625 joint configurations and asserts that MuJoCo's tool position, tool orientation and site Jacobian agree with the analytic formulae to $10^{-9}$. |
| `03_planar_2r_pd_control.py` | RMC (PD control) | Drives the arm to a set-point with MuJoCo's position servos and with an explicit PD law applied through `qfrc_applied`, shows that the two coincide, and plots the response. |
| `outputs/` | — | Destination of the generated figures. Only small result images are kept under version control. |

### Teaching model and real robots

`planar_2r.xml` is deliberately a *minimal teaching model*: its link lengths
($1.0$ m and $0.8$ m), masses and gains are round numbers, so that every result
can be checked by hand. The notebooks also use real robots from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) through
the helper `rgkit.robots`, which downloads them on first use (see
[`docs/MUJOCO.md`](../../docs/MUJOCO.md#91-real-robots-in-the-course-notebooks-rgkitrobots)):

```python
from rgkit.robots import load_robot
model, data = load_robot("2r")      # the real Dynamixel 2R planar arm
model, data = load_robot("ur5e")    # Universal Robots UR5e (6 joints)
model, data = load_robot("panda")   # Franka Emika Panda (7 joints + gripper)
```

These models are not stored in this repository, and each carries its own
licence (see the `LICENSE` file downloaded with it). A good exercise is to
repeat the checks of script 02 on the Dynamixel 2R, reading its link lengths
from `model.body_pos` rather than assuming them.

## Running the scripts

From the repository root, with the course environment activated:

```bash
python simulations/mujoco/01_frames_and_quaternions.py
python simulations/mujoco/02_planar_2r_forward_kinematics.py
python simulations/mujoco/03_planar_2r_pd_control.py
```

Each script accepts `--no-render`, which skips the optional image or GIF and
leaves only the numerical checks (and, for script 03, the Matplotlib plot).

The expected outputs are:

| Script | Output files |
|--------|--------------|
| 01 | `outputs/01_frames_demo.png` (if rendering is available) |
| 02 | `outputs/02_planar_2r.png` (if rendering is available) |
| 03 | `outputs/03_pd_joint_angles.png` (always) and `outputs/03_pd_motion.gif` (if rendering is available) |

## Rendering on a machine without a display

MuJoCo selects its OpenGL back end from the environment variable `MUJOCO_GL`
when it is first imported. On Windows and macOS, and on a Linux desktop, the
default works. On a headless Linux machine (a server, a container, WSL without
a display, a continuous-integration runner) each script probes the headless
back ends `egl` and then `osmesa` in a child process and selects the first
that can render. You may also choose explicitly:

```bash
MUJOCO_GL=egl    PYOPENGL_PLATFORM=egl    python simulations/mujoco/03_planar_2r_pd_control.py
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python simulations/mujoco/03_planar_2r_pd_control.py
```

`egl` uses the graphics card (or Mesa's software EGL if no card is present);
`osmesa` is a pure software renderer that requires the system package
`libosmesa6` on Debian and Ubuntu. If neither is available the scripts print
"Rendering skipped" and still complete all their checks.

## Viewing the models interactively

On a computer with a display, the models can be explored in MuJoCo's viewer:

```bash
python -m mujoco.viewer --mjcf=simulations/mujoco/models/planar_2r.xml
```

Double-click a body to select it, hold Ctrl and drag with the mouse to
apply a force or torque to it, and use the *Control* panel on the right to move the
position servos. On macOS, scripts that use `mujoco.viewer.launch_passive`
must be started with `mjpython` instead of `python`.

## Exercises

1. In `01_frames_and_quaternions.py`, choose $\beta = 90^\circ$ and verify that
   different pairs $(\alpha, \gamma)$ with the same $\alpha - \gamma$ give the
   same `xmat`. Explain the result in terms of the singularity of Z-Y-X Euler
   angles (Craig, Chapter 2).
2. In `02_planar_2r_forward_kinematics.py`, print $\det J(q)$ along the sweep
   and confirm that it vanishes exactly when $\sin q_2 = 0$.
3. In `03_planar_2r_pd_control.py`, rotate the joint axes in the model so that
   the arm moves in a vertical plane (for example, `axis="0 1 0"` and the links
   along $x$). Observe the steady-state error under gravity, then remove it by
   adding the gravity torque `data.qfrc_bias` (computed with zero velocity) to
   the explicit PD law of Run B.
