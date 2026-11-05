# MuJoCo in the Robotics Guide

This document explains why, and how, the physics simulator MuJoCo is used
alongside the mathematical notes of *Mathematical Foundations of Robotics*
(MFoR) and *Robot Motion Control* (RMC). It maps each element of a MuJoCo
model and each MuJoCo quantity onto the notation of J. J. Craig,
*Introduction to Robotics: Mechanics and Control*, which the courses follow.
Installation is covered separately in [`INSTALLATION.md`](INSTALLATION.md);
runnable examples are in [`simulations/mujoco/`](../simulations/mujoco/).

All code in this document was run against MuJoCo 3.17. Where the interface
has changed between releases, a version-independent form is given.

---

## Contents

1. [Why a simulator in a mathematics and control course?](#1-why-a-simulator-in-a-mathematics-and-control-course)
2. [Anatomy of an MJCF model](#2-anatomy-of-an-mjcf-model)
3. [Frames and poses: reading MuJoCo's kinematics](#3-frames-and-poses-reading-mujocos-kinematics)
4. [Jacobians](#4-jacobians-craig-chapter-5)
5. [Dynamics quantities](#5-dynamics-quantities-craig-chapter-6)
6. [Actuators, sensors and control](#6-actuators-sensors-and-control-rmc)
7. [Using MuJoCo inside Jupyter notebooks](#7-using-mujoco-inside-jupyter-notebooks)
8. [Chapter-by-chapter map](#8-chapter-by-chapter-map)
9. [Robot models: MuJoCo Menagerie](#9-robot-models-mujoco-menagerie)
10. [Other simulators and libraries](#10-other-simulators-and-libraries)
11. [References](#11-references)

---

## 1. Why a simulator in a mathematics and control course?

The courses are, first of all, courses in mathematics: rotation matrices,
homogeneous transforms, Jacobians, the manipulator equation
$\tau = M(q)\ddot q + V(q,\dot q) + G(q)$ and the stability of feedback laws.
A simulator does not replace any of this. It serves three subordinate but
valuable purposes.

1. **Independent verification.** A formula derived by hand, such as the forward
   kinematics of a two-link arm, can be compared numerically with the result
   of an engine written by other people for other purposes. Agreement to
   round-off ($10^{-15}$) is strong evidence that both the derivation and its
   implementation are right; disagreement locates an error precisely.
2. **A physical laboratory for control.** In RMC a control law is designed on a
   model and must then act on a "plant". MuJoCo provides that plant: rigid
   bodies with inertia, joint friction, actuator limits and contact, integrated
   in time. Students can observe overshoot, steady-state error, the effect of
   model mismatch on computed-torque control, or the behaviour of an impedance
   controller in contact with a surface, none of which is easily seen on paper.
3. **Visual intuition.** A rendered image of a frame rotated by given Euler
   angles, or a short video of an arm settling on a set-point, makes abstract
   objects concrete.

MuJoCo (*Multi-Joint dynamics with Contact*) is particularly suitable for
teaching. It is free and open source (Apache 2.0), is installed with a single
`pip install mujoco` on Windows, macOS, Linux and Google Colab, needs no
licence key, exposes every intermediate quantity of its computation (mass
matrix, bias forces, Jacobians) through a documented Python interface, and is
fast enough to simulate an arm many times faster than real time on a laptop.

A word of caution belongs here as well. A simulator is itself a model. MuJoCo
uses generalised (joint) coordinates, a particular contact model and a
particular numerical integrator. Results that depend delicately on contact
or on very stiff gains should be interpreted with the same care as any other
model-based prediction.

---

## 2. Anatomy of an MJCF model

A MuJoCo model is written in the XML format *MJCF*. The listing below is an
abridged version of the course's two-link planar arm,
[`simulations/mujoco/models/planar_2r.xml`](../simulations/mujoco/models/planar_2r.xml).

```xml
<mujoco model="planar_2r">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" gravity="0 0 -9.81"/>

  <worldbody>                                         <!-- frame {0}, the world -->
    <light pos="0.5 -1 3" dir="-0.2 0.3 -1"/>
    <geom name="floor" type="plane" size="3 3 0.05"/>

    <body name="link1" pos="0 0 0.1">                  <!-- frame {1} relative to {0} -->
      <joint name="q1" type="hinge" axis="0 0 1"/>     <!-- joint axis = z1 -->
      <geom type="capsule" fromto="0 0 0  1.0 0 0" size="0.04" mass="2.0"/>

      <body name="link2" pos="1.0 0 0">                <!-- frame {2}: L1 along x1 -->
        <joint name="q2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0  0.8 0 0" size="0.035" mass="1.5"/>
        <site name="tool" pos="0.8 0 0"/>              <!-- tool frame {T} -->
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="servo_q1" joint="q1" kp="200" kv="30"/>
    <position name="servo_q2" joint="q2" kp="120" kv="15"/>
  </actuator>

  <sensor>
    <framepos name="tool_pos" objtype="site" objname="tool"/>
  </sensor>
</mujoco>
```

The elements correspond to the concepts of the course as follows.

| MJCF element | Meaning in MuJoCo | Concept in Craig / the course |
|--------------|-------------------|-------------------------------|
| `<worldbody>` | The fixed root of the kinematic tree | The base or world frame $\{0\}$ (Craig's universe frame $\{U\}$) |
| `<body pos=… quat=…>` | A rigid body with its own coordinate frame, placed relative to its parent body | A link frame $\{i\}$; `pos` and `quat` (or `euler`, `axisangle`, `xyaxes`) define the fixed part of ${}^{i-1}_{\;\;i}T$ |
| Nesting of `<body>` | Parent–child relation of a kinematic tree | The serial chain $\{0\}\to\{1\}\to\cdots\to\{N\}$ |
| `<joint type="hinge" axis=…>` | A revolute degree of freedom about an axis expressed in the body's frame, through the point `pos` | Revolute joint $i$; with `axis="0 0 1"` the joint axis is $\hat Z_i$, exactly as in the Denavit–Hartenberg convention |
| `<joint type="slide">` | A prismatic degree of freedom | Prismatic joint; the joint variable is $d_i$ |
| `<freejoint/>` | Six degrees of freedom: position and unit quaternion | A free rigid body; used in MFoR Ch. 2 to illustrate an arbitrary frame |
| `<geom>` | Geometry for collision and visualisation; with `mass` or `density` it also defines inertia | The link's shape and, through it, its mass, centre of mass and inertia tensor (Craig Ch. 6) |
| `<inertial>` | Explicit mass, centre of mass and inertia tensor of a body | The parameters $m_i$, ${}^{C_i}I_i$ and $P_{C_i}$ of Craig Ch. 6 |
| `<site>` | A massless frame attached to a body | A tool frame $\{T\}$, a sensor location, or a target frame $\{G\}$ |
| `<camera>`, `<light>` | Rendering only | — |
| `<actuator>` (`motor`, `position`, `velocity`, `general`) | A force-producing element with a control input `ctrl` | A joint torque source (RMC); `position` implements a PD servo |
| `<sensor>` | A simulated measurement written to `data.sensordata` | Encoders (`jointpos`), tachometers (`jointvel`), wrist force/torque sensors (`force`, `torque`), etc. |
| `<option>` | Global settings: `timestep`, `gravity`, `integrator` | The sampling of the simulated plant |
| `<default>` | Default attribute values for classes of elements | — |

### 2.1 Encoding a Denavit–Hartenberg table

Craig uses the *modified* D-H convention, in which

$$
{}^{i-1}_{\;\;i}T = R_X(\alpha_{i-1})\,D_X(a_{i-1})\,R_Z(\theta_i)\,D_Z(d_i).
$$

Because $R_Z(\theta_i)$ and $D_Z(d_i)$ commute, this equals a fixed transform
$R_X(\alpha_{i-1})\,D_X(a_{i-1})\,D_Z(d_i)$ followed by a rotation about the new
$z$ axis. A revolute link $i$ is therefore written in MJCF as

```xml
<body pos="a_{i-1}   -sin(alpha_{i-1}) d_i   cos(alpha_{i-1}) d_i"
      quat="cos(alpha_{i-1}/2)  sin(alpha_{i-1}/2)  0  0">
  <joint type="hinge" axis="0 0 1"/>
  ...
</body>
```

with the numerical values substituted, and the next link nested inside it.
For a prismatic joint, use `type="slide"` with the same axis and put the
constant offset $\theta_i$ into `quat`. We verified this mapping numerically
for a four-link chain with arbitrary parameters: MuJoCo's body poses agreed with
the product of D-H matrices to $4\times10^{-16}$.

The joint variable that MuJoCo reports is measured from the configuration
written in the XML file (`qpos0`). If a D-H table has a constant offset in
$\theta_i$, either include it in `quat` or add it when comparing angles.

---

## 3. Frames and poses: reading MuJoCo's kinematics

### 3.1 `MjModel`, `MjData` and the forward pass

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("simulations/mujoco/models/planar_2r.xml")
data = mujoco.MjData(model)

data.qpos[:] = np.deg2rad([30.0, 60.0])   # joint angles q = (q1, q2)
mujoco.mj_forward(model, data)            # kinematics, dynamics, sensors
```

`MjModel` holds everything that is constant (geometry, masses, gains);
`MjData` holds the state (`qpos`, `qvel`, `time`), the inputs (`ctrl`,
`qfrc_applied`) and every quantity computed from them. **Nothing in `MjData`
is updated until a computation function is called.** After changing `qpos`,
call `mj_forward` (everything) or `mj_kinematics` (poses only).

### 3.2 Pose fields

| Field | Shape | Meaning, relative to the world frame $\{0\}$ |
|-------|-------|-----------------------------------------------|
| `data.xpos[b]` | (3,) | Origin of body frame $b$: ${}^{0}P_{b\,\mathrm{ORG}}$ |
| `data.xmat[b]` | (9,) | Rotation ${}^{0}_{b}R$, **flattened in row-major order**; use `.reshape(3, 3)` |
| `data.xquat[b]` | (4,) | The same rotation as a unit quaternion $[w, x, y, z]$ (scalar first) |
| `data.xipos[b]`, `data.ximat[b]` | (3,), (9,) | Centre of mass and principal inertia axes of body $b$ |
| `data.site_xpos[s]`, `data.site_xmat[s]` | (3,), (9,) | Pose of site $s$ (e.g. the tool frame) |
| `data.geom_xpos[g]`, `data.geom_xmat[g]` | (3,), (9,) | Pose of geom $g$ |
| `data.xanchor[j]`, `data.xaxis[j]` | (3,), (3,) | A point on joint $j$'s axis and the axis direction |

Objects may be addressed by integer index or, more readably, by name:
`data.body("link2").xpos`, `data.site("tool").xmat`, or
`mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool")`.

### 3.3 Comparing with hand-computed homogeneous transforms

```python
def homogeneous(R, p):
    """4x4 homogeneous transform from a rotation matrix and a position vector."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T

T_0_1 = homogeneous(data.body("link1").xmat.reshape(3, 3), data.body("link1").xpos)
T_0_2 = homogeneous(data.body("link2").xmat.reshape(3, 3), data.body("link2").xpos)
T_0_T = homogeneous(data.site("tool").xmat.reshape(3, 3), data.site("tool").xpos)

T_1_2 = np.linalg.inv(T_0_1) @ T_0_2      # relative transform: inverse, then compound

# Hand computation for the planar arm: rotation about z by q2, then L1 along x.
q1, q2 = data.qpos
c2, s2 = np.cos(q2), np.sin(q2)
T_1_2_hand = np.array([[c2, -s2, 0, 1.0],
                       [s2,  c2, 0, 0.0],
                       [0,    0, 1, 0.0],
                       [0,    0, 0, 1.0]])
assert np.allclose(T_1_2, T_1_2_hand, atol=1e-12)
```

The script [`02_planar_2r_forward_kinematics.py`](../simulations/mujoco/02_planar_2r_forward_kinematics.py)
performs this comparison for the tool frame over 625 configurations.

### 3.4 Conventions that catch students out

| Topic | MuJoCo | Other common conventions |
|-------|--------|--------------------------|
| Quaternion order | $[w, x, y, z]$ (scalar first) | SciPy's `Rotation.as_quat()` returns $[x, y, z, w]$ by default |
| Rotation matrix storage | `xmat` is a flat 9-vector, row-major | Reshape with `.reshape(3, 3)`; do not transpose |
| Euler angles in `mju_euler2Quat(quat, euler, seq)` | Lower-case letters denote *intrinsic* rotations (about moving axes), upper-case *extrinsic*. Craig's Z-Y-X Euler angles are therefore `"zyx"` | SciPy's `Rotation.from_euler` uses the opposite case convention |
| Angles in XML | Degrees unless `<compiler angle="radian"/>`; the course's models use radians | Python functions always use radians |
| Angular velocity of a free joint | `qvel[3:6]` is expressed in the **body** frame | Many texts use the world frame; convert with `xmat` |
| Joint zero | Measured from the pose written in the XML (`qpos0`) | D-H offsets must be added explicitly |

The script [`01_frames_and_quaternions.py`](../simulations/mujoco/01_frames_and_quaternions.py)
demonstrates the first three rows with assertions.

---

## 4. Jacobians (Craig Chapter 5)

MuJoCo computes the Jacobian of any point attached to any body:

| Function | Point whose Jacobian is returned |
|----------|----------------------------------|
| `mj_jac(model, data, jacp, jacr, point, body)` | A world-frame point `point` that moves with body `body` |
| `mj_jacBody(model, data, jacp, jacr, body)` | The origin of the body frame |
| `mj_jacBodyCom(model, data, jacp, jacr, body)` | The body's centre of mass |
| `mj_jacSite(model, data, jacp, jacr, site)` | The origin of a site (e.g. the tool frame) |

`jacp` and `jacr` are $3\times n_v$ arrays for the translational and rotational
parts; either may be `None`. Both are expressed in the **world frame**, so that

$$
{}^{0}v_T = J_p(q)\,\dot q, \qquad {}^{0}\omega_T = J_r(q)\,\dot q,
\qquad {}^{0}J(q) = \begin{bmatrix} J_p \\ J_r \end{bmatrix},
$$

which is Craig's ${}^{0}J(q)$ with the linear velocity stacked above the
angular velocity. The Jacobian in the tool frame is
${}^{T}J = \operatorname{diag}({}^{T}_{0}R,\, {}^{T}_{0}R)\,{}^{0}J$.

```python
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_kinematics(model, data)
mujoco.mj_comPos(model, data)                 # required by the mj_jac* family
mujoco.mj_jacSite(model, data, jacp, jacr, model.site("tool").id)

q1, q2 = data.qpos
J_hand = np.array([[-1.0 * np.sin(q1) - 0.8 * np.sin(q1 + q2), -0.8 * np.sin(q1 + q2)],
                   [ 1.0 * np.cos(q1) + 0.8 * np.cos(q1 + q2),  0.8 * np.cos(q1 + q2)]])
assert np.allclose(jacp[:2], J_hand, atol=1e-12)
assert np.allclose(jacr, [[0, 0], [0, 0], [1, 1]], atol=1e-12)
```

The `mj_jac*` functions require the outputs of `mj_kinematics` and
`mj_comPos`; `mj_forward` computes both. The resulting Jacobian supports the
exercises of MFoR Ch. 5 (singularities, $\det J = L_1 L_2 \sin q_2$ for the
planar arm, static force mapping $\tau = J^{\mathsf T} F$) and the resolved-rate
and operational-space controllers of RMC.

---

## 5. Dynamics quantities (Craig Chapter 6)

MuJoCo writes the equations of motion in the form

$$
M(q)\,\ddot q + c(q,\dot q) = \tau_{\text{actuator}} + \tau_{\text{passive}} + \tau_{\text{applied}} + \tau_{\text{constraint}},
$$

where $c(q,\dot q)$, the *bias force*, collects Coriolis, centrifugal and
gravitational terms. In Craig's notation $c = V(q,\dot q) + G(q)$.

| Quantity | MuJoCo | Craig |
|----------|--------|-------|
| Mass (inertia) matrix | `mj_fullM` (dense) from the sparse factor stored in `data` | $M(q)$ |
| Coriolis, centrifugal and gravity | `data.qfrc_bias` | $V(q,\dot q) + G(q)$ |
| Gravity alone | `data.qfrc_bias` evaluated with `qvel = 0` | $G(q)$ |
| Inverse dynamics | `mj_inverse`; result in `data.qfrc_inverse` | Iterative Newton–Euler formulation |
| Recursive Newton–Euler | `mj_rne(model, data, flg_acc, result)` | Outward and inward Newton–Euler iterations |
| Forward dynamics | `mj_forward`; result in `data.qacc` | $\ddot q = M^{-1}(\tau - V - G)$ |
| Joint damping, springs | `data.qfrc_passive` | Joint friction (viscous) |
| Rotor inertia | `armature` attribute; added to the diagonal of $M$ | Reflected motor inertia |

The interface of `mj_fullM` changed in recent releases; the following form
works with both:

```python
def mass_matrix(model, data):
    """Dense joint-space inertia matrix M(q); requires mj_forward (or mj_crb)."""
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)        # recent releases
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)     # earlier 3.x releases
    return M

data.qpos[:] = [0.3, 0.7]
data.qvel[:] = [0.5, -0.2]
mujoco.mj_forward(model, data)
M = mass_matrix(model, data)
bias = data.qfrc_bias.copy()                    # V(q, qdot) + G(q)

# Inverse dynamics: the torque that produces a chosen acceleration.
data.qacc[:] = [1.0, 2.0]
mujoco.mj_inverse(model, data)
tau = data.qfrc_inverse                         # includes the joint damping
assert np.allclose(tau, M @ data.qacc + bias + model.dof_damping * data.qvel)
```

For the course's planar arm, which moves in a horizontal plane, $G(q) = 0$ and
`qfrc_bias` contains only velocity-product terms. The matrix `M` includes the
`armature` of each joint (0.01 kg m² in the model), which should be added to
the diagonal of a hand-derived $M(q)$ before comparison, or set to zero in
the XML.

---

## 6. Actuators, sensors and control (RMC)

### 6.1 Ways of applying a joint torque

| Mechanism | How to use it | Typical use in RMC |
|-----------|---------------|--------------------|
| `<motor joint=… gear=…>` | `data.ctrl[i] = tau_i` (torque $=$ `gear` $\times$ `ctrl`) | Any controller computed in Python: PD, PID, computed torque, sliding mode |
| `<position joint=… kp=… kv=…>` | `data.ctrl[i] = q_desired_i`; force $= k_p(\text{ctrl} - q) - k_v\dot q$ | A built-in PD servo; the low-level loop of an industrial arm |
| `<velocity joint=… kv=…>` | `data.ctrl[i] = qdot_desired_i`; force $= k_v(\text{ctrl} - \dot q)$ | Velocity loops, resolved-rate control |
| `data.qfrc_applied` | Write a generalised force directly | Prototyping a controller without editing the model |
| `data.xfrc_applied[b]` | A Cartesian force and torque on body $b$ | Disturbances, external wrenches in force control |

Actuator saturation is modelled by `ctrlrange` (limits on the input) and
`forcerange` (limits on the output force), which allows integrator wind-up
and torque limits to be studied. All actuators can be switched off at once
with `model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION`.

### 6.2 The control loop

A digital controller running at the simulation rate has the following
structure:

```python
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION   # use qfrc_applied only
mujoco.mj_resetData(model, data)
q_d = np.deg2rad([60.0, -45.0])
Kp, Kd = np.diag([100.0, 100.0]), np.diag([20.0, 20.0])

for _ in range(3000):                          # 3 s with a 1 ms time step
    mujoco.mj_forward(model, data)             # M(q), bias, sensors at the current state
    e, e_dot = q_d - data.qpos, -data.qvel
    # Computed-torque control (Craig, Chapter 10): tau = M (Kp e + Kd e_dot) + V + G
    M = mass_matrix(model, data)
    data.qfrc_applied[:] = M @ (Kp @ e + Kd @ e_dot) + data.qfrc_bias \
                           + model.dof_damping * data.qvel
    mujoco.mj_step(model, data)

assert np.allclose(data.qpos, q_d, atol=1e-3)
model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)   # re-enable
```

Because the controller cancels the dynamics exactly, each joint error obeys
$\ddot e + K_d\dot e + K_p e = 0$, here critically damped with
$\omega_n = 10$ rad/s. Replacing `M` and `qfrc_bias` by estimates with
deliberate errors is an instructive exercise in robustness (RMC, nonlinear
control). The script [`03_planar_2r_pd_control.py`](../simulations/mujoco/03_planar_2r_pd_control.py)
implements independent-joint PD control in both of the ways listed above and
shows that they coincide to $10^{-16}$.

To run a controller at a lower rate than the simulation, update `qfrc_applied`
or `ctrl` only every $k$-th step and hold it in between (a zero-order hold).

### 6.3 Sensors, contact and force control

Sensors are declared in `<sensor>` and read from `data.sensordata` after
`mj_forward` or `mj_step`; `data.sensor("tool_pos").data` selects one by
name. A `force` and a `torque` sensor attached to a wrist `site` provide the
measurement used in hybrid position/force control (Craig, Ch. 11). Contacts
are listed in `data.contact[:data.ncon]`, and `mujoco.mj_contactForce`
returns the contact wrench in the contact frame. The stiffness and damping of
contact are set by the `solref` and `solimp` attributes of a geom, which make
it possible to study an impedance controller pressing against surfaces of
different stiffness.

---

## 7. Using MuJoCo inside Jupyter notebooks

The notebooks of the course use MuJoCo directly, without a separate window:
the simulation runs in the notebook's kernel, and images and short videos are
rendered offscreen and displayed inline. The same code runs on Google Colab
and on a local installation. The helper module `rgkit.mujoco_utils` wraps the
steps below; this section explains what those helpers do.

### 7.1 First cell: choose the rendering back end

MuJoCo fixes its OpenGL back end when it is first imported, so the choice
must be made before `import mujoco`:

```python
import os, platform
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")          # or "osmesa" (software)
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
import mujoco
```

On Windows, macOS and a Linux desktop the default needs no setting. On Colab
use `osmesa` on a CPU runtime or `egl` on a GPU runtime (see
[`INSTALLATION.md`](INSTALLATION.md#74-rendering-without-a-display-servers-colab-wsl-ci)).
To change the back end afterwards the kernel must be restarted.

### 7.2 Building a model in the notebook

A model may be loaded from a file, written as an XML string inside the
notebook, or constructed programmatically with `MjSpec`:

```python
# (a) From the repository.
model = mujoco.MjModel.from_xml_path("simulations/mujoco/models/planar_2r.xml")

# (b) From a string: convenient for small, self-contained examples.
xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.03" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)

# (c) Programmatically, e.g. in a loop over the rows of a D-H table.
spec = mujoco.MjSpec()
spec.worldbody.add_light(pos=[0, 0, 3])
body = spec.worldbody.add_body(name="pendulum", pos=[0, 0, 1])
body.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0.5, 0, 0],
              size=[0.03, 0, 0], mass=1.0)
model = spec.compile()
print(spec.to_xml())          # the equivalent MJCF, useful for checking
data = mujoco.MjData(model)
```

### 7.3 Rendering a frame to a NumPy image

`mujoco.Renderer` draws the current state of `data` into an RGB array of
shape `(height, width, 3)` and type `uint8`:

```python
model = mujoco.MjModel.from_xml_path("simulations/mujoco/models/planar_2r.xml")
data = mujoco.MjData(model)
data.qpos[:] = np.deg2rad([30.0, 60.0])
mujoco.mj_forward(model, data)

with mujoco.Renderer(model, height=360, width=480) as renderer:
    renderer.update_scene(data, camera="overview")    # a camera defined in the XML
    image = renderer.render()

import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis("off")
plt.show()
# or, equivalently:  import mediapy; mediapy.show_image(image)
```

Useful options:

* **Free camera.** Instead of a named camera, pass an `MjvCamera`:

  ```python
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  cam.azimuth, cam.elevation, cam.distance = 135.0, -25.0, 3.0
  cam.lookat[:] = [0.6, 0.3, 0.1]
  ```

  and call `renderer.update_scene(data, camera=cam)`.
* **Drawing coordinate frames.** This is especially valuable in MFoR:

  ```python
  opt = mujoco.MjvOption()
  opt.frame = mujoco.mjtFrame.mjFRAME_BODY      # or mjFRAME_SITE, mjFRAME_WORLD
  renderer.update_scene(data, camera=cam, scene_option=opt)
  ```

* **Image size.** The image may not exceed the model's offscreen buffer
  (640 × 480 by default). The course models enlarge it with
  `<visual><global offwidth="800" offheight="600"/></visual>`; otherwise
  MuJoCo raises `ValueError: Image width ... > framebuffer width ...`.
* **Resources.** A `Renderer` owns an OpenGL context. Use it in a `with`
  block, or call `renderer.close()`, rather than creating a new one in every
  cell.

### 7.4 Short videos and GIFs

Simulate, keeping one frame every $1/\text{fps}$ seconds:

```python
model = mujoco.MjModel.from_xml_path("simulations/mujoco/models/planar_2r.xml")
data = mujoco.MjData(model)
data.ctrl[:] = np.deg2rad([60.0, -45.0])      # set-points of the position servos

frames, fps, duration = [], 30, 2.0
with mujoco.Renderer(model, height=240, width=320) as renderer:
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * fps:
            renderer.update_scene(data, camera="overview")
            frames.append(renderer.render())
```

The frames can then be shown or saved in several ways:

| Method | Code | Requirement |
|--------|------|-------------|
| Inline video | `mediapy.show_video(frames, fps=fps)` | `ffmpeg` |
| MP4 file | `mediapy.write_video("arm.mp4", frames, fps=fps)` | `ffmpeg` |
| GIF file | `from PIL import Image; imgs = [Image.fromarray(f) for f in frames]; imgs[0].save("arm.gif", save_all=True, append_images=imgs[1:], duration=1000 // fps, loop=0)` | Pillow only |
| Inline, no `ffmpeg` | `from IPython.display import Image as IPImage; IPImage("arm.gif")` | Pillow only |
| Contact sheet | `mediapy.show_images(frames[::10])` | None |

Keep videos short (a few seconds at 320 × 240) so that saved notebooks remain
small; clear large outputs before committing a notebook.

### 7.5 Interaction with sliders

An `ipywidgets` slider can drive the joint angles and re-render a still image.
This is an effective way to explore forward kinematics:

```python
import ipywidgets as widgets
import mediapy

renderer = mujoco.Renderer(model, height=300, width=400)

def show(q1_deg=30.0, q2_deg=60.0):
    data.qpos[:] = np.deg2rad([q1_deg, q2_deg])
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="overview")
    mediapy.show_image(renderer.render())
    print("tool position:", data.site("tool").xpos.round(4))

widgets.interact(show, q1_deg=(-180.0, 180.0, 1.0), q2_deg=(-180.0, 180.0, 1.0))
```

The renderer is created once, outside the callback, so that moving a slider
does not create a new OpenGL context. Call `renderer.close()` when it is no
longer needed.

### 7.6 The interactive viewer and notebooks

The interactive viewer (`python -m mujoco.viewer`, `mujoco.viewer.launch`,
`mujoco.viewer.launch_passive`) opens a native desktop window. It is
therefore unavailable on Colab and on any machine without a display.
`launch` blocks the kernel until the window is closed, and on macOS
`launch_passive` must be run under the `mjpython` launcher, which a Jupyter
kernel is not. On a local Windows or Linux desktop `launch_passive` may be
started from a notebook, but its behaviour there is less predictable than in
a script. For these reasons the notebooks rely only on offscreen rendering,
and the viewer is recommended for exploring the XML models from a terminal:

```bash
python -m mujoco.viewer --mjcf=simulations/mujoco/models/planar_2r.xml
```

---

## 8. Chapter-by-chapter map

The table lists, for each chapter of Craig's book that the courses follow,
the MuJoCo features that support it. Chapter titles are those of Craig's
text; the course notebooks follow the same order.

| Craig chapter | Course | Mathematical content | MuJoCo features | Example |
|---------------|--------|----------------------|-----------------|---------|
| 2. Spatial descriptions and transformations | MFoR | Rotation matrices, Euler angles, quaternions, homogeneous transforms | `freejoint`, `qpos` quaternion, `xpos`, `xmat`, `xquat`, `mju_euler2Quat`, `mju_mat2Quat`, `mju_quat2Mat`, `mju_mulQuat`; rendering with `mjFRAME_BODY` | `01_frames_and_quaternions.py` |
| 3. Manipulator kinematics | MFoR | D-H parameters, link transforms, forward kinematics | Nested `body` + `hinge`/`slide`; `site` as tool frame; `mj_kinematics`; `site_xpos`, `site_xmat` | `02_planar_2r_forward_kinematics.py` |
| 4. Inverse manipulator kinematics | MFoR | Closed-form and numerical IK | Use `site_xpos` to *check* an IK solution; damped least-squares IK with `mj_jacSite`; `mocap` bodies as targets | — |
| 5. Jacobians: velocities and static forces | MFoR | Velocity propagation, Jacobian, singularities, $\tau = J^{\mathsf T}F$ | `mj_jac`, `mj_jacSite`, `mj_jacBody`; `xfrc_applied` for static forces; `mj_objectVelocity` | Script 02 (Jacobian check) |
| 6. Manipulator dynamics | MFoR | Newton–Euler, Lagrange, $M(q)$, $V$, $G$ | `mj_fullM`, `qfrc_bias`, `mj_inverse`, `mj_rne`, `body_mass`, `body_inertia`, `armature`, `damping` | Section 5 above |
| 7. Trajectory generation | MFoR | Cubic and quintic polynomials, LSPB, via points | Feed `ctrl` (position servos) or a computed torque with $q_d(t)$; `sensordata` to record the tracking error | — |
| 9. Linear control of manipulators | RMC | Second-order systems, PD and PID, independent-joint control, disturbance rejection | `position` and `motor` actuators, `kp`/`kv`, `ctrlrange`, `forcerange`; `qfrc_applied` for explicit laws; gravity as a disturbance | `03_planar_2r_pd_control.py` |
| 10. Nonlinear control of manipulators | RMC | Computed torque (partitioned control), robustness, Lyapunov methods, adaptive control | `mj_fullM` and `qfrc_bias` as the model in $\tau = \hat M(\ddot q_d + K_v\dot e + K_p e) + \hat V + \hat G$; perturb `body_mass` to create model mismatch | Section 6.2 above |
| 11. Force control of manipulators | RMC | Hybrid position/force control, impedance and admittance control | `force`/`torque` sensors at a `site`; contact (`data.contact`, `mj_contactForce`); `solref`/`solimp` for environment stiffness; `xfrc_applied` | — |

---

## 9. Robot models: MuJoCo Menagerie

[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) is a
curated collection of high-quality MJCF models maintained by Google DeepMind.
Each model lives in its own directory with a `scene.xml` (robot plus floor and
lights), a robot file, meshes and its own `LICENSE`. Models relevant to the
course include:

| Robot | Directory | Degrees of freedom | Licence | Suggested use |
|-------|-----------|--------------------|---------|---------------|
| Franka Emika Panda | `franka_emika_panda` | 7 (+ gripper) | Apache-2.0 | Redundant arm; Jacobian null space; computed torque |
| Universal Robots UR5e | `universal_robots_ur5e` | 6 | BSD-3-Clause | Classical 6R industrial arm; closed-form IK checks |
| KUKA LBR iiwa 14 | `kuka_iiwa_14` | 7 | BSD-3-Clause | Redundant, torque-controlled arm; impedance control |
| Dynamixel 2R | `dynamixel_2r` | 2 | MIT | A real two-link planar arm, comparable to the course model |

Menagerie notes that the Panda model was derived from Franka Emika's public
URDF description, with inertial parameters matched to the manufacturer's data,
and that its arm uses position-controlled actuators.

### 9.1 Real robots in the course notebooks: `rgkit.robots`

The notebooks use real robot models wherever they help, rather than toy arms.
To keep the repository small, and because each model carries its own licence,
the Menagerie models are **not** copied into this repository. Instead, the
helper module `rgkit.robots` downloads a model directory on demand, from a
fixed (*pinned*) commit of Menagerie, so that every student obtains exactly the
files against which the notebooks were tested.

```python
from rgkit.robots import load_robot, describe

model, data = load_robot("ur5e")      # downloads on first use, then reads the cache
print(describe("ur5e"))               # attribution line: source, maker and licence
```

| Key | Robot | Degrees of freedom | Licence |
|-----|-------|---------------------|---------|
| `"2r"` | Dynamixel 2R planar arm | 2 | MIT |
| `"ur5e"` | Universal Robots UR5e | 6 | BSD-3-Clause |
| `"panda"` | Franka Emika Panda | 7 (plus 2 finger joints, so `model.nq = 9`) | Apache-2.0 |
| `"iiwa14"` | KUKA LBR iiwa 14 | 7 | see the model's `LICENSE` |
| `"sawyer"` | Rethink Robotics Sawyer | 7 | see the model's `LICENSE` |
| `"op3"` | ROBOTIS OP3 humanoid | 20 | see the model's `LICENSE` |

`load_robot` loads the model's `scene.xml` (the robot together with a floor and
lights), resets the state to the model's first keyframe if it has one (for
example the UR5e "home" pose), calls `mj_forward`, and returns
`(model, data)`. Its behaviour is controlled by two environment variables:

| Variable | Effect |
|----------|--------|
| *(neither set)* | Models are downloaded from `raw.githubusercontent.com` into `~/.cache/robotics-guide` on first use; later runs work offline. |
| `RG_CACHE=/some/folder` | Use that folder as the download cache instead. |
| `MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie` | Use an existing local clone of Menagerie; no download takes place, so this works entirely offline. |

The first use of each robot therefore needs an internet connection; this works
on Colab as well. For offline work (an examination room, a laboratory without
internet), prepare a clone in advance:

```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
export MUJOCO_MENAGERIE_PATH="$PWD/mujoco_menagerie"     # Windows PowerShell: $env:MUJOCO_MENAGERIE_PATH = "$PWD\mujoco_menagerie"
```

A clone of the current Menagerie may differ in detail from the pinned commit
used by the notebooks; small numerical differences in printed outputs are then
possible.

**Licences.** The robot models are the work of Google DeepMind and the robot
manufacturers, and they are distributed under licences that differ from model
to model (Apache-2.0, BSD-3-Clause, MIT and others). `rgkit.robots` stores each
model's `LICENSE` file next to the downloaded model; read it before
redistributing a model or an image derived from it, and acknowledge the source
with the line returned by `describe(...)`.

The minimal model `simulations/mujoco/models/planar_2r.xml` remains useful
where every number must be known exactly (for instance, link lengths of
exactly 1.0 m and 0.8 m for hand calculation); the real Dynamixel 2R, UR5e and
Panda are used wherever realistic geometry and inertia matter.

### 9.2 Other ways to obtain Menagerie models

Besides `rgkit.robots`, the Menagerie README describes the following routes,
which are convenient for exploring models outside the notebooks:

```bash
# (1) Clone the repository and open a model in the viewer.
git clone https://github.com/google-deepmind/mujoco_menagerie.git
python -m mujoco.viewer --mjcf=mujoco_menagerie/franka_emika_panda/scene.xml

# (2) The Python package, which downloads and caches models on first use.
pip install mujoco-menagerie
```

```python
import mujoco_menagerie as mm
model = mm.load("franka_emika_panda")      # compiles scene.xml
```

The third-party package `robot_descriptions` also provides Menagerie models.
Before comparing a Menagerie model with a D-H table from a textbook, inspect
the body frames: manufacturers' frames generally do not follow the D-H
convention, so the joint angles agree but the intermediate frames differ.

### 9.3 MJX

For massively parallel simulation on GPUs and TPUs, the MuJoCo project also
provides **MJX** (`pip install mujoco-mjx`), a reimplementation of MuJoCo in
JAX that is API-compatible with MuJoCo but lacks some of its features. MJX is
used in reinforcement learning and differentiable simulation; it is not needed
for this course.

---

## 10. Other simulators and libraries

MuJoCo is the simulator used in the course material, but several other tools
are widely used in teaching and research. The optional ones can be installed
from [`requirements-sim.txt`](../requirements-sim.txt).

| Tool | What it offers | Installation | Remarks |
|------|----------------|--------------|---------|
| [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python) (P. Corke) | D-H and ETS robot models, `fkine`, `jacob0`, `jacobe`, `ikine_*`, dynamics, many predefined robots (e.g. `rtb.models.Panda()`), and the browser-based simulator *Swift* | `pip install roboticstoolbox-python` (Python ≥ 3.10); Swift via `pip install "roboticstoolbox-python[swift]"` | Closest in spirit to Craig's notation; excellent for cross-checking D-H kinematics. The README notes that Swift is not currently supported on Windows. |
| [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | Fast rigid-body dynamics: RNEA, ABA, their analytical derivatives, centroidal dynamics, frictional contact; reads URDF, SDF and MJCF | `conda install pinocchio -c conda-forge`, or `pip install pin` (pip wheels for Linux and macOS only) | The reference library for dynamics; useful for checking $M(q)$ and $V + G$. |
| [Modern Robotics library](https://github.com/NxRLab/ModernRobotics) (Lynch and Park) | Product-of-exponentials kinematics, Jacobians, dynamics and trajectory functions, organised by the chapters of *Modern Robotics* | `pip install modern_robotics` | Uses screw theory rather than D-H parameters: a useful comparison with Craig. |
| [PyBullet](https://github.com/bulletphysics/bullet3) | Physics simulation with URDF loading, inverse kinematics and dynamics, and a GUI | `pip install pybullet` | Widely used in older tutorials. Its latest release provides pre-built wheels only for Linux and Python ≤ 3.11; elsewhere a C++ compiler is required. |
| [Drake](https://github.com/RobotLocomotion/drake) | Model-based design and verification: multibody dynamics, optimisation, trajectory optimisation, systems framework | `pip install drake` (Linux and macOS; not native Windows) | Powerful but large; suited to advanced projects. |
| [ROS 2](https://github.com/ros2/ros2) | The standard middleware for robot software | See [docs.ros.org](https://docs.ros.org) | Beyond the scope of these notes; the natural next step for implementation on hardware. |
| [Gazebo](https://github.com/gazebosim/gz-sim) | Open-source robot simulator with physics, rendering and sensor simulation, integrated with ROS 2 | See the Gazebo installation tutorials | Used with ROS 2 for system-level simulation. |

---

## 11. References

* J. J. Craig, *Introduction to Robotics: Mechanics and Control*, 4th ed.,
  Pearson, 2018.
* E. Todorov, T. Erez and Y. Tassa, "MuJoCo: A physics engine for model-based
  control," *Proc. IEEE/RSJ International Conference on Intelligent Robots and
  Systems (IROS)*, 2012, pp. 5026–5033.
* MuJoCo source code and README: <https://github.com/google-deepmind/mujoco>.
* MuJoCo documentation: <https://mujoco.readthedocs.io> (Python bindings,
  MJCF reference, API reference).
* MuJoCo Python bindings documentation (source):
  <https://github.com/google-deepmind/mujoco/blob/main/doc/python.rst>.
* Official introductory Colab tutorial:
  <https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb>;
  model editing with `MjSpec`:
  <https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb>;
  LQR control of a humanoid:
  <https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/LQR.ipynb>.
* MJX: <https://github.com/google-deepmind/mujoco/tree/main/mjx>.
* MuJoCo Menagerie: <https://github.com/google-deepmind/mujoco_menagerie>.
* `dm_control` (documents the `MUJOCO_GL` back ends `glfw`, `egl`, `osmesa`):
  <https://github.com/google-deepmind/dm_control>.
* K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and
  Control*, Cambridge University Press, 2017.
* P. Corke, *Robotics, Vision and Control: Fundamental Algorithms in Python*,
  3rd ed., Springer, 2023.
