"""Single-joint experiments in MuJoCo for the control-systems chapter of *Robot Motion Control*.

A robot joint is, to a first approximation, a single-degree-of-freedom system: an actuator drives an inertia
against gravity and friction.  This module provides the glue between that textbook model and MuJoCo:

* reading a joint's *effective inertia*, position-servo gains and linearised gravity stiffness from a compiled model,
* *locking* every other joint of a multi-joint robot so that one joint can be identified in isolation,
* a model of a **DC-motor-driven joint** (:func:`dc_joint_xml`) built on MuJoCo's ``dcmotor`` actuator, whose
  transfer function is known in closed form,
* a sampled-data simulation loop that calls a Python controller at a chosen sample time, and
* the linearisation of a compiled model about an operating point (:func:`fd_linearize`).

MuJoCo is imported lazily so that the rest of ``rgkit`` works without it.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

__all__ = ["mass_matrix", "locked", "servo_gains", "effective_inertia", "gravity_stiffness", "servo_step",
           "DC_JOINT", "dc_joint_params", "dc_joint_xml", "dc_joint_tf", "simulate", "fd_linearize"]


def _mj():
    from . import mujoco_utils as mju  # configures the OpenGL backend before MuJoCo is imported
    return mju.mujoco


# ------------------------------------------------------------------------------------------------------
# Reading a joint from a compiled model
# ------------------------------------------------------------------------------------------------------
def mass_matrix(model, data) -> np.ndarray:
    """Dense joint-space inertia matrix ``M(q)`` (works across MuJoCo releases; call after ``mj_forward``)."""
    mj = _mj()
    M = np.zeros((model.nv, model.nv))
    try:
        mj.mj_fullM(model, data, M)
    except TypeError:
        mj.mj_fullM(model, M, data.qM)
    return M


@contextmanager
def locked(model, data, dofs, armature: float = 1e6):
    """Temporarily *lock* the degrees of freedom ``dofs`` at their current positions.

    Each locked dof receives an enormous rotor inertia (``armature``) and zero velocity; the position
    actuator acting on it (if any) is commanded to the current angle.  The effective inertia seen by the remaining
    dofs is then the corresponding block of ``M`` (the Schur complement tends to it as ``armature`` grows).
    The original model is restored on exit.
    """
    dofs = list(np.atleast_1d(dofs))
    saved = model.dof_armature[dofs].copy()
    model.dof_armature[dofs] = armature
    data.qvel[dofs] = 0.0
    try:
        yield
    finally:
        model.dof_armature[dofs] = saved


def servo_gains(model, actuator) -> dict:
    """Read the gains of a *position* actuator: force ``= kp (ctrl - q) - kv qdot``.

    MuJoCo stores ``gainprm[0] = kp`` and ``biasprm = [0, -kp, -kv]`` for ``<position>`` actuators (and for the
    equivalent ``<general>`` ones used by the UR5e model).  Returns ``kp``, ``kv``, the driven ``dof`` and the
    actuator's ``gear``.
    """
    mj = _mj()
    i = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator) if isinstance(actuator, str) else int(actuator)
    gain, bias = model.actuator_gainprm[i], model.actuator_biasprm[i]
    kp, kv = float(gain[0]), float(-bias[2])
    if not np.isclose(-bias[1], kp):
        raise ValueError("actuator is not a position servo (biasprm[1] != -gainprm[0])")
    jid = int(model.actuator_trnid[i, 0])
    return dict(index=i, kp=kp, kv=kv, dof=int(model.jnt_dofadr[jid]), gear=float(model.actuator_gear[i, 0]),
                forcerange=model.actuator_forcerange[i].copy(), ctrlrange=model.actuator_ctrlrange[i].copy())


def effective_inertia(model, data, dof: int, others: str = "locked") -> float:
    """Inertia felt by ``dof``: ``M[dof, dof]`` (other dofs locked) or ``1/(M^-1)[dof, dof]`` (others free)."""
    M = mass_matrix(model, data)
    return float(M[dof, dof]) if others == "locked" else float(1.0 / np.linalg.inv(M)[dof, dof])


def gravity_stiffness(model, data, dof: int, h: float = 1e-5) -> float:
    """``d G / d q`` of the gravity torque on ``dof`` (others fixed), by central differences of ``qfrc_bias``.

    Positive for a pendulum hanging below its pivot (a restoring torque), negative above it.  Evaluated at the
    current state with zero velocity.
    """
    mj = _mj()
    q0, v0 = data.qpos.copy(), data.qvel.copy()
    adr = int(model.dof_jntid[dof])
    qadr = int(model.jnt_qposadr[adr])
    out = []
    data.qvel[:] = 0.0
    for s in (+1, -1):
        data.qpos[:] = q0
        data.qpos[qadr] += s * h
        mj.mj_forward(model, data)
        out.append(data.qfrc_bias[dof])
    data.qpos[:], data.qvel[:] = q0, v0
    mj.mj_forward(model, data)
    return float((out[0] - out[1]) / (2 * h))


def servo_step(model, data, actuator, target: float, T: float, q0: float | None = None, dt: float | None = None,
               ref=None):
    """Step (or arbitrary) reference response of one position-servo joint with every other dof locked.

    The configuration in ``data.qpos`` when the function is called is the *operating point*: all other dofs are
    locked there (see :func:`locked`) and their position actuators commanded to hold it.  ``target`` is the
    commanded angle of the chosen actuator (``ref(t)`` may supply a time-varying reference instead) and ``q0``
    optionally sets the starting angle of the joint (default: its current angle, which is also the initial
    command).  Returns ``(t, q, qd)``; the model state is restored afterwards.
    """
    mj = _mj()
    g = servo_gains(model, actuator)
    dof = g["dof"]
    others = [d for d in range(model.nv) if d != dof]
    saved = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time, model.opt.timestep)
    if dt is not None:
        model.opt.timestep = dt
    qhold = data.qpos.copy()
    with locked(model, data, others):
        data.qvel[:] = 0.0
        data.time = 0.0
        if q0 is not None:
            data.qpos[dof] = q0
        for i in range(model.nu):                                  # hold every other servo where it is
            jid = int(model.actuator_trnid[i, 0])
            data.ctrl[i] = qhold[model.jnt_qposadr[jid]]
        start = data.qpos[dof]
        mj.mj_forward(model, data)
        n = int(round(T / model.opt.timestep))
        t, q, qd = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
        t[0], q[0], qd[0] = 0.0, data.qpos[dof], data.qvel[dof]
        for k in range(n):
            data.ctrl[g["index"]] = target if ref is None else ref(data.time)
            mj.mj_step(model, data)
            t[k + 1], q[k + 1], qd[k + 1] = data.time, data.qpos[dof], data.qvel[dof]
    data.qpos[:], data.qvel[:], data.ctrl[:], data.time, model.opt.timestep = saved
    mj.mj_forward(model, data)
    return t, q, qd


# ------------------------------------------------------------------------------------------------------
# A DC-motor-driven joint (MuJoCo's dcmotor actuator)
# ------------------------------------------------------------------------------------------------------
#: Illustrative parameters of one joint of a small collaborative arm.  These are teaching values chosen to be of
#: realistic magnitude (a ~100 W brushed motor behind a 50:1 gearbox, driving a 0.4 m, 0.8 kg link); they are not
#: taken from a datasheet.
DC_JOINT = dict(
    N=50.0,        # gear ratio
    Kt=0.030,      # torque constant [N m/A] (equal to the back-EMF constant [V s/rad] in SI units)
    Ke=0.030,
    R=1.2,         # armature resistance [ohm]
    L=1.2e-3,      # armature inductance [H]
    Jm=3.0e-5,     # rotor inertia [kg m^2] (motor side)
    b=0.02,        # viscous friction at the joint [N m s/rad]
    m=0.8,         # link mass [kg]
    ell=0.4,       # link length [m] (uniform rod, pivot at one end)
    vmax=24.0,     # supply voltage limit [V]
)


def dc_joint_params(**overrides) -> dict:
    """The parameter dictionary of :data:`DC_JOINT` plus derived quantities, with optional overrides.

    Derived: ``J_rotor = N^2 Jm`` (reflected rotor inertia), ``J_link = m ell^2/3``, ``J = J_rotor + J_link``,
    ``mgd = m g ell/2`` (peak gravity torque) and ``tau_e = L/R``.
    """
    p = dict(DC_JOINT)
    p.update(overrides)
    p["J_rotor"] = p["N"] ** 2 * p["Jm"]
    p["J_link"] = p["m"] * p["ell"] ** 2 / 3
    p["J"] = p["J_rotor"] + p["J_link"]
    p["mgd"] = p["m"] * 9.81 * p["ell"] / 2
    p["tau_e"] = p["L"] / p["R"]
    return p


def dc_joint_xml(p: dict | None = None, gravity: bool = True, timestep: float = 2e-4, limited: bool = False,
                 with_scene: bool = True) -> str:
    """MJCF of the DC-motor joint: a uniform rod hanging from a hinge about the world ``y`` axis.

    The joint angle ``theta`` is zero when the rod hangs straight down (stable equilibrium); ``theta = pi`` is the
    inverted position.  The actuator is a ``dcmotor`` driven by the **terminal voltage** (``ctrl`` in volts, clipped to
    ``+/- vmax`` by MuJoCo); its current is the actuator state ``data.act[0]``.  With ``gear = N`` it produces the joint
    torque ``N Kt i`` and the back-EMF ``N Ke qdot``.
    """
    from . import mujoco_utils as mju
    p = dc_joint_params() if p is None else p
    Icom = p["m"] * p["ell"] ** 2 / 12
    rng = 'range="-3.1 3.1"' if limited else ""
    scene = mju.scene_assets() if with_scene else ""
    floor = mju.floor(z=-0.55, size=1.2) if with_scene else ""
    return f"""
<mujoco model="dc_motor_joint">
  <option timestep="{timestep}" integrator="implicitfast" gravity="0 0 {-9.81 if gravity else 0.0}"/>
  {scene}
  <worldbody>
    {floor}
    <geom type="cylinder" size="0.045 0.03" pos="0 -0.045 0" euler="90 0 0" rgba="0.25 0.28 0.33 1" contype="0" conaffinity="0"/>
    <body name="link">
      <joint name="theta" type="hinge" axis="0 1 0" armature="{p['J_rotor']}" damping="{p['b']}" {rng}/>
      <inertial pos="0 0 {-p['ell'] / 2}" mass="{p['m']}" diaginertia="{Icom} {Icom} 1e-4"/>
      <geom type="capsule" size="0.022" fromto="0 0 0 0 0 {-p['ell']}" rgba="0.85 0.45 0.12 1" contype="0" conaffinity="0" mass="0"/>
      <geom type="sphere" size="0.035" pos="0 0 {-p['ell']}" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0" mass="0"/>
      <site name="tip" pos="0 0 {-p['ell']}" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <dcmotor name="motor" joint="theta" motorconst="{p['Kt']}" resistance="{p['R']}" inductance="{p['L']}"
             gear="{p['N']}" ctrlrange="{-p['vmax']} {p['vmax']}"/>
  </actuator>
</mujoco>"""


def dc_joint_tf(p: dict | None = None, output: str = "angle"):
    """Closed-form transfer function (voltage to joint angle or speed) of the joint built by :func:`dc_joint_xml`."""
    from .linear_control import dc_motor_tf
    p = dc_joint_params() if p is None else p
    return dc_motor_tf(p["Kt"], p["Ke"], p["R"], p["L"], p["J"], p["b"], p["N"], output=output)


# ------------------------------------------------------------------------------------------------------
# Sampled-data simulation
# ------------------------------------------------------------------------------------------------------
def simulate(model, data, controller, T: float, Ts: float, q0=None, qd0=None, measure=None, disturbance=None,
             act0=None, ctrl_index: int = 0, dof: int = 0):
    """Run ``controller`` on a single joint of ``model`` for ``T`` seconds with sample time ``Ts``.

    Every ``Ts`` seconds the controller is called as ``u = controller(t, y)`` where ``y`` is the measured joint
    angle (``measure(q)`` if given, for example a quantised encoder); the command is held (zero-order hold)
    until the next sample and written to ``data.ctrl[ctrl_index]``.  ``disturbance(t)`` adds a torque (N m) to
    ``qfrc_applied``.  Returns a dict of arrays sampled at the control instants: ``t, q, qd, u, i`` (``i`` is the
    actuator state, i.e. the motor current, when the model has one).  The simulator's own timestep must divide ``Ts``.
    """
    mj = _mj()
    mj.mj_resetData(model, data)
    if q0 is not None:
        data.qpos[dof] = q0
    if qd0 is not None:
        data.qvel[dof] = qd0
    if act0 is not None and model.na:
        data.act[:] = act0
    mj.mj_forward(model, data)
    sub = int(round(Ts / model.opt.timestep))
    if abs(sub * model.opt.timestep - Ts) > 1e-9:
        raise ValueError("Ts must be a multiple of the model timestep")
    n = int(round(T / Ts))
    out = {k: np.empty(n + 1) for k in ("t", "q", "qd", "u", "i")}
    for k in range(n + 1):
        y = data.qpos[dof] if measure is None else measure(data.qpos[dof])
        u = float(controller(data.time, y))
        data.ctrl[ctrl_index] = u
        out["t"][k], out["q"][k], out["qd"][k], out["u"][k] = data.time, data.qpos[dof], data.qvel[dof], u
        out["i"][k] = data.act[0] if model.na else np.nan
        if k == n:
            break
        for _ in range(sub):
            if disturbance is not None:
                data.qfrc_applied[dof] = disturbance(data.time)
            mj.mj_step(model, data)
    data.qfrc_applied[:] = 0.0
    return out


# ------------------------------------------------------------------------------------------------------
# Linearisation of a compiled model
# ------------------------------------------------------------------------------------------------------
def fd_linearize(model, data, eps: float = 1e-6):
    """Continuous-time linearisation ``(A, B)`` of a MuJoCo model about the current state and control.

    MuJoCo differentiates one step of the simulator by finite differences (``mjd_transitionFD``) and returns the
    discrete map ``x+ = Ad x + Bd u`` on the state ``x = [dq, dv, act]``.  With ``h`` the timestep, the
    continuous-time pair follows from the matrix logarithm of the augmented matrix ``[[Ad, Bd], [0, I]]``.
    Returns ``(A, B, Ad, Bd)``.
    """
    from scipy.linalg import logm
    mj = _mj()
    n = 2 * model.nv + model.na
    Ad, Bd = np.zeros((n, n)), np.zeros((n, model.nu))
    mj.mj_forward(model, data)
    mj.mjd_transitionFD(model, data, eps, True, Ad, Bd, None, None)
    h = model.opt.timestep
    aug = np.zeros((n + model.nu, n + model.nu))
    aug[:n, :n], aug[:n, n:], aug[n:, n:] = Ad, Bd, np.eye(model.nu)
    L = np.real(logm(aug)) / h
    return L[:n, :n], L[:n, n:], Ad, Bd
