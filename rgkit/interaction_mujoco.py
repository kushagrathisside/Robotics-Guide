"""A Franka Panda pressing a stylus against a table or a spring-loaded plate in MuJoCo.

:class:`ContactRig` builds the scene (with ``MjSpec``, so that the manufacturer's model from MuJoCo Menagerie is
used unchanged and our additions are visible in a few lines), exposes the kinematic and dynamic quantities that
force and impedance controllers need, and reads the contact force in three independent ways:

``contact_force()``   sum of ``mj_contactForce`` over the contacts of the tool, rotated into the **world** frame;
                      the force **exerted by the environment on the tool**.
``ft_force()``        a wrist ``force`` sensor (MuJoCo reports the force the parent body transmits to the hand+tool
                      subtree, expressed in the **site frame**); after rotating to the world frame and removing
                      the weight of the load we obtain the same environment-on-tool force, as a real
                      force/torque sensor would.
``cfrc_ext_force()``  ``data.cfrc_ext`` of the hand body (needs ``mj_rnePostConstraint``): world-frame, the sum of all
                      external forces on that body (contacts + ``xfrc_applied``), here again the contact force.

Sign convention (Chapter 5): ``F_ext`` = force of the environment on the tool; the *contact force* ``f`` that the
tool exerts on the environment is ``-F_ext``.

Torque control is done through ``data.qfrc_applied`` on the seven arm joints; the manufacturer's position actuators
of the arm are disabled by :meth:`ContactRig.torque_mode` and re-enabled by :meth:`ContactRig.position_mode`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import mujoco_utils as mju
from . import robots

mujoco = mju.mujoco

__all__ = ["ContactRig", "build_contact_scene", "ARM_TORQUE_LIMITS", "run_episode", "orientation_error",
           "critical_damping", "CartesianImpedance", "HybridForceControl", "AdmittanceControl", "JointServoTarget"]

#: Torque limits of the seven Panda joints (N m) quoted by the Menagerie model (`forcerange`).
ARM_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
G = 9.81


def build_contact_scene(top=0.40, k_env=None, b_env=20.0, plate_center=(0.50, 0.0), plate_half=(0.24, 0.30),
                        mu=0.3, timestep=1e-3, tool_len=0.08, tool_radius=0.01, solref=None, plate_mass=0.5):
    """Compile the scene: Panda + stylus on the hand + table + (optionally spring-loaded) plate.

    * The stylus is a capsule on the hand body, tip at 0.14 m from the hand frame along its z axis; the site ``tcp``
      sits exactly at the stylus tip and ``wrist_ft`` at the hand frame (flange) where the force sensor is mounted.
    * ``k_env=None``: the plate is welded to the world (a rigid table whose stiffness is whatever MuJoCo's soft contact
      gives). Otherwise the plate slides vertically on a joint with spring ``k_env`` (N/m) and damper ``b_env`` (N s/m),
      so the environment stiffness is a parameter we control. The plate's weight is compensated (``gravcomp``).
    * ``solref`` optionally overrides the contact softness of the plate geom (see Section 5.5).
    """
    d = robots.fetch("panda")
    spec = mujoco.MjSpec.from_file(str(d / "panda.xml"))
    spec.option.timestep = timestep
    gt = mujoco.mjtGeom
    hand = spec.body("hand")
    hand.add_geom(name="tool", type=gt.mjGEOM_CAPSULE, fromto=[0, 0, 0.06, 0, 0, 0.13], size=[tool_radius, 0, 0],
                  rgba=[0.15, 0.15, 0.17, 1], friction=[mu, 0.005, 0.0001], density=500)
    hand.add_site(name="tcp", pos=[0, 0, 0.13 + tool_radius], size=[0.004, 0, 0], rgba=[1, 0.5, 0, 1])
    hand.add_site(name="wrist_ft", pos=[0, 0, 0], size=[0.008, 0, 0], rgba=[0.2, 0.6, 1, 0.5])
    spec.add_sensor(name="ft_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="wrist_ft")
    spec.add_sensor(name="ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="wrist_ft")
    wb = spec.worldbody
    cx, cy = plate_center
    wb.add_geom(name="table", type=gt.mjGEOM_BOX, pos=[cx + 0.15, cy, top - 0.10 - 0.02], size=[0.40, 0.45, 0.02],
                rgba=[0.62, 0.50, 0.40, 1])           # starts at x = 0.25 m: clear of the robot's shoulder
    plate = wb.add_body(name="plate", pos=[cx, cy, top - 0.01])
    if k_env is not None:
        plate.add_joint(name="plate_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], stiffness=k_env,
                        damping=b_env, range=[-0.09, 0.0], limited=mujoco.mjtLimited.mjLIMITED_TRUE)
        plate.gravcomp = 1.0
        rgba = [0.80, 0.68, 0.45, 1]
    else:
        rgba = [0.62, 0.50, 0.40, 1]
    g = plate.add_geom(name="plate", type=gt.mjGEOM_BOX, size=[plate_half[0], plate_half[1], 0.01], mass=plate_mass,
                       rgba=rgba, friction=[mu, 0.005, 0.0001])
    if solref is not None:
        g.solref = list(solref)
    return spec.compile()


@dataclass
class ContactRig:
    """Model, data and bookkeeping for the pressing experiments (see the module docstring)."""

    k_env: float | None = None
    b_env: float = 20.0
    top: float = 0.40
    mu: float = 0.3
    timestep: float = 1e-3
    solref: tuple | None = None
    model: object = field(init=False, repr=False)
    data: object = field(init=False, repr=False)

    def __post_init__(self):
        self.model = build_contact_scene(self.top, self.k_env, self.b_env, mu=self.mu, timestep=self.timestep,
                                         solref=self.solref)
        m = self.model
        self.data = mujoco.MjData(m)
        self.nq = 7
        self.arm = slice(0, 7)
        self.tcp = m.site("tcp").id
        self.wrist = m.site("wrist_ft").id
        self.hand = m.body("hand").id
        self.tool_geom = m.geom("tool").id
        self.plate_geom = m.geom("plate").id
        self.plate_jnt = m.joint("plate_z").id if self.k_env is not None else None
        self.m_load = float(m.body_subtreemass[self.hand])
        self._gain = (m.actuator_gainprm[:7].copy(), m.actuator_biasprm[:7].copy())
        self.q_home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))
        self.reset()
        self.R_down = self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    # ------------------------------------------------------------------ set-up -------------------------------
    def reset(self, q=None, grip=0.2):
        """Return to the home posture (or ``q``), fingers closed on the stylus, plate undeflected, zero velocity."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = self.q_home if q is None else q
        d.qpos[7:9] = 0.04 * grip
        d.ctrl[:7] = d.qpos[:7]
        d.ctrl[7] = 255 * grip
        mujoco.mj_forward(m, d)
        return self

    def torque_mode(self):
        """Switch the seven arm actuators off; joint torques are then given through ``qfrc_applied``."""
        self.model.actuator_gainprm[:7] = 0.0
        self.model.actuator_biasprm[:7] = 0.0
        return self

    def position_mode(self):
        """Restore the manufacturer's joint position servos (``ctrl`` = target joint angles)."""
        self.model.actuator_gainprm[:7], self.model.actuator_biasprm[:7] = self._gain[0].copy(), self._gain[1].copy()
        return self

    @property
    def joint_stiffness(self):
        """Proportional gains ``K_q`` of the position servos (N m/rad): ``gainprm[0]`` of each arm actuator."""
        return self._gain[0][:, 0].copy()

    @property
    def joint_damping_servo(self):
        """Derivative gains of the position servos (N m s/rad): ``-biasprm[2]``."""
        return -self._gain[1][:, 2].copy()

    # ------------------------------------------------------------------ state, kinematics ---------------------
    @property
    def q(self):
        return self.data.qpos[:7].copy()

    @property
    def qd(self):
        return self.data.qvel[:7].copy()

    @property
    def p(self):
        """Tool-tip position in the world frame (m)."""
        return self.data.site_xpos[self.tcp].copy()

    @property
    def R(self):
        """Orientation of the tool frame, ``^0_T R`` (columns = tool axes in the world frame)."""
        return self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    def jacobian(self):
        """6x7 geometric Jacobian of the tool tip in the world frame: rows 0-2 linear, rows 3-5 angular velocity."""
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.tcp)
        return np.vstack([self._jp[:, :7], self._jr[:, :7]])

    def mass_matrix(self):
        """7x7 joint-space inertia matrix of the arm (includes the rotor ``armature`` terms)."""
        try:
            mujoco.mj_fullM(self.model, self.data, self._M)
        except TypeError:                                    # older releases
            mujoco.mj_fullM(self.model, self._M, self.data.qM)
        return self._M[:7, :7].copy()

    def bias(self):
        """``c(q, qd)`` = Coriolis, centrifugal and gravity torques of the arm joints (``qfrc_bias``)."""
        return self.data.qfrc_bias[:7].copy()

    def gravity(self):
        """Gravity torques ``g(q)`` of the arm: ``qfrc_bias`` evaluated at zero velocity."""
        d2 = mujoco.MjData(self.model)
        d2.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, d2)
        return d2.qfrc_bias[:7].copy()

    def task_inertia(self, J=None, M=None, dims=3):
        """Operational-space inertia ``Lambda = (J M^-1 J^T)^-1`` for the first ``dims`` rows of the Jacobian."""
        J = (self.jacobian() if J is None else J)[:dims]
        M = self.mass_matrix() if M is None else M
        return np.linalg.inv(J @ np.linalg.solve(M, J.T))

    def ik(self, pos, R=None, q0=None, iters=200, tol=1e-9, posture=0.1):
        """Damped least-squares inverse kinematics for the tool tip (position and orientation), with a weak
        attraction to the home posture in the null space. Returns joint angles (7,)."""
        R_des = self.R_down if R is None else R
        d2 = mujoco.MjData(self.model)
        d2.qpos[:] = self.data.qpos
        if q0 is not None:
            d2.qpos[:7] = q0
        for _ in range(iters):
            mujoco.mj_forward(self.model, d2)
            mujoco.mj_jacSite(self.model, d2, self._jp, self._jr, self.tcp)
            J = np.vstack([self._jp[:, :7], self._jr[:, :7]])
            Rc = d2.site_xmat[self.tcp].reshape(3, 3)
            e_r = 0.5 * sum(np.cross(Rc[:, i], R_des[:, i]) for i in range(3))
            e = np.r_[np.asarray(pos) - d2.site_xpos[self.tcp], e_r]
            if np.linalg.norm(e) < tol:
                break
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
            dq = Jp @ e + (np.eye(7) - Jp @ J) @ (posture * (self.q_home - d2.qpos[:7]))
            d2.qpos[:7] += dq
        return d2.qpos[:7].copy()

    # ------------------------------------------------------------------ force measurement ---------------------
    def _tool_contacts(self):
        d, g = self.data, self.tool_geom
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == g or c.geom2 == g:
                yield i, c

    def contact_force(self):
        """Force of the environment on the tool (3,), world frame, from ``mj_contactForce`` summed over the tool contacts.

        ``mj_contactForce`` returns the force in the *contact frame* (row 0 of ``contact.frame`` is the normal, pointing
        from ``geom1`` to ``geom2``), acting on ``geom2``; on ``geom1`` it is the opposite.
        """
        f6 = np.zeros(6)
        total = np.zeros(3)
        for i, c in self._tool_contacts():
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]            # contact frame -> world
            total += fw if c.geom2 == self.tool_geom else -fw
        return total

    def normal_force(self):
        """Normal component (N) of :meth:`contact_force` along the plate's upward normal (+z)."""
        return self.contact_force()[2]

    def ft_force(self, compensate=True):
        """Environment-on-tool force estimated from the wrist force sensor, in the **world** frame.

        MuJoCo's ``force`` sensor reports (in the site frame) the force transmitted from the parent to the child
        subtree, so in free static equilibrium it reads the weight ``+m g`` upwards. With ``compensate=True`` that weight
        is removed (``m_load`` is the mass of hand+fingers+stylus, known from the model) and the sign is flipped:
        ``F_ext = m_load g e_z - R_site s``  (the inertial term ``m_load a`` is neglected).
        """
        s = self.data.sensor("ft_force").data.copy()
        Rw = self.data.site_xmat[self.wrist].reshape(3, 3)
        fw = Rw @ s
        return (np.array([0.0, 0.0, self.m_load * G]) - fw) if compensate else fw

    def cfrc_ext_force(self):
        """Force on the hand body from ``data.cfrc_ext`` (world frame; rotational part first, translational last)."""
        mujoco.mj_rnePostConstraint(self.model, self.data)
        return self.data.cfrc_ext[self.hand, 3:6].copy()

    def plate_deflection(self):
        """Downward deflection (m, >= 0) of the spring-loaded plate; 0 for a rigid table."""
        if self.plate_jnt is None:
            return 0.0
        return float(-self.data.qpos[self.model.jnt_qposadr[self.plate_jnt]])

    def plate_spring_force(self):
        """Force (N) in the plate's spring and damper: ``k x + b x'`` (equals the contact force in statics)."""
        if self.plate_jnt is None:
            return float("nan")
        a = self.model.jnt_qposadr[self.plate_jnt]
        v = self.model.jnt_dofadr[self.plate_jnt]
        return float(-self.k_env * self.data.qpos[a] - self.b_env * self.data.qvel[v])

    def surface_height(self):
        """Current height of the plate's upper surface (m)."""
        return self.top - self.plate_deflection()

    # ------------------------------------------------------------------ stepping -------------------------------
    def clip_torque(self, tau):
        return np.clip(tau, -ARM_TORQUE_LIMITS, ARM_TORQUE_LIMITS)

    def render(self, camera=None, width=480, height=360):
        camera = camera or dict(azimuth=140, elevation=-18, distance=1.35, lookat=(0.48, 0.0, 0.42))
        return mju.render(self.model, self.data, camera=camera, width=width, height=height)


def run_episode(rig: ContactRig, controller: Callable, T: float, snapshots=(), camera=None, record_every=1,
                width=480, height=360):
    """Run ``controller(rig, t) -> command`` for ``T`` seconds, one command per model step (1 kHz by default).

    The controller returns either ``("tau", tau7)`` (joint torques written to ``qfrc_applied``, clipped to the
    Panda limits) or ``("q", q7)`` (joint position targets written to ``ctrl`` for the built-in servos, with the
    bias torque ``qfrc_bias`` added as gravity feed-forward, as the real Panda controller does) and may
    return a third element, a dict of values to log. Quantities are read between ``mj_step1`` and ``mj_step2``
    so that kinematics, Jacobian, mass matrix and bias are those of the current state; forces are read after the
    previous step, i.e. with one sample of delay, as on a real robot.

    Returns a dict of arrays: time ``t``, tool position ``p``, ``F_contact`` (mj_contactForce, world frame, environment
    on tool), ``F_ft`` (wrist sensor estimate), ``defl`` (plate deflection), ``tau`` and any logged extras; plus
    ``frames`` (list of rendered images at the requested ``snapshots`` times).
    """
    m, d = rig.model, rig.data
    n = int(round(T / m.opt.timestep))
    log = {k: [] for k in ("t", "p", "F_contact", "F_ft", "defl", "tau")}
    extras: dict = {}
    frames, snap = [], sorted(snapshots)
    for k in range(n):
        mujoco.mj_step1(m, d)
        t = d.time
        out = controller(rig, t)
        kind, cmd = out[0], np.asarray(out[1], float)
        d.qfrc_applied[:] = 0.0
        if kind == "tau":
            tau = rig.clip_torque(cmd)
            d.qfrc_applied[:7] = tau
        elif kind == "q":
            d.ctrl[:7] = cmd
            d.qfrc_applied[:7] = d.qfrc_bias[:7]                 # gravity (and Coriolis) feed-forward of the servo
            tau = d.qfrc_applied[:7].copy()
        else:
            raise ValueError(kind)
        mujoco.mj_step2(m, d)
        if k % record_every == 0:
            log["t"].append(t)
            log["p"].append(rig.p)
            log["F_contact"].append(rig.contact_force())
            log["F_ft"].append(rig.ft_force())
            log["defl"].append(rig.plate_deflection())
            log["tau"].append(tau)
            if len(out) > 2 and out[2]:
                for key, val in out[2].items():
                    extras.setdefault(key, []).append(np.asarray(val, float))
        while snap and d.time >= snap[0]:
            frames.append(rig.render(camera, width, height))
            snap.pop(0)
    res = {k: np.array(v) for k, v in log.items()}
    res.update({k: np.array(v) for k, v in extras.items()})
    res["frames"] = frames
    return res


# ============================================================================================================
# Controllers (each is a callable ``controller(rig, t)`` for :func:`run_episode`)
# ============================================================================================================
def orientation_error(R, R_d):
    """Small-rotation vector (world frame) that takes the current orientation ``R`` towards ``R_d``:
    ``e_r = 1/2 sum_i  R[:, i] x R_d[:, i]``. Its norm is ``sin(angle)`` for a rotation of that angle about an axis."""
    return 0.5 * sum(np.cross(R[:, i], R_d[:, i]) for i in range(3))


def critical_damping(K, Lam, zeta=1.0):
    """Damping matrix ``B`` that gives every mode of ``Lam e'' + B e' + K e = 0`` the damping ratio ``zeta``.

    Uses the simultaneous diagonalisation ``Lam^-1/2 K Lam^-1/2 = Q diag(w^2) Q^T`` and returns
    ``B = 2 zeta Lam^1/2 Q diag(w) Q^T Lam^1/2`` (symmetric, positive definite).
    """
    wl, Vl = np.linalg.eigh(Lam)
    Lh, Lmh = Vl @ np.diag(np.sqrt(wl)) @ Vl.T, Vl @ np.diag(1 / np.sqrt(wl)) @ Vl.T
    w2, Q = np.linalg.eigh(Lmh @ K @ Lmh)
    return 2 * zeta * Lh @ Q @ np.diag(np.sqrt(w2)) @ Q.T @ Lh


def _nullspace_posture(rig, J, M, Lam, k=5.0, d=2.0):
    """Dynamically consistent null-space torque pulling the arm towards its home posture."""
    Minv = np.linalg.inv(M)
    Nt = np.eye(7) - J.T @ Lam @ J @ Minv
    return Nt @ (k * (rig.q_home - rig.q) - d * rig.qd)


@dataclass
class CartesianImpedance:
    """Cartesian stiffness/damping control with gravity and Coriolis compensation (task-space impedance, ``M_d = Lambda``).

        tau = J^T [ K (x_d - x) + B (v_d - v) ] + c(q, qd) + N^T tau_posture

    ``K`` is a 6x6 (or 3x3 with orientation gains ``K_rot``) stiffness, ``B`` is computed by :func:`critical_damping`
    from the local operational-space inertia with damping ratio ``zeta`` unless a matrix ``B`` is given.
    ``target(t) -> (p_d, v_d)`` gives the desired tool position and velocity; the orientation is held at ``R_d``.
    """
    K_t: np.ndarray                    # 3x3 translational stiffness (N/m)
    target: Callable
    K_r: float = 60.0                  # rotational stiffness (N m/rad), isotropic
    zeta: float = 1.0
    B_t: np.ndarray | None = None
    R_d: np.ndarray | None = None
    posture: bool = True
    f_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))   # constant force added to the commanded wrench

    def __call__(self, rig, t):
        J, M = rig.jacobian(), rig.mass_matrix()
        Lam = np.linalg.inv(J @ np.linalg.solve(M, J.T))
        K = np.zeros((6, 6))
        K[:3, :3], K[3:, 3:] = self.K_t, self.K_r * np.eye(3)
        B = critical_damping(K, Lam, self.zeta)
        if self.B_t is not None:
            B[:3, :3] = self.B_t
        p_d, v_d = self.target(t)
        R_d = rig.R_down if self.R_d is None else self.R_d
        e = np.r_[p_d - rig.p, orientation_error(rig.R, R_d)]
        v = J @ rig.qd
        w = K @ e + B @ (np.r_[v_d, np.zeros(3)] - v)
        w[:3] += self.f_offset
        tau = J.T @ w + rig.bias()
        if self.posture:
            tau = tau + _nullspace_posture(rig, J, M, Lam)
        return "tau", tau, {"p_d": p_d, "e": e[:3]}


@dataclass
class HybridForceControl:
    """Raibert-Craig hybrid position/force control in operational space (world frame, contact normal ``+z``).

        F = S Lam S a_pos + (I - S) f_cmd,     a_pos = a_d + K_v (v_d - v) + K_p (p_d - p),
        f_cmd = f_d + K_f (f_d - f) + K_i int (f_d - f) dt - B_f v_n,

    where ``f = -F_ext`` is the force exerted by the tool on the environment, measured with the wrist sensor
    (``source="ft"``) or, for comparison, with ``mj_contactForce`` (``source="contact"``). The orientation is held with a
    stiff PD law; a null-space torque keeps the arm near its home posture. ``tau = J^T w + c(q, qd) + N^T tau_posture``.

    ``S_diag`` holds the three diagonal entries of the translational selection matrix (1 = position controlled);
    the default ``(1, 1, 0)`` position-controls x and y and force-controls z (the normal of the plate).
    ``approach_v > 0`` adds a guarded move: the z axis first descends at ``approach_v`` m/s (velocity control) until
    the measured contact force exceeds ``f_thresh``; the force loop then takes over.
    """
    target: Callable                           # t -> (p_d, v_d, a_d), only x, y matter on the position axes
    f_d: float                                 # desired pressing force (N), positive = pressing down
    K_f: float = 0.0
    K_i: float = 0.0
    B_f: float = 0.0                           # damping (N s/m) on the force-controlled axis (inner velocity loop)
    K_p: float = 400.0
    K_v: float = 40.0
    S_diag: tuple = (1.0, 1.0, 0.0)
    K_r: float = 80.0
    zeta_r: float = 1.0
    source: str = "ft"
    approach_v: float = 0.0
    f_thresh: float = 1.0
    i_max: float = 50.0
    f_filter: float = 0.0                      # 0 = no filtering; else first-order low-pass coefficient in (0, 1)

    def __post_init__(self):
        self.I = 0.0
        self.contact = self.approach_v <= 0.0
        self.f_f = 0.0

    def __call__(self, rig, t):
        J, M = rig.jacobian(), rig.mass_matrix()
        Lam = np.linalg.inv(J @ np.linalg.solve(M, J.T))
        S = np.diag(list(self.S_diag))
        F_ext = rig.ft_force() if self.source == "ft" else rig.contact_force()
        f_raw = F_ext[2]                                            # pressing force: the plate pushes the tool up (+z)
        a = self.f_filter
        self.f_f = f_raw if a == 0 else (1 - a) * self.f_f + a * f_raw
        f = self.f_f
        p_d, v_d, a_d = self.target(t)
        v = (J @ rig.qd)[:3]
        e_p, e_v = p_d - rig.p, v_d - v
        if not self.contact and f > self.f_thresh:
            self.contact = True
        if self.contact:
            err = self.f_d - f
            self.I = float(np.clip(self.I + err * rig.model.opt.timestep, -self.i_max, self.i_max))
            f_cmd = np.array([0.0, 0.0, -(self.f_d + self.K_f * err + self.K_i * self.I) - self.B_f * v[2]])
            a_pos = a_d + self.K_v * e_v + self.K_p * e_p
            S3 = S
        else:                                                       # free-space approach: velocity control along z
            e_p = e_p * np.array([1.0, 1.0, 0.0])
            e_v = e_v.copy()
            e_v[2] = -self.approach_v - v[2]
            f_cmd = np.zeros(3)
            a_pos = a_d + self.K_v * e_v + self.K_p * e_p
            S3 = np.eye(3)
        # 6-D operational-space hybrid law; orientation always position controlled
        S6 = np.eye(6)
        S6[:3, :3] = S3
        e_r = orientation_error(rig.R, rig.R_down)                  # orientation held by a PD wrench
        w_r = self.K_r * e_r - 2 * self.zeta_r * np.sqrt(self.K_r * np.mean(np.diag(Lam[3:, 3:]))) * (J @ rig.qd)[3:]
        a6 = np.r_[a_pos, np.zeros(3)]
        w = S6 @ Lam @ S6 @ a6
        w[:3] += (np.eye(3) - S3) @ f_cmd
        w[3:] += w_r
        tau = J.T @ w + rig.bias() + _nullspace_posture(rig, J, M, Lam)
        return "tau", tau, {"f": f, "f_raw": f_raw, "p_d": p_d, "contact": float(self.contact)}


@dataclass
class JointServoTarget:
    """Position-controlled robot following a straight-line / fixed target through differential inverse kinematics.

    Uses the manufacturer's joint position servos (``("q", q_cmd)``), i.e. the setting of an ordinary industrial arm.
    ``target(t) -> p_d`` (tool position, orientation held); the commanded joint angles are advanced by
    ``dq = J^+ (p_d - p_cmd)`` on a private copy of the model, so that they never depend on the measured state.
    """
    target: Callable

    def __post_init__(self):
        self.q_cmd = None
        self.d2 = None

    def __call__(self, rig, t):
        m = rig.model
        if self.q_cmd is None:
            self.q_cmd = rig.q
            self.d2 = mujoco.MjData(m)
        d2 = self.d2
        d2.qpos[:7] = self.q_cmd
        mujoco.mj_kinematics(m, d2)
        mujoco.mj_comPos(m, d2)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d2, jp, jr, rig.tcp)
        J = np.vstack([jp[:, :7], jr[:, :7]])
        Rc = d2.site_xmat[rig.tcp].reshape(3, 3)
        e = np.r_[self.target(t) - d2.site_xpos[rig.tcp], orientation_error(Rc, rig.R_down)]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        self.q_cmd = self.q_cmd + dq
        return "q", self.q_cmd, {"p_d": self.target(t)}


@dataclass
class AdmittanceControl:
    """Outer admittance loop around the position-controlled arm (the usual set-up of industrial robots).

    The virtual tool (position ``x_c`` along the world axes ``axes``) obeys ``m_a (x_c'' - a_d) + b_a (x_c' - v_d)
    + k_a (x_c - x_d) = F_ext`` with ``F_ext`` the environment-on-tool force from the wrist sensor; the other
    axes follow ``target`` exactly. The virtual pose is turned into joint targets by differential kinematics.
    """
    target: Callable                  # t -> p_d (3,)
    m_a: float = 2.0
    b_a: float = 60.0
    k_a: float = 500.0
    axes: tuple = (2,)
    f_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.x_c = None
        self.v_c = np.zeros(3)
        self.ik = JointServoTarget(self._pc)

    def _pc(self, t):
        return self.x_c

    def __call__(self, rig, t):
        dt = rig.model.opt.timestep
        p_d = self.target(t)
        if self.x_c is None:
            self.x_c = p_d.copy()
        F_ext = rig.ft_force() - self.f_bias
        for ax in self.axes:
            acc = (F_ext[ax] - self.b_a * self.v_c[ax] - self.k_a * (self.x_c[ax] - p_d[ax])) / self.m_a
            self.v_c[ax] += dt * acc
            self.x_c[ax] += dt * self.v_c[ax]
        for ax in set(range(3)) - set(self.axes):
            self.x_c[ax] = p_d[ax]
        kind, q_cmd, _ = self.ik(rig, t)
        return kind, q_cmd, {"x_c": self.x_c, "p_d": p_d}
