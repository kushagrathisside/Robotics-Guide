"""MuJoCo arms driven by joint torques, for the control experiments of RMC Chapter 7.

The Menagerie arms ship with *position* actuators (a built-in PD law). Optimal and learning-based control
need the opposite: a plant whose input is the joint torque vector. :func:`make_arm` loads a robot from
:mod:`rgkit.robots`, optionally adds a payload (a sphere with a given mass attached to the tool frame), turns
every actuator into an ideal torque source, optionally removes joint friction, and returns a compiled model.
:class:`TorquePlant` then exposes the state ``x = (q, qdot)`` and the one-step map ``x+ = f(x, u)`` and its
Jacobians (through ``mujoco.mjd_transitionFD``), plus the quantities a model-based controller needs
(the mass matrix and the bias force) — all of them taken from MuJoCo itself.

Only hinge/slide-joint arms are supported (``nq == nv``).
"""

from __future__ import annotations

import numpy as np

from . import mujoco_utils as mju
from . import robots

mujoco = mju.mujoco

__all__ = ["make_arm", "TorquePlant", "ARMS", "draw_arm_xz"]

#: per robot: model file, name of the tool site, body that carries the payload, torque limits used in the notebooks
ARMS = {
    "2r": dict(site="end", body="second_segment", tau_max=(6.0, 6.0)),
    "ur5e": dict(site="attachment_site", body="wrist_3_link", tau_max=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0)),
}


def _add_payload(spec, body_name: str, site_name: str, mass: float, radius: float):
    """Attach a solid sphere of ``mass`` at a site by updating the body's explicit inertial properties.

    Menagerie bodies carry an ``<inertial>`` element, so a new geom would not change their dynamics; the
    combined mass, centre of mass and inertia tensor are therefore computed with the parallel-axis theorem.
    """
    m0 = spec.compile()
    body = spec.body(body_name)
    bid = m0.body(body_name).id
    site = next(s for s in spec.sites if s.name == site_name)
    p_site = np.array(site.pos, float)               # expressed in the body frame, as is body_ipos
    m_b, c_b = float(m0.body_mass[bid]), m0.body_ipos[bid].copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m0.body_iquat[bid])
    R = R.reshape(3, 3)
    I_b = R @ np.diag(m0.body_inertia[bid]) @ R.T   # inertia about the body's own COM, body-frame axes
    M = m_b + mass
    c = (m_b * c_b + mass * p_site) / M
    shift = lambda mm, d: mm * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    I = I_b + shift(m_b, c_b - c) + (0.4 * mass * radius ** 2) * np.eye(3) + shift(mass, p_site - c)
    w, V = np.linalg.eigh(0.5 * (I + I.T))           # principal axes: MuJoCo stores diag inertia + orientation
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1.0
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, V.ravel())
    body.explicitinertial = True
    body.mass = M
    body.ipos = c
    body.fullinertia = np.full(6, np.nan)
    body.inertia = w
    body.iquat = quat
    g = body.add_geom()                               # massless marker so that the payload is visible
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[0] = radius
    g.pos = p_site
    g.contype, g.conaffinity = 0, 0
    g.rgba = [0.9, 0.35, 0.1, 1.0]
    g.mass = 0.0
    g.name = "payload"


def make_arm(name: str = "2r", *, payload: float = 0.0, payload_radius: float = 0.025, frictionloss=None,
             damping=None, dt: float | None = 0.005, torque_limit="default", with_scene: bool = True):
    """Compile a torque-controlled model of a Menagerie arm.

    Parameters
    ----------
    name : ``"2r"`` (Dynamixel 2R, moves in a vertical plane) or ``"ur5e"``.
    payload : mass in kg of a sphere fixed to the tool site (``0`` for none).
    frictionloss, damping : override joint Coulomb friction (N m) / viscous damping (N m s/rad); ``None`` keeps
        the manufacturer's values.
    dt : integration step in seconds (one control step equals one simulation step).
    torque_limit : ``"default"`` (see :data:`ARMS`), ``None`` for none, or an array of per-joint limits.
    """
    spec_info = ARMS[name]
    robots.load_robot(name)                                    # makes sure the model (and its scene file) is cached
    d = robots.fetch(name)
    path = d / ("rg_scene_light.xml" if with_scene else robots.ROBOTS[name]["robot"])
    spec = mujoco.MjSpec.from_file(str(path))
    if payload > 0:
        _add_payload(spec, spec_info["body"], spec_info["site"], payload, payload_radius)
    m = spec.compile()
    if dt is not None:
        m.opt.timestep = dt
    # --- ideal torque sources: force = ctrl
    m.actuator_gainprm[:, 0] = 1.0
    m.actuator_gainprm[:, 1:] = 0.0
    m.actuator_biasprm[:] = 0.0
    m.actuator_biastype[:] = 0
    m.actuator_ctrllimited[:] = 0
    if torque_limit == "default":
        torque_limit = spec_info["tau_max"]
    if torque_limit is None:
        m.actuator_forcelimited[:] = 0
    else:
        lim = np.asarray(torque_limit, float) * np.ones(m.nu)
        # actuator i drives joint trnid[i]; the limit list is per joint in joint order
        jid = m.actuator_trnid[:, 0]
        m.actuator_forcelimited[:] = 1
        m.actuator_forcerange[:, 0] = -lim[jid]
        m.actuator_forcerange[:, 1] = lim[jid]
    if frictionloss is not None:
        m.dof_frictionloss[:] = frictionloss
    if damping is not None:
        m.dof_damping[:] = damping
    return m


class TorquePlant:
    """Joint-torque interface to a MuJoCo arm: ``x = (q, qdot)``, ``u = tau`` (one value per joint).

    ``step`` advances the simulation by one timestep of the model. ``f(x, u)`` is the same map written as a pure
    function (it overwrites the internal state), which is what trajectory optimisers need.
    """

    def __init__(self, model, site: str | None = None):
        if model.nq != model.nv:
            raise ValueError("only models with nq == nv (hinge/slide joints) are supported")
        if model.nu != model.nv:
            raise ValueError("expected one torque actuator per joint")
        self.model = model
        self.data = mujoco.MjData(model)
        self.nq = self.nv = model.nv
        self.nx, self.nu = 2 * model.nv, model.nu
        self.dt = model.opt.timestep
        self.site = site
        # ctrl[a] drives joint trnid[a]; perm[j] is the actuator of joint j
        dof_of_act = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        self.perm = np.empty(self.nv, int)
        self.perm[dof_of_act] = np.arange(model.nu)
        self.tau_lo = np.full(self.nv, -np.inf)
        self.tau_hi = np.full(self.nv, np.inf)
        if model.actuator_forcelimited.all():
            self.tau_lo[dof_of_act] = model.actuator_forcerange[:, 0]
            self.tau_hi[dof_of_act] = model.actuator_forcerange[:, 1]
        self.reset()

    # -- state handling --------------------------------------------------------------------------------
    def reset(self, x=None):
        mujoco.mj_resetData(self.model, self.data)
        if x is not None:
            self.set_state(x)
        else:
            mujoco.mj_forward(self.model, self.data)
        return self.state()

    def set_state(self, x):
        x = np.asarray(x, float)
        self.data.qpos[:] = x[:self.nq]
        self.data.qvel[:] = x[self.nq:]
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _ctrl(self, u):
        c = np.empty(self.nu)
        c[self.perm] = np.asarray(u, float)
        self.data.ctrl[:] = c

    # -- dynamics --------------------------------------------------------------------------------------
    def step(self, u) -> np.ndarray:
        """Apply joint torques ``u`` for one timestep; return the new state."""
        self._ctrl(u)
        mujoco.mj_step(self.model, self.data)
        return self.state()

    def f(self, x, u) -> np.ndarray:
        """Pure one-step map ``x+ = f(x, u)`` (does not keep track of warm starts or contacts)."""
        self.set_state(x)
        return self.step(u)

    def clip(self, u) -> np.ndarray:
        return np.clip(u, self.tau_lo, self.tau_hi)

    def linearize(self, x, u, eps: float = 1e-6):
        """``(A, B)`` of the one-step map at ``(x, u)`` from ``mujoco.mjd_transitionFD`` (centred differences)."""
        self.set_state(x)
        self._ctrl(u)
        mujoco.mj_forward(self.model, self.data)
        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nu))
        mujoco.mjd_transitionFD(self.model, self.data, eps, True, A, B, None, None)
        return A, B[:, self.perm]

    # -- model-based quantities --------------------------------------------------------------------------
    def mass_matrix(self, q) -> np.ndarray:
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, self.data, M)
        return M

    def bias(self, q, qd=None) -> np.ndarray:
        """``h(q, qdot) = C(q, qdot) qdot + g(q)`` — the torque that holds ``qacc = 0``."""
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0 if qd is None else qd
        mujoco.mj_forward(self.model, self.data)
        out = np.zeros(self.nv)
        self.data.qacc[:] = 0.0
        mujoco.mj_rne(self.model, self.data, 0, out)
        return out

    def gravity(self, q) -> np.ndarray:
        return self.bias(q, None)

    def inverse_dynamics(self, q, qd, qdd) -> np.ndarray:
        """``M(q) qdd + h(q, qd)`` (rigid-body part only: no friction, no joint damping)."""
        return self.mass_matrix(q) @ np.asarray(qdd, float) + self.bias(q, qd)

    def tool_position(self, q) -> np.ndarray:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site(self.site).xpos.copy()

    def tool_jacobian(self, q) -> np.ndarray:
        """3 x nv translational Jacobian of the tool site (world frame)."""
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        Jp = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, Jp, None, self.model.site(self.site).id)
        return Jp

    def joint_positions(self, q) -> np.ndarray:
        """World positions of the joint anchors followed by the tool site, shape ``(nv + 1, 3)``."""
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        pts = [self.data.xanchor[j].copy() for j in range(self.model.njnt)]
        if self.site is not None:
            pts.append(self.data.site(self.site).xpos.copy())
        return np.array(pts)


def draw_arm_xz(ax, pts, color="#444", lw=6, alpha=0.7, joint_color="k", tool_color="#ff7f0e", label=None):
    """Draw a planar arm in the world x-z plane from ``pts = TorquePlant.joint_positions(q)`` (matplotlib)."""
    pts = np.asarray(pts)
    ax.plot(pts[:, 0], pts[:, 2], "-", color=color, lw=lw, solid_capstyle="round", alpha=alpha, label=label)
    ax.plot(pts[:-1, 0], pts[:-1, 2], "o", color=joint_color, ms=7)
    ax.plot(pts[-1:, 0], pts[-1:, 2], "o", color=tool_color, ms=8)
