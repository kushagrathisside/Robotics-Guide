"""A master-slave pair of two real arms in one MuJoCo scene (Robot Motion Control, Chapter 6).

Both arms are the **Dynamixel 2R** planar arm of MuJoCo Menagerie (see :func:`rgkit.robots.describe`),
attached to one world with ``MjSpec.attach`` so that every joint, site and body of the two copies carries the
prefix ``m_`` (master) or ``s_`` (slave).  The arms hang from their base and move in the vertical x-z plane;
with ``q = 0`` both links hang straight down.  A rigid vertical **wall** stands beside the slave, and the 7 cm-radius
disk at the end of the slave's last link ("tip") is the only geom that can touch it.

The class :class:`Pair` offers what the notebooks need: gravity compensation, planar Jacobians and tip
positions, the contact force on the slave tip, a *scripted human* (a Cartesian spring-damper that pulls the
master's tip towards an intended path) and a joint-space teleoperation loop that runs any
:class:`rgkit.teleop.Architecture` on each joint, with communication delay.

All quantities are SI (radian, metre, newton-metre).  The planar kinematics of one arm are
``p = (L1 sin q1 + L2 sin(q1+q2),  z0 - L1 cos q1 - L2 cos(q1+q2))`` with ``L1 = L2 = 0.18 m`` and the first joint
axis at height ``z0 = 0.5452 m`` (checked against the simulator in the notebooks).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import mujoco_utils as mju
from . import robots
from .teleop import Sig

mujoco = mju.mujoco

L1 = L2 = 0.18
Z0 = 0.5452
TIP_R = 0.07
__all__ = ["Pair", "PairResult", "planar_fk", "planar_jacobian", "L1", "L2", "Z0", "TIP_R"]


def planar_fk(q):
    """Tip position ``(x, z)`` of one arm relative to its first joint axis (x) and the floor (z)."""
    q1, q12 = q[0], q[0] + q[1]
    return np.array([L1 * np.sin(q1) + L2 * np.sin(q12), Z0 - L1 * np.cos(q1) - L2 * np.cos(q12)])


def planar_jacobian(q):
    """2x2 Jacobian of :func:`planar_fk`."""
    q1, q12 = q[0], q[0] + q[1]
    return np.array([[L1 * np.cos(q1) + L2 * np.cos(q12), L2 * np.cos(q12)],
                     [L1 * np.sin(q1) + L2 * np.sin(q12), L2 * np.sin(q12)]])


@dataclass
class PairResult:
    """Histories of a :meth:`Pair.run` (arrays of shape ``(n, ...)``)."""

    t: np.ndarray
    qm: np.ndarray        # master joint angles (n, 2)
    qs: np.ndarray        # slave joint angles (n, 2)
    pm: np.ndarray        # master tip (x, z) (n, 2)
    ps: np.ndarray        # slave tip (x, z) (n, 2)
    f_hand: np.ndarray    # Cartesian hand force on the master tip (n, 2)
    f_env: np.ndarray     # Cartesian contact force on the slave tip from the wall (n, 2)
    tau_m: np.ndarray     # joint torques commanded by the controller on the master (n, 2)
    tau_s: np.ndarray     # same for the slave
    frames: list          # list of (t, rgb image) when render_at was given


class Pair:
    """Two Dynamixel 2R arms and a wall in one scene.

    Parameters
    ----------
    touch_x : x-position of the slave's tip centre at the instant it touches the wall, measured from the slave's
        base axis (metres).  The tip is the 7 cm-radius disk at the end of the arm, so the wall face is at
        ``touch_x + 0.07``.
    sep : distance between the two base axes along world x (metres); master at ``-sep/2``, slave at ``+sep/2``.
    damping : viscous friction added to every joint (N m s/rad) -- the Menagerie model has none.
    friction : keep the model's dry friction (``frictionloss = 0.1 N m``) if True, remove it if False.
    dt : simulation (and control) time step.
    """

    def __init__(self, touch_x=0.22, sep=0.9, damping=0.02, friction=False, dt=0.001):
        d = robots.fetch("2r")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parent = mujoco.MjSpec.from_string(f"""
<mujoco model="teleop pair">
  <option timestep="{dt}" integrator="implicitfast" noslip_iterations="1"/>
  {mju.scene_assets()}
  <worldbody>
    {mju.floor(z=0.0, size=2)}
    <frame name="mf" pos="{-sep / 2} 0 0"/>
    <frame name="sf" pos="{sep / 2} 0 0"/>
    <geom name="plate_m" type="box" pos="{-sep / 2} 0 0.595" size="0.05 0.06 0.004" rgba="0.2 0.45 0.8 1" contype="0" conaffinity="0"/>
    <geom name="plate_s" type="box" pos="{sep / 2} 0 0.595" size="0.05 0.06 0.004" rgba="0.95 0.5 0.1 1" contype="0" conaffinity="0"/>
    <geom name="wall" type="box" pos="{sep / 2 + touch_x + TIP_R + 0.01} 0 0.30" size="0.01 0.12 0.30" rgba="0.55 0.57 0.62 0.9" friction="0.1 0.005 0.0001"/>
  </worldbody>
</mujoco>""")
            for prefix, frame in (("m_", "mf"), ("s_", "sf")):
                child = mujoco.MjSpec.from_file(str(d / "dynamixel_2r.xml"))
                seg = child.body("second_segment")
                rgba = [0.2, 0.45, 0.8, 1] if prefix == "m_" else [0.95, 0.5, 0.1, 1]
                col = 1 if prefix == "s_" else 0
                seg.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[TIP_R, 0.007, 0], pos=[0, -L2, 0.02347],
                             name="tip", rgba=[0, 0, 0, 0], contype=col, conaffinity=col, friction=[0.1, 0.005, 0.0001])
                seg.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.04, 0.0085, 0], pos=[0, -L2, 0.02347],
                             name="marker", rgba=rgba, contype=0, conaffinity=0)
                parent.attach(child, prefix=prefix, frame=parent.frame(frame))
            self.model = parent.compile()
        m = self.model
        m.actuator_gainprm[:, :] = 0.0                    # the built-in position servos are switched off:
        m.actuator_biasprm[:, :] = 0.0                    # we apply joint torques ourselves
        m.dof_damping[:] = damping
        if not friction:
            m.dof_frictionloss[:] = 0.0
        self.data = mujoco.MjData(m)
        self._gdata = mujoco.MjData(m)                    # scratch data for gravity terms
        self.dt = dt
        self.sep, self.touch_x = sep, touch_x
        self.dof = {a: np.array([m.joint(f"{a}_R1").dofadr[0], m.joint(f"{a}_R2").dofadr[0]]) for a in ("m", "s")}
        self.qadr = {a: np.array([m.joint(f"{a}_R1").qposadr[0], m.joint(f"{a}_R2").qposadr[0]]) for a in ("m", "s")}
        self.site = {a: m.site(f"{a}_end").id for a in ("m", "s")}
        self.geom_wall, self.geom_tip = m.geom("wall").id, m.geom("s_tip").id
        self.origin = {"m": -sep / 2, "s": sep / 2}
        self.reset()

    # -- state ------------------------------------------------------------------------------------------
    def reset(self, qm=(0.0, 0.0), qs=(0.0, 0.0)):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.qadr["m"]] = qm
        self.data.qpos[self.qadr["s"]] = qs
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def q(self, arm):
        return self.data.qpos[self.qadr[arm]].copy()

    def qd(self, arm):
        return self.data.qvel[self.dof[arm]].copy()

    def tip(self, arm):
        """Tip position (x, z) of an arm in its *own* planar coordinates (x measured from its base axis)."""
        p = self.data.site_xpos[self.site[arm]]
        return np.array([p[0] - self.origin[arm], p[2]])

    def jac(self, arm):
        """Planar Jacobian (x, z) x (R1, R2) of an arm from the simulator (``mj_jacSite``)."""
        Jp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, Jp, None, self.site[arm])
        return Jp[np.ix_([0, 2], self.dof[arm])]

    def mass_matrix(self, arm):
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M)
        return M[np.ix_(self.dof[arm], self.dof[arm])]

    def gravity(self):
        """Joint torques that hold the arms still against gravity (``qfrc_bias`` at zero velocity)."""
        g = self._gdata
        g.qpos[:] = self.data.qpos
        g.qvel[:] = 0.0
        mujoco.mj_forward(self.model, g)
        return g.qfrc_bias.copy()

    def contact_force(self):
        """World-frame (x, z) force that the wall exerts on the slave's tip, newtons."""
        F = np.zeros(2)
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if set(pair) != {self.geom_wall, self.geom_tip}:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            Fw = c.frame.reshape(3, 3).T @ f6[:3]       # force of geom1 on geom2, in world axes
            if c.geom2 == self.geom_wall:
                Fw = -Fw
            F += Fw[[0, 2]]
        return F

    def step(self, tau_m, tau_s):
        """Apply joint torques (gravity is compensated here) and advance one time step."""
        d = self.data
        d.qfrc_applied[:] = self.gravity()
        d.qfrc_applied[self.dof["m"]] += tau_m
        d.qfrc_applied[self.dof["s"]] += tau_s
        mujoco.mj_step(self.model, d)

    def render(self, camera=None, width=480, height=360):
        camera = camera or dict(azimuth=90, elevation=-6, distance=1.45, lookat=(0.0, 0.0, 0.36))
        return mju.render(self.model, self.data, camera=camera, width=width, height=height)

    # -- teleoperation loop --------------------------------------------------------------------------------
    def run(self, arch, path, t_end=4.0, delay=(0.0, 0.0), hand=(60.0, 6.0), qm0=(0.0, 0.0), qs0=None,
            render_at=(), camera=None, size=(400, 300), contact_gain=None):
        """Run joint-space teleoperation.

        ``arch`` is any :class:`rgkit.teleop.Architecture`; it is applied to *each joint separately* with the
        joint angle in place of position, the joint torque of the hand in place of the hand force and
        ``f_e = -(J_s^T F_wall)`` in place of the environment force.  ``path(t)`` returns the *intended* tip
        position (x, z) of the scripted human, who pulls the master tip towards it with the Cartesian
        spring-damper ``hand = (K_h, B_h)``.  ``delay = (T_fwd, T_bwd)`` in seconds (at least one sample).
        ``render_at`` lists times at which an RGB frame of the scene is stored.
        """
        qs0 = qm0 if qs0 is None else qs0
        self.reset(qm0, qs0)
        n = int(round(t_end / self.dt))
        Nf, Nb = max(1, int(round(delay[0] / self.dt))), max(1, int(round(delay[1] / self.dt)))
        nfwd, nbwd = arch.n_fwd, arch.n_bwd
        fwd = np.zeros((n, 2, nfwd))
        bwd = np.zeros((n, 2, nbwd))
        out = {k: np.zeros((n, 2)) for k in "qm qs pm ps f_hand f_env tau_m tau_s".split()}
        render_steps = {int(round(t / self.dt)): t for t in render_at}
        frames = []
        for k in range(n):
            t = k * self.dt
            qm, qs, qdm, qds = self.q("m"), self.q("s"), self.qd("m"), self.qd("s")
            Jm, Js = self.jac("m"), self.jac("s")
            pm, ps = self.tip("m"), self.tip("s")
            F_hand = hand[0] * (np.asarray(path(t)) - pm) - hand[1] * (Jm @ qdm)
            tau_h = Jm.T @ F_hand                               # hand torques on the master joints
            F_env = self.contact_force()
            f_e = -(Js.T @ F_env)                               # >0 when the wall resists positive joint motion
            tm = np.zeros(2)
            ts = np.zeros(2)
            for j in range(2):
                m_sig = Sig(qm[j], qdm[j], 0.0, tau_h[j])
                s_sig = Sig(qs[j], qds[j], 0.0, f_e[j])
                rx_f = fwd[k - Nf, j] if k >= Nf else fwd[0, j]
                rx_b = bwd[k - Nb, j] if k >= Nb else bwd[0, j]
                if k == 0:
                    fwd[0, j] = arch.fwd(m_sig, np.zeros(nbwd))
                    bwd[0, j] = arch.bwd(s_sig, np.zeros(nfwd))
                    rx_f, rx_b = fwd[0, j], bwd[0, j]
                else:
                    fwd[k, j] = arch.fwd(m_sig, rx_b)
                    bwd[k, j] = arch.bwd(s_sig, rx_f)
                tm[j] = arch.master(m_sig, rx_b)
                ts[j] = arch.slave(s_sig, rx_f)
            for key, val in dict(qm=qm, qs=qs, pm=pm, ps=ps, f_hand=F_hand, f_env=F_env, tau_m=tm, tau_s=ts).items():
                out[key][k] = val
            if k in render_steps:
                frames.append((render_steps[k], self.render(camera, *size)))
            self.step(tm + tau_h, ts)
        return PairResult(t=np.arange(n) * self.dt, frames=frames, **out)
