"""Tests for rgkit.joint_sim: single-joint experiments in MuJoCo against closed-form models (no downloads)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from rgkit import joint_sim as js          # noqa: E402
from rgkit import linear_control as lc     # noqa: E402

TWO_LINK = """
<mujoco>
  <option timestep="0.0005" integrator="implicitfast"/>
  <worldbody>
    <body name="l1">
      <joint name="a" axis="0 1 0" armature="0.01"/>
      <inertial pos="0 0 -0.2" mass="1.0" diaginertia="0.01 0.01 0.001"/>
      <body name="l2" pos="0 0 -0.4">
        <joint name="b" axis="0 1 0" armature="0.005"/>
        <inertial pos="0 0 -0.15" mass="0.5" diaginertia="0.004 0.004 0.001"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a" joint="a" kp="40" kv="1.5"/>
    <position name="b" joint="b" kp="30" kv="0.8"/>
  </actuator>
</mujoco>"""


def test_servo_gains_and_inertia_by_hand():
    m = mujoco.MjModel.from_xml_string(TWO_LINK)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = js.servo_gains(m, "b")
    assert (g["kp"], g["kv"], g["dof"]) == (30.0, 0.8, 1)
    # distal joint: armature + Icom + m d^2 (parallel-axis theorem)
    J = 0.005 + 0.004 + 0.5 * 0.15**2
    assert np.isclose(js.effective_inertia(m, d, 1), J)
    # with the other joint free the apparent inertia is smaller
    assert js.effective_inertia(m, d, 1, others="free") < J
    # gravity stiffness of a pendulum hanging below its pivot: + m g d
    assert np.isclose(js.gravity_stiffness(m, d, 1), 0.5 * 9.81 * 0.15, rtol=1e-6)


def test_locked_dof_does_not_move_and_is_restored():
    m = mujoco.MjModel.from_xml_string(TWO_LINK)
    d = mujoco.MjData(m)
    d.qpos[:] = [0.3, 0.0]
    mujoco.mj_forward(m, d)
    before = m.dof_armature.copy()
    t, q, qd = js.servo_step(m, d, "b", 0.4, 1.0)
    assert np.allclose(m.dof_armature, before) and np.allclose(d.qpos, [0.3, 0.0])
    assert len(t) == len(q)


def test_servo_step_matches_second_order_model():
    m = mujoco.MjModel.from_xml_string(TWO_LINK)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    J = js.effective_inertia(m, d, 1)
    kg = js.gravity_stiffness(m, d, 1)
    g = js.servo_gains(m, "b")
    amp = 0.02                                              # small step: linear regime
    t, q, _ = js.servo_step(m, d, "b", amp, 2.0)
    # linearised model about q = 0: J q'' + kv q' + (kp + kg) q = kp u
    G = lc.TF([g["kp"]], [J, g["kv"], g["kp"] + kg])
    assert np.allclose(q, amp * G.step(t), atol=2e-4)




def test_dcmotor_joint_matches_transfer_function():
    p = js.dc_joint_params()
    m = mujoco.MjModel.from_xml_string(js.dc_joint_xml(p, gravity=False, with_scene=False))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert np.isclose(js.effective_inertia(m, d, 0), p["J"])
    G = js.dc_joint_tf(p)
    V = 5.0
    out = js.simulate(m, d, lambda t, y: V, T=0.5, Ts=2e-4)
    ref = V * G.step(out["t"])
    assert np.abs(out["q"] - ref).max() < 2e-3 * ref[-1]       # 0.2 % of the final angle
    # current settles to the friction-only value: i = (V - N Ke qd)/R
    assert np.isclose(out["i"][-1], (V - p["N"] * p["Ke"] * out["qd"][-1]) / p["R"], rtol=0.03)


def test_dcmotor_joint_saturates_voltage():
    p = js.dc_joint_params()
    m = mujoco.MjModel.from_xml_string(js.dc_joint_xml(p, gravity=False, with_scene=False))
    d = mujoco.MjData(m)
    big = js.simulate(m, d, lambda t, y: 1000.0, T=0.3, Ts=2e-4)
    lim = js.simulate(m, d, lambda t, y: p["vmax"], T=0.3, Ts=2e-4)
    assert np.allclose(big["q"], lim["q"])


def test_fd_linearization_matches_analytic_eigenvalues():
    p = js.dc_joint_params()
    m = mujoco.MjModel.from_xml_string(js.dc_joint_xml(p, with_scene=False))
    d = mujoco.MjData(m)
    A, B, Ad, Bd = js.fd_linearize(m, d)
    Aan, Ban, _, _ = lc.dc_motor_ss(p["Kt"], p["Ke"], p["R"], p["L"], p["J"], p["b"], p["N"], kg=p["mgd"])
    ev, evan = np.sort(np.linalg.eigvals(A).real), np.sort(np.linalg.eigvals(Aan).real)
    assert np.allclose(ev, evan, rtol=5e-3)
    # upright equilibrium: one unstable eigenvalue
    d.qpos[0] = np.pi
    A, *_ = js.fd_linearize(m, d)
    assert np.sum(np.linalg.eigvals(A).real > 0) == 1


def test_simulate_zero_order_hold_and_sampling():
    p = js.dc_joint_params()
    m = mujoco.MjModel.from_xml_string(js.dc_joint_xml(p, with_scene=False))
    d = mujoco.MjData(m)
    calls = []
    out = js.simulate(m, d, lambda t, y: calls.append(t) or 0.0, T=0.1, Ts=0.01)
    assert len(calls) == 11 and np.allclose(np.diff(calls), 0.01)
    with pytest.raises(ValueError):
        js.simulate(m, d, lambda t, y: 0.0, T=0.1, Ts=0.00033)
