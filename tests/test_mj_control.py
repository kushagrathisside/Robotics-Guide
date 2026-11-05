import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
from rgkit import mujoco_utils as mju  # noqa: E402

pytestmark = pytest.mark.skipif(not mju.available(), reason="mujoco not installed")


@pytest.fixture(scope="module")
def arm():
    from rgkit import mj_control as mc
    try:
        m = mc.make_arm("2r", frictionloss=0.0, torque_limit=None)
    except Exception as e:                     # model download not possible offline
        pytest.skip(f"2R model unavailable: {e}")
    return mc.TorquePlant(m, site="end")


def test_gravity_torque_holds_the_arm(arm):
    q = np.array([0.8, -0.4])
    arm.reset(np.r_[q, 0, 0])
    tau = arm.gravity(q)
    arm.set_state(np.r_[q, 0, 0])
    for _ in range(200):
        x = arm.step(tau)
    assert np.allclose(x[:2], q, atol=1e-6) and np.allclose(x[2:], 0, atol=1e-6)


def test_mass_matrix_symmetric_positive_definite_and_inverse_dynamics(arm):
    q = np.array([0.3, 1.1])
    M = arm.mass_matrix(q)
    assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)
    # applying tau = M qdd + h from rest gives, in one semi-implicit Euler step, qd+ = dt * qdd
    qdd = np.array([1.0, -2.0])
    tau = arm.inverse_dynamics(q, [0, 0], qdd)
    x1 = arm.f(np.r_[q, 0, 0], tau)
    assert np.allclose(x1[2:], arm.dt * qdd, atol=1e-6)


def test_transition_jacobians_match_central_differences(arm):
    from rgkit.optimal_control import linearize
    x0 = np.array([0.8, -0.4, 0.2, 0.1])
    u0 = arm.gravity(x0[:2])
    A, B = arm.linearize(x0, u0)
    A2, B2 = linearize(arm.f, x0, u0, 1e-6)
    assert np.allclose(A, A2, atol=1e-6) and np.allclose(B, B2, atol=1e-6)


def test_payload_adds_mass_and_torque_limit_is_enforced():
    from rgkit import mj_control as mc
    try:
        m0 = mc.make_arm("2r")
        m1 = mc.make_arm("2r", payload=0.3)
    except Exception as e:
        pytest.skip(f"2R model unavailable: {e}")
    assert np.isclose(m1.body_mass.sum() - m0.body_mass.sum(), 0.3)
    p = mc.TorquePlant(m0, site="end")
    assert np.allclose(p.tau_hi, 6.0)
    x = p.f(np.zeros(4), np.array([100.0, 0.0]))          # saturates at 6 N m on joint R1
    x_ref = p.f(np.zeros(4), np.array([6.0, 0.0]))
    assert np.allclose(x, x_ref)


def test_tool_jacobian_matches_finite_difference(arm):
    q = np.array([0.7, -0.2])
    J = arm.tool_jacobian(q)
    Jfd = np.stack([(arm.tool_position(q + 1e-6 * e) - arm.tool_position(q - 1e-6 * e)) / 2e-6 for e in np.eye(2)], 1)
    assert np.allclose(J, Jfd, atol=1e-6)


def test_payload_changes_mass_matrix_like_a_point_mass_plus_sphere_inertia():
    from rgkit import mj_control as mc
    try:
        p0 = mc.TorquePlant(mc.make_arm("2r"), site="end")
        p1 = mc.TorquePlant(mc.make_arm("2r", payload=0.3, payload_radius=0.025), site="end")
    except Exception as e:
        pytest.skip(f"2R model unavailable: {e}")
    import mujoco
    q = np.array([0.6, -0.9])
    p1.data.qpos[:] = q
    mujoco.mj_forward(p1.model, p1.data)
    Jp, Jr = np.zeros((3, 2)), np.zeros((3, 2))
    mujoco.mj_jacSite(p1.model, p1.data, Jp, Jr, p1.model.site("end").id)
    dM = 0.3 * Jp.T @ Jp + 0.4 * 0.3 * 0.025 ** 2 * Jr.T @ Jr
    assert np.allclose(p1.mass_matrix(q) - p0.mass_matrix(q), dM, atol=1e-9)
    assert np.allclose(p1.gravity(q) - p0.gravity(q), 0.3 * 9.81 * Jp[2] , atol=1e-9)
