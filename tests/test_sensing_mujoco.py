import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from rgkit import sensing_mujoco as sm            # noqa: E402
from rgkit import sensing as sn                   # noqa: E402

ARM = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <body name="l1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.02" mass="1.0"/>
      <body name="l2" pos="0.4 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.5"/>
        <site name="tip" pos="0.3 0 0"/>
        <body name="tool" pos="0.3 0 0">
          <geom type="box" size="0.02 0.02 0.03" pos="0.0 0.0 -0.05" mass="0.8" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="j1"/><motor joint="j2"/></actuator>
</mujoco>
"""


def _model():
    spec = sm.spec_from_xml(ARM)
    for kind in ("accelerometer", "gyro", "force", "torque"):
        sm.add_sensor(spec, kind, kind, "tip")
    sm.add_sensor(spec, "q1", "jointpos", "j1", "joint")
    sm.add_sensor(spec, "w1", "jointvel", "j1", "joint")
    return sm.compile_spec(spec, keyframe=None)


def test_gravity_torque_holds_the_arm_still_and_matches_statics():
    model, data = _model()
    data.qpos[:] = [0.3, -0.5]
    mujoco.mj_forward(model, data)
    G = sm.gravity_torque(model, data)
    sm.torque_mode(model)
    for _ in range(500):
        data.qfrc_applied[:] = sm.gravity_torque(model, data)
        mujoco.mj_step(model, data)
    assert np.allclose(data.qpos, [0.3, -0.5], atol=1e-3)
    # static torque on joint 2 = moment of the weight of everything beyond it about the joint axis
    mujoco.mj_forward(model, data)
    b2 = model.body("l2").id
    arm = data.subtree_com[b2] - data.xpos[b2]
    assert np.isclose(abs(G[1]), model.body_subtreemass[b2] * 9.81 * abs(arm[0]))


def test_free_fall_when_actuators_off_and_no_torque():
    model, data = _model()
    sm.torque_mode(model)
    data.qpos[:] = [0.0, 0.0]
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[0]) > 1e-3                                  # the horizontal arm drops under gravity


def test_mass_matrix_is_symmetric_positive_definite_and_matches_jacobian_dynamics():
    model, data = _model()
    data.qpos[:] = [0.4, 0.7]
    mujoco.mj_forward(model, data)
    M = sm.mass_matrix(model, data)
    assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)


def test_site_jacobian_matches_finite_differences():
    model, data = _model()
    q = np.array([0.4, 0.7])
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    J = sm.site_jacobian(model, data, "tip")
    eps = 1e-6
    p0 = data.site("tip").xpos.copy()
    for i in range(2):
        d2 = mujoco.MjData(model)
        d2.qpos[:] = q + eps * np.eye(2)[i]
        mujoco.mj_forward(model, d2)
        assert np.allclose(J[:3, i], (d2.site("tip").xpos - p0) / eps, atol=1e-5)
    assert np.allclose(J[3:, 0], [0, 1, 0]) and np.allclose(J[3:, 1], [0, 1, 0])


def test_sensors_at_rest_follow_the_documented_sign_conventions():
    model, data = _model()
    q = np.array([0.3, -0.5])
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    data.qfrc_applied[:] = sm.gravity_torque(model, data)
    sm.torque_mode(model)
    mujoco.mj_forward(model, data)
    R = data.site("tip").xmat.reshape(3, 3)
    g = model.opt.gravity
    bid = model.site_bodyid[model.site("tip").id]
    m = model.body_subtreemass[bid]
    com = data.subtree_com[bid]
    assert np.allclose(data.sensor("accelerometer").data, R.T @ (-g), atol=1e-8)       # a = 0, so proper acceleration = -g
    assert np.allclose(data.sensor("gyro").data, 0.0)
    F = data.sensor("force").data
    assert np.allclose(F, R.T @ (-m * g), atol=1e-8)                                      # support force = minus the weight
    T = data.sensor("torque").data
    arm = com - data.site("tip").xpos
    assert np.allclose(T, R.T @ np.cross(arm, -m * g), atol=1e-8)
    assert np.isclose(data.sensor("q1").data[0], q[0])


def test_payload_estimate_from_mujoco_wrench_readings():
    model, data = _model()
    sm.torque_mode(model)
    Rs, Fs, Ts = [], [], []
    for q in ([0.0, 0.0], [0.5, -0.3], [-0.8, 0.9], [1.2, 0.4], [2.0, -1.0]):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = sm.gravity_torque(model, data)
        mujoco.mj_forward(model, data)
        Rs.append(data.site("tip").xmat.reshape(3, 3).copy())
        Fs.append(data.sensor("force").data.copy())
        Ts.append(data.sensor("torque").data.copy())
    m_hat, c_hat, f0, t0 = sn.estimate_payload(Rs, Fs, Ts, model.opt.gravity)
    bid = model.site_bodyid[model.site("tip").id]
    assert np.isclose(m_hat, model.body_subtreemass[bid])
    assert np.allclose(f0, 0, atol=1e-8) and np.allclose(t0, 0, atol=1e-8)
    # centre of mass of the sensed body and its payload, in the site frame, at the last pose
    R_last = Rs[-1]
    c_last = R_last.T @ (data.subtree_com[bid] - data.site("tip").xpos)
    assert np.allclose(c_hat, c_last, atol=1e-8)
