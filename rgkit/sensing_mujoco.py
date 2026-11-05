"""MuJoCo helpers for the sensing and visual-servoing notebooks (Chapter 8 of RMC).

MuJoCo is optional for the rest of ``rgkit``; this module imports it lazily through
:mod:`rgkit.mujoco_utils` (which also selects an OpenGL back end).

The helpers use MuJoCo's programmatic model editor, :class:`mujoco.MjSpec`, to add sensors, cameras, sites and
payload bodies *to the unmodified manufacturer model* of a Menagerie robot, so that notebooks never need a
modified copy of the robot.

* :func:`robot_spec` — an ``MjSpec`` containing a Menagerie robot, a floor and extra world-body XML.
* :func:`add_sensor` — attach a built-in MuJoCo sensor (``jointpos``, ``jointvel``, ``accelerometer``, ``gyro``,
  ``force``, ``torque``, …) to a site or joint.
* :func:`compile_spec` — compile and put the model in its first keyframe.
* :func:`torque_mode`, :func:`gravity_torque`, :func:`mass_matrix`, :func:`site_jacobian` — what a torque-level
  controller needs.

Sign conventions of the MuJoCo sensors that matter in the notebooks (verified in ``tests/test_sensing_mujoco.py``):

* ``accelerometer`` returns the *proper acceleration* ``R^T (a - g)`` of the site (the reaction to gravity points
  *up* for a body at rest);
* ``force`` and ``torque`` return the wrench that the *parent* body exerts on the sensor's body (and everything
  attached beyond it) at the site, expressed in the site frame: ``F = m R^T (a - g)`` for a rigid payload;
* ``gyro`` returns the angular velocity of the site in its own frame.

MuJoCo's ``noise`` attribute is *not* applied by the engine; notebooks inject noise themselves.
"""

from __future__ import annotations

import numpy as np

from . import mujoco_utils as mju

mujoco = mju.mujoco

__all__ = ["robot_spec", "spec_from_xml", "add_sensor", "compile_spec", "torque_mode", "gravity_torque",
           "mass_matrix", "site_jacobian", "SENSOR_KINDS"]


def _sensor_kinds():
    S = mujoco.mjtSensor
    return {"jointpos": S.mjSENS_JOINTPOS, "jointvel": S.mjSENS_JOINTVEL, "accelerometer": S.mjSENS_ACCELEROMETER,
            "gyro": S.mjSENS_GYRO, "force": S.mjSENS_FORCE, "torque": S.mjSENS_TORQUE,
            "framepos": S.mjSENS_FRAMEPOS, "framequat": S.mjSENS_FRAMEQUAT, "velocimeter": S.mjSENS_VELOCIMETER,
            "actuatorfrc": S.mjSENS_ACTUATORFRC, "touch": S.mjSENS_TOUCH}


SENSOR_KINDS = _sensor_kinds() if mujoco is not None else {}


def robot_spec(name: str, worldbody: str = "", tag: str = "rmc8", floor_z: float = 0.0):
    """``MjSpec`` of the Menagerie robot ``name`` (see :data:`rgkit.robots.ROBOTS`) in this repository's light scene.

    ``worldbody`` is extra MJCF placed inside ``<worldbody>`` (targets, tables, …). The wrapper file is written
    next to the downloaded model so that relative mesh paths resolve, like :func:`rgkit.robots.load_robot`.
    """
    from . import robots

    d = robots.fetch(name)
    wrapper = d / f"rg_{tag}_{name}.xml"
    wrapper.write_text(
        f'<mujoco model="{name}-{tag}">\n  <include file="{robots.ROBOTS[name]["robot"]}"/>\n'
        f'  {mju.scene_assets(offwidth=800, offheight=600)}\n  <worldbody>\n    {mju.floor(z=floor_z, size=3)}\n'
        f'    {worldbody}\n  </worldbody>\n</mujoco>\n')
    return mujoco.MjSpec.from_file(str(wrapper))


def spec_from_xml(xml: str):
    """``MjSpec`` from an MJCF string (used by the tests and by notebooks that build their own model)."""
    return mujoco.MjSpec.from_string(xml)


def add_sensor(spec, name: str, kind: str, obj: str, objtype: str = "site"):
    """Add the built-in sensor ``kind`` named ``name`` to the site/joint/body called ``obj``."""
    types = {"site": mujoco.mjtObj.mjOBJ_SITE, "joint": mujoco.mjtObj.mjOBJ_JOINT, "body": mujoco.mjtObj.mjOBJ_BODY,
             "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR}
    return spec.add_sensor(name=name, type=SENSOR_KINDS[kind], objtype=types[objtype], objname=obj)


def compile_spec(spec, keyframe: int | None = 0):
    """Compile ``spec``; return ``(model, data)``, reset to ``keyframe`` (if the model has one) and run forward."""
    model = spec.compile()
    data = mujoco.MjData(model)
    if keyframe is not None and model.nkey > keyframe:
        mujoco.mj_resetDataKeyframe(model, data, keyframe)
    mujoco.mj_forward(model, data)
    return model, data


def torque_mode(model):
    """Disable the model's own (position) actuators so that ``data.qfrc_applied`` is the only joint input."""
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION


def gravity_torque(model, data, scratch=None) -> np.ndarray:
    """Joint torques that hold the current configuration at rest, ``G(q)`` (computed with zero velocity).

    Pass a pre-allocated ``scratch = mujoco.MjData(model)`` when calling this inside a control loop.
    """
    tmp = scratch if scratch is not None else mujoco.MjData(model)
    tmp.qpos[:] = data.qpos
    tmp.qvel[:] = 0.0
    mujoco.mj_forward(model, tmp)
    return tmp.qfrc_bias.copy()


def mass_matrix(model, data) -> np.ndarray:
    """Joint-space mass matrix ``M(q)`` (including the ``armature`` inertia the model adds to each joint)."""
    M = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, M)                     # MuJoCo >= 3.3
    except TypeError:
        mujoco.mj_fullM(model, M, data.qM)
    return M


def site_jacobian(model, data, site: str) -> np.ndarray:
    """6 x nv Jacobian ``[J_v; J_w]`` of a site, expressed in the world frame (rows 0-2 linear, 3-5 angular)."""
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, model.site(site).id)
    return np.vstack([jacp, jacr])
