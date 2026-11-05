"""Small helpers for using MuJoCo inside Jupyter notebooks.

MuJoCo (https://mujoco.org) is the physics simulator used throughout the
repository. In the notebooks we build models from short MJCF (XML) strings
so that every line of the model is visible to the reader, step or pose the
model, and render images offscreen with :class:`mujoco.Renderer`.

Offscreen rendering needs an OpenGL backend chosen *before* ``mujoco`` is
imported, through the ``MUJOCO_GL`` environment variable:

* Google Colab / machines with a GPU: ``egl``
* headless Linux without a GPU: ``osmesa`` (install ``libosmesa6``)
* desktops with a display (Windows, macOS, Linux): leave it unset (``glfw``)

:func:`configure_gl` makes a sensible choice automatically; call it (or
import this module) before ``import mujoco``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

__all__ = ["configure_gl", "load", "render", "frame_axes_geoms", "set_free_pose",
           "available", "scene_assets", "floor"]

_PROBE = (
    "import mujoco\n"
    "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>')\n"
    "mujoco.Renderer(m,8,8)\n"
)


def configure_gl() -> str:
    """Choose a MuJoCo OpenGL backend if the user has not set one; return it."""
    if "MUJOCO_GL" in os.environ:
        return os.environ["MUJOCO_GL"]
    if "google.colab" in sys.modules:
        backend = "egl"
    elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        backend = "osmesa"
        for candidate in ("egl", "osmesa"):
            env = dict(os.environ, MUJOCO_GL=candidate, PYOPENGL_PLATFORM=candidate)
            try:
                ok = subprocess.run([sys.executable, "-c", _PROBE], env=env, timeout=60,
                                    capture_output=True).returncode == 0
            except Exception:  # noqa: BLE001 - any failure means "try the next one"
                ok = False
            if ok:
                backend = candidate
                break
    else:
        return "glfw"
    os.environ["MUJOCO_GL"] = backend
    os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    return backend


configure_gl()

try:
    import mujoco  # noqa: E402
except ImportError:  # pragma: no cover - MuJoCo is an optional dependency
    mujoco = None


def available() -> bool:
    """True if the ``mujoco`` package can be imported."""
    return mujoco is not None


def load(xml: str):
    """Compile an MJCF string; return ``(model, data)`` after one forward pass."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


_renderers: dict = {}


def render(model, data, camera=-1, width=480, height=360, scene_option=None):
    """Render the current state of ``data`` to an RGB ``numpy`` array.

    ``camera`` is a camera name or id defined in the model (``-1`` is the
    default free camera), or a dict such as
    ``dict(azimuth=135, elevation=-25, distance=3.0, lookat=(0, 0, 0))``
    describing a free (orbiting) camera. Renderers are cached per (model, size).
    """
    key = (id(model), width, height)
    if key not in _renderers:
        _renderers[key] = mujoco.Renderer(model, height=height, width=width)
    r = _renderers[key]
    if isinstance(camera, dict):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        for k, v in camera.items():
            if k == "lookat":
                cam.lookat[:] = v
            else:
                setattr(cam, k, v)
        camera = cam
    r.update_scene(data, camera=camera, scene_option=scene_option)
    return r.render().copy()


def frame_axes_geoms(length=0.3, radius=0.012, alpha=1.0, prefix="") -> str:
    """MJCF snippet: three capsules along the body's X (red), Y (green), Z (blue).

    Paste it inside a ``<body>`` to make that body's frame visible.
    """
    L = length
    return (
        f'<geom name="{prefix}x" type="capsule" fromto="0 0 0 {L} 0 0" size="{radius}" '
        f'rgba="0.84 0.15 0.16 {alpha}" contype="0" conaffinity="0"/>\n'
        f'<geom name="{prefix}y" type="capsule" fromto="0 0 0 0 {L} 0" size="{radius}" '
        f'rgba="0.17 0.63 0.17 {alpha}" contype="0" conaffinity="0"/>\n'
        f'<geom name="{prefix}z" type="capsule" fromto="0 0 0 0 0 {L}" size="{radius}" '
        f'rgba="0.12 0.47 0.71 {alpha}" contype="0" conaffinity="0"/>\n'
    )


def set_free_pose(model, data, joint, position, quat_wxyz):
    """Set the pose of a body attached to the world by a ``freejoint``.

    MuJoCo stores a free joint as 7 numbers: position (3) then a unit
    quaternion in scalar-first order ``[w, x, y, z]`` — the same convention
    used in these notes.
    """
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
    data.qpos[adr:adr + 3] = position
    data.qpos[adr + 3:adr + 7] = np.asarray(quat_wxyz) / np.linalg.norm(quat_wxyz)
    mujoco.mj_forward(model, data)


def scene_assets(offwidth=800, offheight=600) -> str:
    """MJCF snippet (place directly under ``<mujoco>``): sky, floor grid and render size.

    Gives every notebook render the same soft gradient background and a
    chequered floor, so figures look consistent across chapters.
    """
    return f"""
  <visual>
    <global offwidth="{offwidth}" offheight="{offheight}"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.05 0.05 0.05"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.96 0.97 1.0" rgb2="0.78 0.83 0.90"
             width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.94 0.94 0.94" rgb2="0.86 0.86 0.88"
             width="512" height="512" mark="edge" markrgb="0.75 0.75 0.78"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>"""


def floor(z=0.0, size=2.0) -> str:
    """MJCF snippet (place inside ``<worldbody>``): a light and a chequered floor at height ``z``."""
    return (f'<light pos="0.6 -0.6 3" dir="-0.2 0.2 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>\n'
            f'<geom name="floor" type="plane" pos="0 0 {z}" size="{size} {size} 0.01" material="grid"/>')
