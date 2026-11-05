"""
02_planar_2r_forward_kinematics.py
==================================

Purpose
-------
This script verifies the forward kinematics of the planar two-link (2R)
manipulator, the standard first example of MFoR Chapter 3 (Craig,
Chapter 3), against the MuJoCo model ``models/planar_2r.xml``.
For a grid of joint angles (q1, q2) it compares

* MuJoCo's position of the tool site, ``data.site_xpos``, with the analytic
  result  x = L1 cos q1 + L2 cos(q1 + q2),  y = L1 sin q1 + L2 sin(q1 + q2);
* MuJoCo's orientation of the tool site, ``data.site_xmat``, with the
  rotation  Rz(q1 + q2)  predicted by the product of homogeneous transforms
  T = T_01(q1) T_12(q2) T_2T; and
* MuJoCo's translational Jacobian of the site, obtained from ``mj_jacSite``,
  with the analytic Jacobian of MFoR Chapter 5 (Craig, Chapter 5),

      J(q) = [ -L1 s1 - L2 s12   -L2 s12 ]
             [  L1 c1 + L2 c12    L2 c12 ].

Every comparison must agree to 1e-9, which in practice means agreement to
round-off.  If an offscreen OpenGL context is available, a PNG of the arm in
one configuration is written to ``outputs/02_planar_2r.png``.

Usage
-----
    python 02_planar_2r_forward_kinematics.py
    python 02_planar_2r_forward_kinematics.py --no-render
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def configure_headless_gl() -> None:
    """Select a working headless OpenGL back end for MuJoCo, if one is needed.

    MuJoCo chooses its OpenGL back end at import time from the environment
    variable MUJOCO_GL.  On a Linux machine without a display we probe "egl"
    and then "osmesa" in a child process and keep the first that works.
    """
    if "MUJOCO_GL" in os.environ or platform.system() != "Linux":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    # The probe renders one tiny image, which exercises the whole pipeline.
    probe = (
        "import mujoco\n"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/>"
        "</worldbody></mujoco>')\n"
        "d = mujoco.MjData(m)\n"
        "with mujoco.Renderer(m, 16, 16) as r:\n"
        "    r.update_scene(d)\n"
        "    r.render()\n"
    )
    for backend in ("egl", "osmesa"):
        env = dict(os.environ, MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend)
        try:
            result = subprocess.run([sys.executable, "-c", probe], env=env,
                                    capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            os.environ["MUJOCO_GL"] = backend
            os.environ["PYOPENGL_PLATFORM"] = backend
            return


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "planar_2r.xml"
OUTPUT_DIR = HERE / "outputs"

# Link lengths of the analytic model.  The script also reads them from the
# MuJoCo model and checks that the two descriptions coincide.
L1, L2 = 1.0, 0.8
TOL = 1e-9


def fk_analytic(q1: float, q2: float) -> np.ndarray:
    """Tool position (x, y) of the planar 2R arm."""
    return np.array([L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
                     L1 * np.sin(q1) + L2 * np.sin(q1 + q2)])


def jacobian_analytic(q1: float, q2: float) -> np.ndarray:
    """2x2 translational Jacobian d(x, y)/d(q1, q2) of the planar 2R arm."""
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Planar 2R forward kinematics check.")
    parser.add_argument("--no-render", action="store_true",
                        help="skip the optional PNG rendering")
    args = parser.parse_args()
    if not args.no_render:
        configure_headless_gl()

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool")
    link2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
    base_height = model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link1")][2]

    # The link lengths are encoded as the offset of body "link2" in frame {1}
    # and of the site "tool" in frame {2}.
    assert abs(model.body_pos[link2_id][0] - L1) < TOL
    assert abs(model.site_pos[site_id][0] - L2) < TOL
    print(f"MuJoCo version {mujoco.__version__}; model '{MODEL_PATH.name}'")
    print(f"nq = {model.nq}, nv = {model.nv}, nu = {model.nu}; "
          f"L1 = {L1} m, L2 = {L2} m, base height = {base_height} m\n")

    # Jacobian buffers: 3 x nv for translation and for rotation.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    grid = np.linspace(-np.pi, np.pi, 25)
    max_pos_err = max_rot_err = max_jac_err = 0.0
    for q1 in grid:
        for q2 in grid:
            data.qpos[:] = (q1, q2)
            # mj_kinematics suffices for positions; mj_comPos is additionally
            # required before mj_jac* (mj_forward would perform both).
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            p_mj = data.site_xpos[site_id]
            R_mj = data.site_xmat[site_id].reshape(3, 3)
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

            pos_err = np.max(np.abs(p_mj[:2] - fk_analytic(q1, q2)))
            z_err = abs(p_mj[2] - base_height)
            rot_err = np.max(np.abs(R_mj - rot_z(q1 + q2)))
            jac_err = np.max(np.abs(jacp[:2, :] - jacobian_analytic(q1, q2)))
            # The angular Jacobian of a planar arm: each joint spins about z.
            jacr_err = np.max(np.abs(jacr - np.array([[0, 0], [0, 0], [1, 1]])))

            assert max(pos_err, z_err) < TOL, f"position mismatch at q = {(q1, q2)}"
            assert rot_err < TOL, f"orientation mismatch at q = {(q1, q2)}"
            assert max(jac_err, jacr_err) < TOL, f"Jacobian mismatch at q = {(q1, q2)}"
            max_pos_err = max(max_pos_err, pos_err, z_err)
            max_rot_err = max(max_rot_err, rot_err)
            max_jac_err = max(max_jac_err, jac_err, jacr_err)

    n = grid.size ** 2
    print(f"Configurations tested             : {n}")
    print(f"max |site_xpos - analytic FK|     : {max_pos_err:.2e} m")
    print(f"max |site_xmat - Rz(q1 + q2)|     : {max_rot_err:.2e}")
    print(f"max |mj_jacSite - analytic J|     : {max_jac_err:.2e}")

    # A short table for a handful of configurations, for the reader.
    print("\n   q1 (deg)   q2 (deg) |  x MuJoCo   x analytic |  y MuJoCo   y analytic")
    for q1d, q2d in [(0, 0), (90, 0), (30, 60), (45, -90), (120, 45)]:
        q1, q2 = np.deg2rad([q1d, q2d])
        data.qpos[:] = (q1, q2)
        mujoco.mj_kinematics(model, data)
        x, y = data.site_xpos[site_id][:2]
        xa, ya = fk_analytic(q1, q2)
        print(f"  {q1d:8.1f} {q2d:10.1f} | {x:9.6f} {xa:11.6f} | {y:9.6f} {ya:11.6f}")
    print(f"\nAll forward-kinematics and Jacobian checks passed (tolerance {TOL:g}).")

    if not args.no_render:
        data.qpos[:] = np.deg2rad([30.0, 60.0])
        mujoco.mj_forward(model, data)
        render_png(model, data, OUTPUT_DIR / "02_planar_2r.png")


def render_png(model, data, path: Path) -> None:
    """Render the current state from the 'overview' camera; never raise on failure."""
    try:
        import mujoco
        from PIL import Image

        with mujoco.Renderer(model, height=360, width=480) as renderer:
            renderer.update_scene(data, camera="overview")
            pixels = renderer.render()
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path, optimize=True)
        print(f"Rendered image written to {path} "
              f"(MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')})")
    except Exception as exc:  # noqa: BLE001 - rendering is strictly optional
        print(f"Rendering skipped ({type(exc).__name__}: {exc}).")
        print("Set MUJOCO_GL=egl or MUJOCO_GL=osmesa on a headless machine; "
              "see docs/INSTALLATION.md.")


if __name__ == "__main__":
    main()
