"""
01_frames_and_quaternions.py
============================

Purpose
-------
This script connects the rotation matrices of MFoR Chapter 2 (Craig,
*Introduction to Robotics*, Chapter 2) with the way MuJoCo stores
and reports orientation.  A free-floating body, whose geometry is a set of
red, green and blue arrows, represents a coordinate frame {B}.  For several
sets of Z-Y-X Euler angles (alpha, beta, gamma) the script

1. builds the rotation matrix  R = Rz(alpha) Ry(beta) Rx(gamma)  with NumPy,
   as in Craig's treatment of Z-Y-X Euler angles;
2. converts R to a unit quaternion and writes it into the free joint's
   configuration ``qpos = [px, py, pz, qw, qx, qy, qz]``;
3. calls ``mj_forward`` so that MuJoCo computes the kinematics;
4. reads back ``data.xmat`` (a row-major 3x3 rotation matrix) and
   ``data.xquat`` (a quaternion, scalar first) for the body; and
5. asserts that MuJoCo's results agree with the hand computation, and that
   MuJoCo's own Euler-angle routine ``mju_euler2Quat`` produces the same
   orientation.

If an offscreen OpenGL context can be created, a PNG of the final frame is
written to ``outputs/01_frames_demo.png``.  Rendering is optional: the
numerical checks are the substance of the script.

Usage
-----
    python 01_frames_and_quaternions.py            # checks and, if possible, a PNG
    python 01_frames_and_quaternions.py --no-render
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Offscreen rendering set-up.  MuJoCo chooses its OpenGL back end when it is
# first imported, according to the environment variable MUJOCO_GL.  On a
# Linux machine without a display (a server, a container, a CI runner) the
# default back end (GLFW) cannot open a window, so we probe the headless back
# ends "egl" (GPU) and "osmesa" (software) in a child process and select the
# first that works.  This must happen *before* ``import mujoco``.
# ---------------------------------------------------------------------------
def configure_headless_gl() -> None:
    """Select a working headless OpenGL back end for MuJoCo, if one is needed."""
    if "MUJOCO_GL" in os.environ or platform.system() != "Linux":
        return  # The user has chosen, or the platform default is adequate.
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return  # A display server exists; GLFW will work.
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
MODEL_PATH = HERE / "models" / "frames_demo.xml"
OUTPUT_DIR = HERE / "outputs"


# ---------------------------------------------------------------------------
# Elementary rotations about the principal axes, and quaternion algebra.
# ---------------------------------------------------------------------------
def rot_x(theta: float):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(theta: float):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(theta: float):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_zyx_to_matrix(alpha: float, beta: float, gamma: float):
    """Z-Y-X Euler angles (rotations about the moving axes) -> rotation matrix.

    The frame is rotated first by alpha about Z, then by beta about the new Y,
    then by gamma about the newest X, so that R = Rz(alpha) Ry(beta) Rx(gamma).
    """
    return rot_z(alpha) @ rot_y(beta) @ rot_x(gamma)


def quat_multiply(p, q):
    """Hamilton product p * q of quaternions stored as [w, x, y, z]."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def quat_about_axis(axis, theta: float):
    """Unit quaternion of a rotation by theta about a unit axis: [cos(t/2), sin(t/2) k]."""
    axis = np.asarray(axis, dtype=float)
    return np.concatenate(([np.cos(theta / 2.0)], np.sin(theta / 2.0) * axis))


def same_rotation(q1, q2, tol: float) -> bool:
    """Quaternions q and -q describe the same rotation (the double cover of SO(3))."""
    return min(np.max(np.abs(q1 - q2)), np.max(np.abs(q1 + q2))) < tol


# ---------------------------------------------------------------------------
# Main programme.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-render", action="store_true",
                        help="skip the optional PNG rendering")
    args = parser.parse_args()

    if not args.no_render:
        configure_headless_gl()

    import mujoco

    np.set_printoptions(precision=6, suppress=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")

    # The free joint is the only joint, so its 7 coordinates start at qpos[0].
    qpos_adr = model.jnt_qposadr[model.body_jntadr[body_id]]
    position = np.array([0.0, 0.0, 0.6])

    test_angles_deg = [
        (0.0, 0.0, 0.0),
        (90.0, 0.0, 0.0),
        (0.0, 90.0, 0.0),
        (30.0, -45.0, 60.0),
        (-120.0, 15.0, 170.0),
        (45.0, 30.0, 20.0),   # the configuration that is rendered at the end
    ]

    tol = 1e-12
    print(f"MuJoCo version {mujoco.__version__}; model '{MODEL_PATH.name}'\n")
    for angles_deg in test_angles_deg:
        alpha, beta, gamma = np.deg2rad(angles_deg)

        # (1) Hand computation with NumPy.
        R_numpy = euler_zyx_to_matrix(alpha, beta, gamma)

        # (2) Rotation matrix -> quaternion (MuJoCo utility), then into qpos.
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_numpy.flatten())  # expects row-major order
        data.qpos[qpos_adr:qpos_adr + 3] = position
        data.qpos[qpos_adr + 3:qpos_adr + 7] = quat

        # (3) Forward kinematics: populates xpos, xmat, xquat for every body.
        mujoco.mj_forward(model, data)

        # (4) Read back MuJoCo's view of frame {B} relative to the world {0}.
        R_mujoco = data.xmat[body_id].reshape(3, 3)
        q_mujoco = data.xquat[body_id].copy()
        p_mujoco = data.xpos[body_id].copy()

        # Independent quaternion: q = q_z(alpha) * q_y(beta) * q_x(gamma).
        q_numpy = quat_multiply(quat_multiply(quat_about_axis([0, 0, 1], alpha),
                                              quat_about_axis([0, 1, 0], beta)),
                                quat_about_axis([1, 0, 0], gamma))

        # MuJoCo's own Euler routine.  Its docstring states that lower-case
        # letters denote intrinsic rotations (about the moving axes), so the
        # Z-Y-X convention of Craig is the sequence "zyx".  (Note that SciPy
        # uses the opposite case convention.)
        q_euler = np.zeros(4)
        mujoco.mju_euler2Quat(q_euler, np.array([alpha, beta, gamma]), "zyx")

        # (5) Assertions.
        assert np.allclose(R_mujoco, R_numpy, atol=tol), "xmat disagrees with NumPy"
        assert same_rotation(q_mujoco, q_numpy, tol), "xquat disagrees with NumPy"
        assert same_rotation(q_euler, q_numpy, tol), "mju_euler2Quat disagrees"
        assert np.allclose(p_mujoco, position, atol=tol), "xpos disagrees"
        assert np.allclose(R_mujoco.T @ R_mujoco, np.eye(3), atol=tol), "R not orthonormal"
        assert abs(np.linalg.det(R_mujoco) - 1.0) < tol, "det(R) is not +1"

        print(f"Z-Y-X Euler angles (deg) = {angles_deg}")
        print("  R from NumPy       =\n" + _indent(R_numpy))
        print("  data.xmat (3x3)    =\n" + _indent(R_mujoco))
        print(f"  data.xquat [w x y z] = {q_mujoco}")
        print(f"  max |xmat - R|       = {np.max(np.abs(R_mujoco - R_numpy)):.2e}\n")

    # The columns of R are the unit vectors of {B} expressed in {0}: the tip
    # of the red arrow (length 0.4 m) must lie at p + 0.4 * R[:, 0].
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "x_head")
    expected_tip = position + 0.4 * R_mujoco[:, 0]
    assert np.allclose(data.geom_xpos[geom_id], expected_tip, atol=1e-12)
    print("The tip of the x-axis arrow lies at p + 0.4 R[:, 0], as expected.")
    print("All orientation checks passed.")

    if not args.no_render:
        render_png(model, data, OUTPUT_DIR / "01_frames_demo.png")


def _indent(matrix) -> str:
    return "\n".join("      " + line for line in str(matrix).splitlines())


def render_png(model, data, path: Path) -> None:
    """Render the current state from the 'iso' camera; never raise on failure."""
    try:
        import mujoco
        from PIL import Image

        with mujoco.Renderer(model, height=360, width=480) as renderer:
            renderer.update_scene(data, camera="iso")
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
