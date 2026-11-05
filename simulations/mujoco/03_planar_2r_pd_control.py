"""
03_planar_2r_pd_control.py
==========================

Purpose
-------
This script demonstrates independent-joint PD set-point control (RMC, the
chapter on PD and PID control; Craig, Chapter 9) on the planar 2R arm of
``models/planar_2r.xml``.  The arm starts at rest at q = (0, 0) and is
commanded to the set-point q_d = (60 deg, -45 deg).  The same control law,

    tau = Kp (q_d - q) - Kd qdot,

is realised in two ways, and the two simulations are compared:

* **Run A** uses MuJoCo's built-in ``<position>`` actuators, whose force is
  ``kp (ctrl - q) - kv qdot``; we simply set ``data.ctrl = q_d``.
* **Run B** disables the actuators and computes the PD torque in Python at
  every step, writing it into ``data.qfrc_applied`` (generalised forces
  applied directly to the joints).  This is the pattern used for every
  model-based controller in RMC, such as computed torque.

Because the arm moves in a horizontal plane, gravity produces no joint
torque, and PD control therefore drives the steady-state error to zero.
(Were the arm vertical, a constant gravity torque would leave a steady-state
error; this motivates gravity compensation and integral action in RMC.)

Outputs
-------
* ``outputs/03_pd_joint_angles.png`` : joint angles and torques against time.
* ``outputs/03_pd_motion.gif``       : a short animation, written only if an
  offscreen OpenGL context is available.

Usage
-----
    python 03_planar_2r_pd_control.py
    python 03_planar_2r_pd_control.py --no-render
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

Q_DESIRED = np.deg2rad([60.0, -45.0])   # set-point (rad)
DURATION = 3.0                          # simulated time (s)


def simulate(model, data, use_builtin_actuators: bool):
    """Simulate one PD set-point response and return (t, q, qdot, tau) arrays."""
    import mujoco

    # Gains of the position servos, read from the model so that Run B uses
    # exactly the same numbers.  For a <position> actuator MuJoCo stores
    # gainprm[0] = kp and biasprm[:3] = (0, -kp, -kv).
    kp = model.actuator_gainprm[:, 0].copy()
    kv = -model.actuator_biasprm[:, 2].copy()

    mujoco.mj_resetData(model, data)           # q = 0, qdot = 0, t = 0
    if use_builtin_actuators:
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        data.ctrl[:] = Q_DESIRED
    else:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)

    n_steps = int(round(DURATION / model.opt.timestep))
    t = np.zeros(n_steps + 1)
    q = np.zeros((n_steps + 1, model.nq))
    qdot = np.zeros((n_steps + 1, model.nv))
    tau = np.zeros((n_steps + 1, model.nv))

    for k in range(n_steps + 1):
        if use_builtin_actuators:
            mujoco.mj_forward(model, data)     # computes data.qfrc_actuator
            tau[k] = data.qfrc_actuator
        else:
            tau[k] = kp * (Q_DESIRED - data.qpos) - kv * data.qvel
            data.qfrc_applied[:] = tau[k]
        t[k], q[k], qdot[k] = data.time, data.qpos, data.qvel
        if k < n_steps:
            mujoco.mj_step(model, data)

    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    data.qfrc_applied[:] = 0.0
    return t, q, qdot, tau


def main() -> None:
    parser = argparse.ArgumentParser(description="PD set-point control of the planar 2R arm.")
    parser.add_argument("--no-render", action="store_true",
                        help="skip the optional GIF rendering")
    args = parser.parse_args()
    if not args.no_render:
        configure_headless_gl()

    import mujoco
    import matplotlib
    matplotlib.use("Agg")                 # file output only; no window needed
    import matplotlib.pyplot as plt

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    print(f"MuJoCo version {mujoco.__version__}; model '{MODEL_PATH.name}'")
    print(f"time step = {model.opt.timestep} s, duration = {DURATION} s")
    print(f"kp = {model.actuator_gainprm[:, 0]}, kv = {-model.actuator_biasprm[:, 2]}")

    t, qA, qdA, tauA = simulate(model, data, use_builtin_actuators=True)
    _, qB, qdB, tauB = simulate(model, data, use_builtin_actuators=False)

    final_error = np.abs(qA[-1] - Q_DESIRED)
    run_gap = np.max(np.abs(qA - qB))
    overshoot = np.max((qA - qA[0]) / (Q_DESIRED - qA[0]), axis=0) - 1.0
    print(f"final |q - q_d| (rad)            : {final_error}")
    print(f"final |qdot| (rad/s)             : {np.abs(qdA[-1])}")
    print(f"percentage overshoot (joint 1, 2): {np.round(100 * np.maximum(overshoot, 0), 2)}")
    print(f"max |q_A - q_B| over the run     : {run_gap:.2e} rad")

    assert np.all(final_error < 1e-3), "PD control failed to reach the set-point"
    assert np.all(np.abs(qdA[-1]) < 1e-2), "the arm has not come to rest"
    assert run_gap < 1e-9, "built-in servo and explicit PD law disagree"
    print("PD set-point checks passed; Run A and Run B coincide.")

    # ------------------------------------------------------------------ plot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_q, ax_tau) = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True,
                                       constrained_layout=True)
    colours = ("#1f77b4", "#ff7f0e")
    for j in range(2):
        ax_q.plot(t, np.rad2deg(qA[:, j]), color=colours[j], lw=2,
                  label=f"$q_{j + 1}$ (MuJoCo servo)")
        ax_q.plot(t[::100], np.rad2deg(qB[::100, j]), "o", color=colours[j], ms=3.5,
                  mfc="none", label=f"$q_{j + 1}$ (explicit PD law)")
        ax_q.axhline(np.rad2deg(Q_DESIRED[j]), color=colours[j], ls="--", lw=1,
                     label=f"$q_{{d,{j + 1}}}$")
        ax_tau.plot(t, tauA[:, j], color=colours[j], lw=1.6, label=rf"$\tau_{j + 1}$")
    ax_q.set_ylabel("joint angle (deg)")
    ax_q.set_title("Planar 2R arm: PD set-point response in MuJoCo")
    ax_q.legend(ncol=3, fontsize=8, loc="center right")
    ax_q.grid(alpha=0.3)
    ax_tau.set_xlabel("time (s)")
    ax_tau.set_ylabel("joint torque (N m)")
    ax_tau.legend(fontsize=8)
    ax_tau.grid(alpha=0.3)
    plot_path = OUTPUT_DIR / "03_pd_joint_angles.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"Plot written to {plot_path}")

    if not args.no_render:
        render_gif(model, data, qA, t, OUTPUT_DIR / "03_pd_motion.gif")


def render_gif(model, data, q_history, t, path: Path, fps: int = 15,
               seconds: float = 2.0) -> None:
    """Replay the recorded joint trajectory as a small GIF; never raise on failure."""
    try:
        import mujoco
        from PIL import Image

        frames = []
        frame_times = np.arange(0.0, min(seconds, t[-1]) + 1e-9, 1.0 / fps)
        with mujoco.Renderer(model, height=240, width=320) as renderer:
            for tf in frame_times:
                k = int(np.searchsorted(t, tf))
                data.qpos[:] = q_history[k]
                mujoco.mj_forward(model, data)       # kinematics only; no stepping
                renderer.update_scene(data, camera="overview")
                frames.append(Image.fromarray(renderer.render()))
        # A 128-colour palette per frame keeps the file small (a few hundred kB).
        frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
        print(f"Animation written to {path} ({path.stat().st_size / 1024:.0f} kB, "
              f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')})")
    except Exception as exc:  # noqa: BLE001 - rendering is strictly optional
        print(f"Rendering skipped ({type(exc).__name__}: {exc}).")
        print("Set MUJOCO_GL=egl or MUJOCO_GL=osmesa on a headless machine; "
              "see docs/INSTALLATION.md.")


if __name__ == "__main__":
    main()
