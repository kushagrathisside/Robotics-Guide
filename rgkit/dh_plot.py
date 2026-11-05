"""Matplotlib 3-D figures for D-H frames: one link with its four parameters, and whole arms with their frames.

Presentation only; the mathematics lives in :mod:`rgkit.dh_tools` and :mod:`rgkit.kinematics`. Colours follow the
repository convention (X red, Y green, Z blue; grey construction lines; orange for the quantity of interest).
"""

from __future__ import annotations

import numpy as np

from .plotting import draw_frame, new_axes3d, ACCENT, GREY, AXIS_COLORS
from .transforms import rotx, rotz, make_T, transl
from .kinematics import fk_dh

__all__ = ["draw_arc3d", "draw_dh_link", "draw_dh_step", "draw_arm", "equal_limits"]

PARAM_COLORS = {"alpha": "#9467bd", "a": "#222222", "d": "#2ca02c", "theta": "#d62728"}


def draw_arc3d(ax, centre, axis, v_from, angle, radius, color="k", lw=1.6, label=None, fontsize=10, n=30):
    """Circular arc about ``axis`` through ``centre`` starting in direction ``v_from`` and turning by ``angle``."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    v = np.asarray(v_from, float)
    v = v - np.dot(v, axis) * axis
    v = v / np.linalg.norm(v)
    w = np.cross(axis, v)
    t = np.linspace(0, angle, n)
    pts = np.asarray(centre, float)[:, None] + radius * (np.outer(v, np.cos(t)) + np.outer(w, np.sin(t)))
    ax.plot(*pts, color=color, lw=lw)
    if label:
        mid = pts[:, n // 2] + 0.18 * radius * (pts[:, n // 2] - np.asarray(centre, float)) / radius
        ax.text(*mid, label, color=color, fontsize=fontsize)
    return pts


def equal_limits(ax, pts, pad=0.15):
    """Cubic axis limits that contain every point of ``pts`` (shape ``(m, 3)``)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    c = 0.5 * (pts.max(0) + pts.min(0))
    r = 0.5 * np.max(pts.max(0) - pts.min(0)) * (1 + pad) + 1e-3
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def draw_dh_link(ax, alpha, a, d, theta, length=1.0, show_intermediate=False, names=("i-1", "i"), fontsize=10,
                 frame_scale=0.38):
    """Draw the four modified D-H parameters of one link.

    Frame ``{i-1}`` is placed at the origin (axes along x, y, z). The picture shows joint axis ``i-1`` and axis ``i``
    (grey dashed), the common normal of length ``a`` along ``X_{i-1}``, the offset ``d`` along axis ``i``, the arcs of the
    twist ``alpha`` (about ``X_{i-1}``) and of the joint angle ``theta`` (about ``Z_i``), and the frame ``{i}``. With
    ``show_intermediate`` the three intermediate frames ``{R}``, ``{Q}``, ``{P}`` of the four elementary moves
    ``Rx(alpha) -> Dx(a) -> Rz(theta) -> Dz(d)`` are added (thin, labelled by name only). ``length`` scales the
    picture; the axes are switched off. Returns the 4x4 pose of ``{i}``.
    """
    L = length
    R_ = make_T(rotx(alpha))
    Q = R_ @ transl(a, 0, 0)
    P = Q @ make_T(rotz(theta))
    Fr = P @ transl(0, 0, d)
    foot = Q[:3, 3]
    z_prev = np.array([0.0, 0.0, 1.0])
    z_i = Q[:3, 2]
    ext = 1.25 * L
    for origin, direction in ((np.zeros(3), z_prev), (foot, z_i)):                 # the two joint axes
        seg = np.array([origin - 0.35 * L * direction, origin + ext * direction])
        ax.plot(*seg.T, color=GREY, ls="--", lw=1.2)
    ax.text(*(1.04 * ext * z_prev), "joint axis $i-1$", color=GREY, fontsize=fontsize)
    ax.text(*(foot + 1.04 * ext * z_i), "joint axis $i$", color=GREY, fontsize=fontsize)
    ax.plot(*np.array([np.zeros(3), foot]).T, color=PARAM_COLORS["a"], lw=3.0)      # common normal
    ax.text(*(0.5 * foot + np.array([0, 0.0, -0.1 * L])), r"$a_{i-1}$", color=PARAM_COLORS["a"], fontsize=fontsize + 2)
    ax.plot(*np.array([foot, Fr[:3, 3]]).T, color=PARAM_COLORS["d"], lw=3.0)         # offset along axis i
    ax.text(*(0.5 * (foot + Fr[:3, 3]) + 0.1 * L * np.array([1, 0, 0])), r"$d_i$", color=PARAM_COLORS["d"], fontsize=fontsize + 2)
    ref = 0.62 * L                                                                    # parallel copies at the foot
    for direction in (np.array([1.0, 0, 0]), z_prev):
        ax.plot(*np.array([foot, foot + ref * direction]).T, color=GREY, ls=":", lw=1.1)
    if abs(alpha) > 1e-6:                                                             # twist about X_{i-1}
        draw_arc3d(ax, foot, [1, 0, 0], z_prev, alpha, 0.55 * L, PARAM_COLORS["alpha"], label=r"$\alpha_{i-1}$", fontsize=fontsize + 2)
    if abs(theta) > 1e-6:                                                             # joint angle about Z_i
        draw_arc3d(ax, foot, z_i, [1, 0, 0], theta, 0.55 * L, PARAM_COLORS["theta"], label=r"$\theta_i$", fontsize=fontsize + 2)
    draw_frame(ax, np.eye(4), name=names[0], length=frame_scale * L, fontsize=fontsize)
    draw_frame(ax, Fr, name=names[1], length=frame_scale * L, fontsize=fontsize)
    if show_intermediate:
        for T, nm in ((R_, "R"), (Q, "Q"), (P, "P")):
            draw_frame(ax, T, name=nm, length=0.6 * frame_scale * L, alpha=0.55, lw=1.3, label_axes=False, fontsize=fontsize - 1)
    pts = np.array([np.zeros(3), foot, Fr[:3, 3], ext * z_prev, foot + ext * z_i, -0.35 * L * z_prev,
                    foot + 0.75 * L * np.array([1.0, 0, 0])])
    equal_limits(ax, pts, 0.1)
    ax.set_axis_off()
    return Fr


def draw_dh_step(ax, step, alpha, a, d, theta, length=1.0, fontsize=10):
    """One panel of the four elementary moves ``Rx(alpha) -> Dx(a) -> Rz(theta) -> Dz(d)`` of the modified link transform.

    ``step`` is 1 to 4: the frame ``{i-1}`` (faded) is carried through the first ``step`` moves and the moving frame
    is drawn at full strength together with the motion just performed (an arc or an arrow). Returns the pose reached.
    """
    L = length
    k = 0.45 * L
    poses = [np.eye(4)]
    poses.append(poses[-1] @ make_T(rotx(alpha)))
    poses.append(poses[-1] @ transl(a, 0, 0))
    poses.append(poses[-1] @ make_T(rotz(theta)))
    poses.append(poses[-1] @ transl(0, 0, d))
    draw_frame(ax, poses[0], name="i-1", length=k, alpha=0.35 if step > 0 else 1.0, label_axes=False, fontsize=fontsize)
    cur = poses[step]
    names = {1: "R", 2: "Q", 3: "P", 4: "i"}
    draw_frame(ax, cur, name=names[step], length=k, fontsize=fontsize)
    for j in range(1, step):                                      # earlier intermediate frames, faded
        draw_frame(ax, poses[j], length=0.7 * k, alpha=0.3, lw=1.2, label_axes=False)
    prev = poses[step - 1]
    if step == 1:
        draw_arc3d(ax, np.zeros(3), [1, 0, 0], [0, 0, 1], alpha, 0.8 * L * 0.6, PARAM_COLORS["alpha"],
                   label=r"$\alpha_{i-1}$ about $\hat X_{i-1}$", fontsize=fontsize)
    elif step == 2:
        ax.plot(*np.array([prev[:3, 3], cur[:3, 3]]).T, color=PARAM_COLORS["a"], lw=3)
        ax.text(*(0.5 * (prev[:3, 3] + cur[:3, 3]) + np.array([0, 0, -0.1 * L])), r"$a_{i-1}$ along $\hat X$", color=PARAM_COLORS["a"], fontsize=fontsize)
    elif step == 3:
        draw_arc3d(ax, prev[:3, 3], prev[:3, 2], prev[:3, 0], theta, 0.8 * L * 0.6, PARAM_COLORS["theta"],
                   label=r"$\theta_i$ about $\hat Z$", fontsize=fontsize)
    else:
        ax.plot(*np.array([prev[:3, 3], cur[:3, 3]]).T, color=PARAM_COLORS["d"], lw=3)
        ax.text(*(0.5 * (prev[:3, 3] + cur[:3, 3]) + 0.1 * L * np.array([1, 0, 0])), r"$d_i$ along $\hat Z$", color=PARAM_COLORS["d"], fontsize=fontsize)
    pts = np.array([p[:3, 3] + off for p in poses for off in (np.zeros(3), 0.5 * L * np.ones(3), -0.4 * L * np.ones(3))])
    equal_limits(ax, pts, 0.05)
    ax.set_axis_off()
    return cur


def draw_arm(ax, table, q=None, convention="modified", base=None, frame_length=0.12, names=None,
             link_color="#444444", axis_length=0.16, joint_axes=True, lw=4.0, fontsize=9, T_tool=None):
    """Draw a serial arm described by a D-H table: links, joint axes and every frame.

    ``base`` is the 4x4 pose of ``{0}`` in the drawing frame (default identity). In the modified convention the links
    follow the common normals (``a_{i-1}``) and offsets (``d_i``); in the standard convention the polyline simply
    joins the frame origins. ``T_tool`` (optional) is a fixed transform from the last frame to a tool frame that
    is drawn in orange. Returns the list of frame poses ``[{0}, ..., {N}]`` in the drawing frame.
    """
    B = np.eye(4) if base is None else np.asarray(base)
    Ts = [B @ T for T in fk_dh(table, q, convention)]
    n = len(table)
    names = names or [str(i) for i in range(n + 1)]
    pts = [Ts[0][:3, 3]]
    for i in range(1, n + 1):
        o = Ts[i][:3, 3]
        if convention == "modified":
            q_i = 0.0 if q is None else float(np.asarray(q)[i - 1])
            d_i = table[i - 1][2] + (q_i if table[i - 1][4] == "P" else 0.0)
            foot = o - d_i * Ts[i][:3, 2]
            seg = [pts[-1], foot, o] if i > 1 else [Ts[0][:3, 3], foot, o]
            ax.plot(*np.array(seg).T, color=link_color, lw=lw, solid_capstyle="round", alpha=0.75)
            pts.extend([foot, o])
        else:
            ax.plot(*np.array([Ts[i - 1][:3, 3], o]).T, color=link_color, lw=lw, solid_capstyle="round", alpha=0.75)
            pts.append(o)
    if joint_axes:
        for i in range(1, n + 1):
            T = Ts[i] if convention == "modified" else Ts[i - 1]
            seg = np.array([T[:3, 3] - axis_length * T[:3, 2], T[:3, 3] + axis_length * T[:3, 2]])
            ax.plot(*seg.T, color=GREY, ls=":", lw=1.4)
    for i, T in enumerate(Ts):
        draw_frame(ax, T, name=names[i], length=frame_length, lw=1.8, fontsize=fontsize)
    if T_tool is not None:
        Tt = Ts[-1] @ T_tool
        ax.plot(*np.array([Ts[-1][:3, 3], Tt[:3, 3]]).T, color=ACCENT, lw=2.5)
        draw_frame(ax, Tt, name="T", length=frame_length, lw=1.8, fontsize=fontsize)
        pts.append(Tt[:3, 3])
    ax._dh_points = np.array(pts)
    return Ts
