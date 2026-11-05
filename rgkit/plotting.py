"""Drawing helpers for frames, vectors and rigid bodies (matplotlib).

These are presentation utilities only; none of the robotics mathematics
lives here. Colour convention used throughout the repository:
X axis = red, Y axis = green, Z axis = blue ("RGB = XYZ").
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

AXIS_COLORS = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
ACCENT = "#ff7f0e"
GREY = "#7f7f7f"
PURPLE = "#9467bd"

__all__ = [
    "AXIS_COLORS", "ACCENT", "GREY", "PURPLE",
    "new_axes3d", "set_axes_equal", "draw_frame", "draw_vector", "draw_point",
    "draw_box", "draw_frame_2d", "draw_vector_2d", "matrix_to_latex",
]


def new_axes3d(ax=None, lim=1.2, title=None, elev=22, azim=35, figsize=(5.2, 5.2),
               ticks=True):
    """Create (or reuse) a 3-D axis with equal scaling and a clean look."""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    if np.size(lim) == 1:                      # symmetric cube [-lim, lim]^3
        lim = [(-float(lim), float(lim))] * 3
    (x0, x1), (y0, y1), (z0, z1) = lim         # or explicit ((x0,x1),(y0,y1),(z0,z1))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if not ticks:
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.04)
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def set_axes_equal(ax):
    """Give a 3-D axis equal scale on x, y and z (matplotlib does not by default)."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centre = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d(centre[0] - radius, centre[0] + radius)
    ax.set_ylim3d(centre[1] - radius, centre[1] + radius)
    ax.set_zlim3d(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _split_pose(pose, origin):
    pose = np.asarray(pose, dtype=float)
    if pose.shape == (4, 4):
        return pose[:3, :3], pose[:3, 3]
    return pose, np.zeros(3) if origin is None else np.asarray(origin, dtype=float)


def draw_frame(ax, pose=np.eye(3), origin=None, name=None, length=1.0, lw=2.2,
               alpha=1.0, ls="-", label_axes=True, fontsize=10, colors=None):
    """Draw a coordinate frame as three arrows.

    ``pose`` may be a 3x3 rotation (with optional ``origin``) or a 4x4
    homogeneous transform. Axis labels read e.g. ``X_B`` when ``name='B'``.
    """
    R, o = _split_pose(pose, origin)
    colors = colors or AXIS_COLORS
    artists = []
    for i, axis in enumerate("xyz"):
        d = R[:, i] * length
        artists.append(ax.quiver(*o, *d, color=colors[axis], lw=lw, alpha=alpha,
                                 arrow_length_ratio=0.12, linestyle=ls))
        if label_axes:
            sub = f"_{{{name}}}" if name else ""
            artists.append(ax.text(*(o + 1.1 * d), rf"$\hat{{{axis.upper()}}}{sub}$",
                                   color=colors[axis], fontsize=fontsize, alpha=alpha))
    if name:
        artists.append(ax.text(*(o - 0.12 * length * np.ones(3)), rf"$\{{{name}\}}$",
                               fontsize=fontsize + 1, color="k", alpha=alpha))
    return artists


def draw_vector(ax, start, vec, color=ACCENT, label=None, lw=2.0, alpha=1.0, ls="-",
                fontsize=10):
    """Draw an arrow from ``start`` along ``vec``."""
    start = np.asarray(start, dtype=float); vec = np.asarray(vec, dtype=float)
    artists = [ax.quiver(*start, *vec, color=color, lw=lw, alpha=alpha,
                         arrow_length_ratio=0.1, linestyle=ls)]
    if label:
        artists.append(ax.text(*(start + 1.05 * vec), label, color=color, fontsize=fontsize))
    return artists


def draw_point(ax, p, color="k", label=None, size=30, fontsize=10):
    """Mark a point with an optional text label."""
    p = np.asarray(p, dtype=float)
    artists = [ax.scatter(*p, color=color, s=size, depthshade=False)]
    if label:
        artists.append(ax.text(*(p + 0.05), label, color=color, fontsize=fontsize))
    return artists


def draw_box(ax, pose=np.eye(4), size=(0.6, 0.3, 0.12), color="#9ecae1", alpha=0.55,
             edgecolor="#3182bd"):
    """Draw a solid box (a stand-in rigid body) at the given pose."""
    R, o = _split_pose(pose, None)
    sx, sy, sz = np.asarray(size) / 2.0
    corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    pts = corners @ R.T + o
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    poly = Poly3DCollection([pts[f] for f in faces], facecolor=color, edgecolor=edgecolor,
                            alpha=alpha, linewidths=0.8)
    ax.add_collection3d(poly)
    return [poly]


def draw_frame_2d(ax, theta=0.0, origin=(0.0, 0.0), name=None, length=1.0, alpha=1.0,
                  lw=2.0, ls="-", fontsize=11):
    """Draw a planar frame rotated by ``theta`` (radians) about the out-of-page Z axis."""
    o = np.asarray(origin, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    artists = []
    for axis, d in (("x", np.array([c, s])), ("y", np.array([-s, c]))):
        artists.append(ax.annotate("", xy=o + length * d, xytext=o,
                                   arrowprops=dict(arrowstyle="-|>", color=AXIS_COLORS[axis],
                                                   lw=lw, alpha=alpha, linestyle=ls)))
        sub = f"_{{{name}}}" if name else ""
        artists.append(ax.text(*(o + 1.12 * length * d), rf"$\hat{{{axis.upper()}}}{sub}$",
                               color=AXIS_COLORS[axis], fontsize=fontsize, alpha=alpha,
                               ha="center", va="center"))
    return artists


def draw_vector_2d(ax, start, vec, color=ACCENT, label=None, lw=2.0, ls="-", fontsize=11):
    """Planar arrow from ``start`` along ``vec``."""
    start = np.asarray(start, dtype=float); vec = np.asarray(vec, dtype=float)
    artists = [ax.annotate("", xy=start + vec, xytext=start,
                           arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls))]
    if label:
        artists.append(ax.text(*(start + 0.55 * vec + np.array([0.04, 0.04])), label,
                               color=color, fontsize=fontsize))
    return artists


def matrix_to_latex(M, precision=3, name=None):
    """Return a LaTeX ``bmatrix`` string for display with IPython.display.Math."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    rows = []
    for row in M:
        cells = []
        for v in row:
            v = 0.0 if abs(v) < 0.5 * 10 ** (-precision) else v
            cells.append(f"{v:.{precision}f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v)}")
        rows.append(" & ".join(cells))
    body = r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"
    return f"{name} = {body}" if name else body
