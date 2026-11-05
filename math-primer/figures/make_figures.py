#!/usr/bin/env python3
"""Generate every figure used in the Mathematics Primer (math-primer/M*.md).

Usage (from any directory):

    python math-primer/figures/make_figures.py

Requires numpy, scipy and matplotlib. Figures are written next to this script
as PNG files. Style conventions shared by all figures:

* white background, thin lines, recessive grid;
* coordinate axes: x red (#d62728), y green (#2ca02c), z blue (#1f77b4),
  always also labelled with text so that colour is never the only cue;
* other data series use, in this fixed order, SERIES[0..3] together with
  distinct line styles and direct labels.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from scipy import signal  # noqa: E402

OUT = Path(__file__).resolve().parent
DPI = 120

X_RED, Y_GREEN, Z_BLUE = "#d62728", "#2ca02c", "#1f77b4"
SERIES = ["#1f77b4", "#c86400", "#7b4fbf", "#0e9f8f"]
STYLES = ["-", "--", "-.", ":"]
INK, MUTED = "#222222", "#666666"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "mathtext.fontset": "dejavusans",
    }
)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def arrow2d(ax, start, end, color, lw=2.0, ls="-"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                        shrinkA=0, shrinkB=0, mutation_scale=14),
    )


class Arrow3D(FancyArrowPatch):
    """A 3-D arrow for mplot3d axes."""

    def __init__(self, start, end, **kw):
        super().__init__((0, 0), (0, 0), **kw)
        self._xyz = (start, end)

    def do_3d_projection(self, renderer=None):
        (x0, y0, z0), (x1, y1, z1) = self._xyz
        xs, ys, zs = proj3d.proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------
def fig_m1_projection():
    a = np.array([1.0, 2.2])
    b = np.array([3.0, 0.8])
    proj = (a @ b) / (b @ b) * b
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    arrow2d(ax, (0, 0), b * 1.15, MUTED, lw=1.4)
    arrow2d(ax, (0, 0), a, SERIES[0])
    arrow2d(ax, (0, 0), proj, SERIES[1], lw=2.6)
    arrow2d(ax, proj, a, SERIES[2], ls="--")
    # right-angle marker
    u = b / np.linalg.norm(b)
    n = np.array([-u[1], u[0]])
    s = 0.12
    corner = np.array([proj - s * u, proj - s * u + s * n, proj + s * n])
    ax.plot(corner[:, 0], corner[:, 1], color=MUTED, lw=0.9)
    ax.text(*(a * 0.5 + np.array([-0.35, 0.1])), r"$\mathbf{a}$", color=INK, fontsize=13)
    ax.text(*(b * 1.15 + np.array([0.05, -0.05])), r"$\mathbf{b}$", color=INK, fontsize=13)
    ax.text(*(proj * 0.55 + np.array([0.05, -0.35])),
            r"$\mathrm{proj}_{\mathbf{b}}\,\mathbf{a}$", color=INK, fontsize=11)
    ax.text(*((proj + a) / 2 + np.array([0.08, 0.05])), r"$\mathbf{a}_{\perp}$", color=INK, fontsize=12)
    th = np.linspace(np.arctan2(b[1], b[0]), np.arctan2(a[1], a[0]), 40)
    ax.plot(0.45 * np.cos(th), 0.45 * np.sin(th), color=MUTED, lw=0.9)
    ax.text(0.5, 0.42, r"$\theta$", color=INK, fontsize=11)
    ax.set_xlim(-0.3, 3.8)
    ax.set_ylim(-0.6, 2.6)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")
    save(fig, "m1_projection.png")


def fig_m1_cross_product():
    a = np.array([1.0, 0.1, 0.25])
    b = np.array([0.25, 1.0, 0.1])
    c = np.cross(a, b)
    fig = plt.figure(figsize=(4.8, 4.2))
    ax = fig.add_subplot(projection="3d")
    poly = np.array([[0, 0, 0], a, a + b, b, [0, 0, 0]])
    ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], color=MUTED, lw=0.8)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.add_collection3d(Poly3DCollection([poly[:4]], facecolor="#cfd8e3", alpha=0.6, edgecolor="none"))
    for vec, col, lab, off in [
        (a, SERIES[0], r"$\mathbf{a}$", (0.05, -0.12, 0.0)),
        (b, SERIES[1], r"$\mathbf{b}$", (0.1, -0.08, -0.08)),
        (c, SERIES[2], r"$\mathbf{a}\times\mathbf{b}$", (-0.32, 0.0, -0.05)),
    ]:
        ax.add_artist(Arrow3D((0, 0, 0), vec, arrowstyle="-|>", color=col, lw=2.2, mutation_scale=14))
        ax.text(*(vec + np.array(off)), lab, color=INK, fontsize=12)
    # reference axes
    for vec, col, lab in [((1.3, 0, 0), X_RED, "x"), ((0, 1.45, 0), Y_GREEN, "y"), ((0, 0, 1.2), Z_BLUE, "z")]:
        ax.add_artist(Arrow3D((0, 0, 0), vec, arrowstyle="-|>", color=col, lw=1.0, mutation_scale=10, alpha=0.8))
        ax.text(*(np.array(vec) * 1.06), lab, color=col, fontsize=10)
    mid = (a + b) / 2
    ax.text(mid[0] + 0.12, mid[1] + 0.02, mid[2], "area = |a×b|", color=MUTED, fontsize=9)
    ax.set_xlim(-0.4, 1.3)
    ax.set_ylim(-0.2, 1.45)
    ax.set_zlim(0, 1.2)
    ax.view_init(elev=20, azim=-62)
    ax.set_axis_off()
    save(fig, "m1_cross_product.png")


# ---------------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------------
def fig_m3_svd_ellipse():
    J = np.array([[-1.0, -1.0], [1.0, 0.0]])  # 2R arm, l1=l2=1, theta=(0, 90 deg)
    U, S, Vt = np.linalg.svd(J)
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    ell = J @ circle
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.8))
    ax = axs[0]
    ax.plot(circle[0], circle[1], color=INK, lw=1.2)
    for i in range(2):
        arrow2d(ax, (0, 0), Vt[i], SERIES[i], lw=2.2)
        ax.text(*(Vt[i] * 1.12 + np.array([0.02, 0.02])), rf"$\mathbf{{v}}_{i + 1}$", color=INK, fontsize=12)
    ax.set_title(r"joint velocities, $\Vert\dot{\mathbf{q}}\Vert = 1$", fontsize=10, color=INK)
    ax.set_xlabel(r"$\dot\theta_1$")
    ax.set_ylabel(r"$\dot\theta_2$")
    ax = axs[1]
    ax.plot(ell[0], ell[1], color=INK, lw=1.2)
    for i in range(2):
        arrow2d(ax, (0, 0), S[i] * U[:, i], SERIES[i], lw=2.2)
        ax.text(*(S[i] * U[:, i] * 1.08 + np.array([0.03, 0.03])),
                rf"$\sigma_{i + 1}\mathbf{{u}}_{i + 1}$", color=INK, fontsize=12)
    ax.set_title(r"tool velocities $J\dot{\mathbf{q}}$  ($\sigma_1=1.618,\ \sigma_2=0.618$)", fontsize=10, color=INK)
    ax.set_xlabel(r"$\dot x$")
    ax.set_ylabel(r"$\dot y$")
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlim(-1.9, 1.9)
        ax.set_ylim(-1.9, 1.9)
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        ax.axvline(0, color="#bbbbbb", lw=0.6)
    fig.tight_layout()
    save(fig, "m3_svd_ellipse.png")


# ---------------------------------------------------------------------------
# M4
# ---------------------------------------------------------------------------
def fig_m4_atan2():
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color="#bbbbbb", lw=0.8)
    arrow2d(ax, (-1.5, 0), (1.6, 0), X_RED, lw=1.0)
    arrow2d(ax, (0, -1.5), (0, 1.6), Y_GREEN, lw=1.0)
    ax.text(1.62, -0.12, "x", color=X_RED, fontsize=11)
    ax.text(0.06, 1.6, "y", color=Y_GREEN, fontsize=11)
    for (px, py), col, lab in [((1, 1), SERIES[0], "(1, 1)"), ((-1, -1), SERIES[1], "(−1, −1)")]:
        arrow2d(ax, (0, 0), (px, py), col, lw=2.0)
        ax.plot(px, py, "o", color=col, ms=6)
        ax.text(px + (0.08 if px > 0 else -0.62), py + (0.08 if py > 0 else -0.2), lab, color=INK, fontsize=10)
    a1 = np.linspace(0, np.pi / 4, 30)
    ax.plot(0.35 * np.cos(a1), 0.35 * np.sin(a1), color=SERIES[0], lw=1.2)
    ax.text(0.42, 0.12, "atan2 = 45°", color=INK, fontsize=9)
    a2 = np.linspace(0, -3 * np.pi / 4, 60)
    ax.plot(0.55 * np.cos(a2), 0.55 * np.sin(a2), color=SERIES[1], lw=1.2, ls="--")
    ax.text(0.15, -0.85, "atan2 = −135°", color=INK, fontsize=9)
    ax.text(-1.55, 1.35, "atan(y/x) = 45° for both", color=MUTED, fontsize=9)
    ax.set_xlim(-1.7, 1.8)
    ax.set_ylim(-1.6, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "m4_atan2.png")


def fig_m4_2r_ik():
    l1, l2 = 1.0, 0.8
    target = np.array([1.2, 0.9])
    c2 = (target @ target - l1**2 - l2**2) / (2 * l1 * l2)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for k, sgn in enumerate([1, -1]):
        s2 = sgn * np.sqrt(1 - c2**2)
        t2 = np.arctan2(s2, c2)
        t1 = np.arctan2(target[1], target[0]) - np.arctan2(l2 * s2, l1 + l2 * c2)
        elbow = l1 * np.array([np.cos(t1), np.sin(t1)])
        tip = elbow + l2 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])
        assert np.allclose(tip, target)
        lab = (f"A: θ1 = {np.degrees(t1):.2f}°, θ2 = {np.degrees(t2):.2f}° (elbow down)" if sgn > 0
               else f"B: θ1 = {np.degrees(t1):.2f}°, θ2 = {np.degrees(t2):.2f}° (elbow up)")
        ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], color=SERIES[k], ls=STYLES[k],
                lw=3, marker="o", ms=7, label=lab)
    ax.plot(*target, marker="*", color=INK, ms=13, ls="none")
    ax.text(target[0] + 0.05, target[1] + 0.05, "target (1.2, 0.9)", color=INK, fontsize=9)
    ax.plot([0, target[0]], [0, target[1]], color="#bbbbbb", lw=0.8, ls=":")
    arrow2d(ax, (0, 0), (0.45, 0), X_RED, lw=1.2)
    arrow2d(ax, (0, 0), (0, 0.45), Y_GREEN, lw=1.2)
    ax.text(0.47, -0.08, "x", color=X_RED)
    ax.text(-0.1, 0.45, "y", color=Y_GREEN)
    ax.set_xlim(-0.3, 1.8)
    ax.set_ylim(-0.3, 1.45)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save(fig, "m4_2r_ik.png")


# ---------------------------------------------------------------------------
# M5
# ---------------------------------------------------------------------------
def fig_m5_free_response():
    wn = 5.0
    t = np.linspace(0, 3, 1500)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for k, z in enumerate([0.0, 0.2, 1.0, 2.0]):
        A = np.array([[0, 1], [-wn**2, -2 * z * wn]])
        sys = signal.StateSpace(A, np.zeros((2, 1)), np.array([[1.0, 0.0]]), np.zeros((1, 1)))
        _, y, _ = signal.lsim(sys, np.zeros_like(t), t, X0=[1.0, 0.0])
        name = {0.0: "undamped", 0.2: "underdamped", 1.0: "critically damped", 2.0: "overdamped"}[z]
        ax.plot(t, y, color=SERIES[k], ls=STYLES[k], label=f"ζ = {z:g} ({name})")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("x(t)")
    ax.set_xlim(0, 3)
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "m5_free_response.png")


# ---------------------------------------------------------------------------
# M6
# ---------------------------------------------------------------------------
def fig_m6_step_responses():
    wn = 1.0
    zetas = [0.2, 0.5, 1.0, 2.0]
    t = np.linspace(0, 20, 2000)
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axs[0]
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    ax.plot(wn * np.cos(th), wn * np.sin(th), color="#cccccc", lw=0.8, ls="--")
    markers = ["o", "s", "D", "^"]
    for k, z in enumerate(zetas):
        p = np.roots([1, 2 * z * wn, wn**2])
        ax.plot(p.real, p.imag, ls="none", marker=markers[k], color=SERIES[k], ms=8,
                markerfacecolor="white" if k % 2 else SERIES[k], markeredgewidth=1.8, label=f"ζ = {z:g}")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlim(-4, 0.5)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("Re s")
    ax.set_ylabel("Im s")
    ax.set_title("poles (ωn = 1)", fontsize=10, color=INK)
    ax.legend(loc="lower left", fontsize=8.5)
    ax = axs[1]
    for k, z in enumerate(zetas):
        _, y = signal.step(signal.TransferFunction([wn**2], [1, 2 * z * wn, wn**2]), T=t)
        ax.plot(t, y, color=SERIES[k], ls=STYLES[k], label=f"ζ = {z:g}")
    ax.axhline(1, color="#999999", lw=0.8)
    ax.axhspan(0.98, 1.02, color="#eeeeee", zorder=0)
    ax.set_xlim(0, 20)
    ax.set_xlabel("ωn t")
    ax.set_ylabel("y(t)")
    ax.set_title("unit-step response", fontsize=10, color=INK)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(17.0, 1.08, "±2% band", color=MUTED, fontsize=8)
    fig.tight_layout()
    save(fig, "m6_step_responses.png")


# ---------------------------------------------------------------------------
# M7
# ---------------------------------------------------------------------------
def fig_m7_integrators():
    g, l, h, n = 9.81, 1.0, 0.01, 1000

    def f(s):
        return np.array([s[1], -g / l * np.sin(s[0])])

    def energy(s):
        return 0.5 * l**2 * s[1] ** 2 + g * l * (1 - np.cos(s[0]))

    s_e = np.array([1.0, 0.0])
    s_s = s_e.copy()
    s_r = s_e.copy()
    E = np.zeros((3, n + 1))
    E[:, 0] = energy(s_e)
    for k in range(n):
        s_e = s_e + h * f(s_e)
        v = s_s[1] - h * g / l * np.sin(s_s[0])
        s_s = np.array([s_s[0] + h * v, v])
        k1 = f(s_r)
        k2 = f(s_r + h / 2 * k1)
        k3 = f(s_r + h / 2 * k2)
        k4 = f(s_r + h * k3)
        s_r = s_r + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        E[:, k + 1] = [energy(s_e), energy(s_s), energy(s_r)]
    t = np.arange(n + 1) * h
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.axhline(E[0, 0], color=INK, lw=0.8, label="exact (constant)")
    for k, lab in enumerate(["explicit Euler", "semi-implicit Euler", "RK4"]):
        ax.plot(t, E[k], color=SERIES[k], ls=STYLES[k], label=lab)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("energy per unit mass (J/kg)")
    ax.set_xlim(0, 10)
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "m7_integrators.png")


# ---------------------------------------------------------------------------
# M8
# ---------------------------------------------------------------------------
def fig_m8_two_link():
    l1, l2 = 1.0, 0.8
    t1, t2 = np.radians(35), np.radians(50)
    p1 = l1 * np.array([np.cos(t1), np.sin(t1)])
    p2 = p1 + l2 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], color=MUTED, lw=3, solid_capstyle="round")
    ax.plot(0, 0, marker="o", ms=10, color="white", markeredgecolor=INK, markeredgewidth=1.5)
    ax.plot(*p1, marker="o", ms=13, color=SERIES[0])
    ax.plot(*p2, marker="o", ms=13, color=SERIES[1])
    ax.text(p1[0] + 0.07, p1[1] - 0.1, r"$m_1$", fontsize=13, color=INK)
    ax.text(p2[0] + 0.07, p2[1], r"$m_2$", fontsize=13, color=INK)
    ax.text(*(p1 / 2 + np.array([0.04, -0.12])), r"$l_1$", fontsize=12, color=INK)
    ax.text(*((p1 + p2) / 2 + np.array([0.06, 0.0])), r"$l_2$", fontsize=12, color=INK)
    a = np.linspace(0, t1, 30)
    ax.plot(0.3 * np.cos(a), 0.3 * np.sin(a), color=INK, lw=1)
    ax.text(0.33, 0.08, r"$\theta_1$", fontsize=12, color=INK)
    ext = p1 + 0.45 * np.array([np.cos(t1), np.sin(t1)])
    ax.plot([p1[0], ext[0]], [p1[1], ext[1]], color="#bbbbbb", lw=0.8, ls="--")
    a = np.linspace(t1, t1 + t2, 30)
    ax.plot(p1[0] + 0.28 * np.cos(a), p1[1] + 0.28 * np.sin(a), color=INK, lw=1)
    ax.text(p1[0] + 0.25, p1[1] + 0.24, r"$\theta_2$", fontsize=12, color=INK)
    arrow2d(ax, (0, 0), (0.55, 0), X_RED, lw=1.2)
    arrow2d(ax, (0, 0), (0, 0.55), Y_GREEN, lw=1.2)
    ax.text(0.57, -0.07, "x", color=X_RED)
    ax.text(-0.09, 0.56, "y", color=Y_GREEN)
    arrow2d(ax, (1.35, 1.2), (1.35, 0.85), INK, lw=1.4)
    ax.text(1.4, 1.0, r"$g$", fontsize=12, color=INK)
    ax.plot([-0.2, 0.2], [-0.06, -0.06], color=INK, lw=1.2)
    for xh in np.linspace(-0.18, 0.18, 6):
        ax.plot([xh, xh - 0.05], [-0.06, -0.12], color=INK, lw=0.8)
    ax.set_xlim(-0.35, 1.6)
    ax.set_ylim(-0.2, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "m8_two_link.png")


if __name__ == "__main__":
    fig_m1_projection()
    fig_m1_cross_product()
    fig_m3_svd_ellipse()
    fig_m4_atan2()
    fig_m4_2r_ik()
    fig_m5_free_response()
    fig_m6_step_responses()
    fig_m7_integrators()
    fig_m8_two_link()
