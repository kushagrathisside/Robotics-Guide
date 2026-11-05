"""Plotting helpers for the control-systems notebooks (block diagrams, pole maps, root locus, Bode, Nyquist).

All functions draw into a Matplotlib ``Axes`` you pass in (or create one) and return it, so figures can be
composed freely.  Colours follow the repository convention: orange for the quantity of interest, grey for
construction lines; poles are crosses, zeros are circles.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Arc

from . import linear_control as lc

__all__ = ["draw_chain", "draw_loop", "pz_map", "s_plane_regions", "annotate_step", "root_locus_plot", "bode_plot",
           "asymptotic_bode", "nyquist_plot", "ORANGE", "BLUE", "GREY", "GREEN", "RED"]

ORANGE, BLUE, GREY, GREEN, RED = "#e8710a", "#1f6fb4", "#8a8a8a", "#2a9d4b", "#c0392b"


# ---------------------------------------------------------------------------------------------- block diagrams
def _box(ax, cx, cy, w, h, text, fc="#eef3fa", ec=BLUE, fontsize=11):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=4)


def _arrow(ax, p0, p1, color="#333", lw=1.4, **kw):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, **kw),
                zorder=2)


def draw_chain(ax, labels, signals=None, w=1.5, gap=1.0, h=0.8, fontsize=11):
    """Draw boxes ``labels`` left to right joined by arrows; ``signals`` (len = len(labels)+1) label the arrows.

    A label may contain ``\\n``; box widths follow the longest line.
    """
    x = 0.0
    centres, widths = [], []
    for lab in labels:
        wi = max(w, 0.15 * max(len(line) for line in lab.split("\n")) + 0.5)
        hi = h + 0.3 * (lab.count("\n"))
        centres.append(x + wi / 2)
        widths.append(wi)
        _box(ax, x + wi / 2, 0, wi, hi, lab, fontsize=fontsize)
        x += wi + gap
    n = len(labels)
    _arrow(ax, (-gap, 0), (0, 0))
    for i in range(n - 1):
        _arrow(ax, (centres[i] + widths[i] / 2, 0), (centres[i + 1] - widths[i + 1] / 2, 0))
    _arrow(ax, (centres[-1] + widths[-1] / 2, 0), (centres[-1] + widths[-1] / 2 + gap, 0))
    if signals:
        xs = [-gap / 2] + [centres[i] + widths[i] / 2 + gap / 2 for i in range(n)]
        for xv, s in zip(xs, signals):
            ax.text(xv, 0.12, s, ha="center", va="bottom", fontsize=fontsize)
    ax.set_xlim(-gap - 0.2, x + 0.1 - gap + gap)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def draw_loop(ax, forward=("C(s)", "G(s)"), feedback="H(s)", r="r", y="y", d=None, d_at=1, fontsize=11):
    """Unity- or non-unity-feedback loop: summing junction, forward blocks, feedback block; optional disturbance."""
    w, h, gap = 1.3, 0.8, 0.8
    x0 = 0.0
    ax.add_patch(Circle((x0, 0), 0.22, fc="white", ec="#333", lw=1.4, zorder=3))
    ax.text(x0 - 0.07, 0.0, "+", ha="center", va="center", fontsize=9, zorder=4)
    ax.text(x0 + 0.02, -0.38, "−", ha="center", va="center", fontsize=11, zorder=4)
    _arrow(ax, (x0 - 1.1, 0), (x0 - 0.22, 0))
    ax.text(x0 - 0.95, 0.15, r, ha="center", va="bottom", fontsize=fontsize)
    x = x0 + 0.22 + gap
    centres = []
    prev = (x0 + 0.22, 0)
    for lab in forward:
        centres.append(x + w / 2)
        _box(ax, x + w / 2, 0, w, h, lab, fontsize=fontsize)
        _arrow(ax, prev, (x, 0))
        prev = (x + w, 0)
        x += w + gap
    xend = prev[0] + gap
    _arrow(ax, prev, (xend, 0))
    ax.text(xend - 0.1, 0.15, y, ha="center", va="bottom", fontsize=fontsize)
    xb = prev[0] + gap / 2
    ax.plot([xb, xb], [0, -1.4], color="#333", lw=1.4)
    ax.plot(xb, 0, "o", color="#333", ms=4)
    if feedback:
        xm = (x0 + xb) / 2
        _box(ax, xm, -1.4, w, h, feedback, fontsize=fontsize)
        ax.plot([xb, xm + w / 2], [-1.4, -1.4], color="#333", lw=1.4)
        _arrow(ax, (xm - w / 2, -1.4), (x0, -1.4 + 0.0))
        ax.plot([x0, x0], [-1.4, -0.22], color="#333", lw=1.4)
        _arrow(ax, (x0, -0.8), (x0, -0.22))
    else:
        ax.plot([xb, x0], [-1.4, -1.4], color="#333", lw=1.4)
        _arrow(ax, (x0, -0.9), (x0, -0.22))
        ax.plot([x0, x0], [-1.4, -0.9], color="#333", lw=1.4)
    if d is not None:
        xd = centres[d_at] - w / 2 - gap / 2 if d_at > 0 else centres[0] - w / 2 - gap / 2
        ax.add_patch(Circle((xd, 0), 0.2, fc="white", ec="#333", lw=1.4, zorder=3))
        ax.text(xd, 0, "+", ha="center", va="center", fontsize=9, zorder=4)
        _arrow(ax, (xd, 1.1), (xd, 0.2))
        ax.text(xd + 0.1, 1.0, d, ha="left", va="center", fontsize=fontsize)
    ax.set_xlim(x0 - 1.3, xend + 0.3)
    ax.set_ylim(-2.1, 1.5 if d is not None else 1.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


# ---------------------------------------------------------------------------------------------- s-plane
def pz_map(ax=None, poles=(), zeros=(), color=BLUE, label=None, size=70):
    """Pole-zero map: poles as crosses, zeros as circles."""
    ax = ax or plt.subplots(figsize=(4.6, 4.0))[1]
    poles, zeros = np.atleast_1d(poles), np.atleast_1d(zeros)
    if poles.size:
        ax.scatter(poles.real, poles.imag, marker="x", s=size, color=color, lw=2, zorder=5, label=label)
    if zeros.size:
        ax.scatter(zeros.real, zeros.imag, marker="o", s=size, facecolor="none", edgecolor=color, lw=1.8, zorder=5)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Re $s$ (rad/s)")
    ax.set_ylabel("Im $s$ (rad/s)")
    ax.grid(alpha=0.3)
    return ax


def s_plane_regions(ax, zeta_min=None, wn_max=None, sigma_min=None, xlim=(-12, 2), ylim=(-10, 10)):
    """Shade the part of the left half-plane that meets damping, natural-frequency and decay-rate specifications.

    ``zeta_min``: poles must lie inside the cone around the negative real axis; ``wn_max``: inside the circle;
    ``sigma_min``: to the left of ``-sigma_min``.
    """
    xs = np.linspace(*xlim, 800)
    ys = np.linspace(*ylim, 800)
    X, Y = np.meshgrid(xs, ys)
    S = X + 1j * Y
    ok = X < 0
    if zeta_min is not None:
        ok &= np.abs(Y) <= np.abs(X) * np.sqrt(1 - zeta_min**2) / zeta_min
    if wn_max is not None:
        ok &= np.abs(S) <= wn_max
    if sigma_min is not None:
        ok &= X <= -sigma_min
    ax.contourf(X, Y, ok.astype(float), levels=[0.5, 1.5], colors=[ORANGE], alpha=0.22)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return ax


def annotate_step(ax, t, y, final=1.0, band=0.02, metrics=None, color=BLUE):
    """Plot a step response with the usual specification markers (final value, tolerance band, peak, settling time)."""
    m = metrics or lc.step_metrics(t, y, final=final, band=band)
    ax.plot(t, y, color=color, lw=2)
    ax.axhline(final, color=GREY, lw=1, ls="--")
    ax.axhspan(final * (1 - band), final * (1 + band), color=GREY, alpha=0.15)
    if m["Mp"] > 1e-4:
        ip = int(np.argmax(y))
        ax.plot(t[ip], y[ip], "o", color=ORANGE, zorder=5)
        ax.annotate(f"$M_p$ = {100 * m['Mp']:.1f} %", (t[ip], y[ip]), (t[ip] + 0.06 * t[-1], y[ip] + 0.05 * final),
                    fontsize=9, arrowprops=dict(arrowstyle="-", color=GREY))
    if np.isfinite(m["ts"]):
        ax.axvline(m["ts"], color=ORANGE, lw=1, ls=":")
        ax.text(m["ts"], 0.05 * final, f" $t_s$ = {m['ts']:.3g} s", color=ORANGE, fontsize=9, va="bottom")
    ax.set_xlabel("time (s)")
    return ax


def root_locus_plot(ax, G, gains, mark_gains=(), show_asymptotes=True, xlim=None, ylim=None, color=BLUE):
    """Root locus of ``1 + K G(s) = 0`` for ``K`` in ``gains`` with poles, zeros, asymptotes and marked gains."""
    P = lc.root_locus(G, gains)
    for j in range(P.shape[1]):
        ax.plot(P[:, j].real, P[:, j].imag, color=color, lw=1.8)
        k = len(P) // 2
        if abs(P[k + 1, j] - P[k, j]) > 0:
            ax.annotate("", xy=(P[k + 1, j].real, P[k + 1, j].imag), xytext=(P[k, j].real, P[k, j].imag),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))
    pz_map(ax, G.poles(), G.zeros(), color="k")
    if show_asymptotes:
        c, ang = lc.rl_asymptotes(G)
        if len(ang) and np.isfinite(c):
            L = 1e3
            for a in ang:
                ax.plot([c, c + L * np.cos(np.radians(a))], [0, L * np.sin(np.radians(a))], color=GREY, lw=1, ls="--")
    for K in mark_gains:
        r = lc.root_locus(G, [K])[0]
        ax.scatter(r.real, r.imag, s=46, color=ORANGE, zorder=6)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    return ax


# ---------------------------------------------------------------------------------------------- frequency domain
def asymptotic_bode(factors, w):
    """Straight-line (asymptotic) magnitude in dB of a product of standard factors.

    ``factors`` is a list of tuples: ``("K", k)`` constant gain; ``("int", n)`` ``1/s^n``; ``("lag", wc)`` ``1/(s/wc+1)``;
    ``("lead", wc)`` ``(s/wc+1)``; ``("quad", wn)`` second-order lag ``1/((s/wn)^2 + 2 zeta (s/wn) + 1)`` (corner at ``wn``).
    """
    w = np.asarray(w, float)
    mag = np.zeros_like(w)
    for kind, v in factors:
        if kind == "K":
            mag += 20 * np.log10(abs(v))
        elif kind == "int":
            mag += -20 * v * np.log10(w)
        elif kind == "lag":
            mag += -20 * np.log10(np.maximum(w / v, 1.0))
        elif kind == "lead":
            mag += 20 * np.log10(np.maximum(w / v, 1.0))
        elif kind == "quad":
            mag += -40 * np.log10(np.maximum(w / v, 1.0))
    return mag


def bode_plot(axs, G, w, color=BLUE, label=None, margins=False, asymptotes=None, lw=2):
    """Magnitude (dB) and phase (deg) of ``G`` on the pair of axes ``axs = (ax_mag, ax_phase)``.

    ``margins=True`` marks the gain and phase margins.  ``asymptotes`` (a list for :func:`asymptotic_bode`) adds the
    straight-line magnitude approximation.
    """
    ax_m, ax_p = axs
    mag, ph = lc.bode(G, w)
    ax_m.semilogx(w, mag, color=color, lw=lw, label=label)
    ax_p.semilogx(w, ph, color=color, lw=lw)
    if asymptotes is not None:
        ax_m.semilogx(w, asymptotic_bode(asymptotes, w), color=GREY, lw=1.2, ls="--", label="asymptotes")
    ax_m.axhline(0, color="#999", lw=0.8)
    ax_p.axhline(-180, color="#999", lw=0.8)
    if margins:
        m = lc.margins(G, wmin=w[0] / 10, wmax=w[-1] * 10)
        if np.isfinite(m["pm"]):
            ax_m.axvline(m["wgc"], color=ORANGE, lw=1, ls=":")
            ax_p.axvline(m["wgc"], color=ORANGE, lw=1, ls=":")
            p0 = np.interp(np.log10(m["wgc"]), np.log10(w), ph)
            ax_p.annotate("", xy=(m["wgc"], -180), xytext=(m["wgc"], p0),
                          arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
            ax_p.text(m["wgc"] * 1.08, (p0 - 180) / 2, f"PM = {m['pm']:.0f}°", color=ORANGE, fontsize=9, va="center")
        if np.isfinite(m["gm"]):
            ax_m.axvline(m["wpc"], color=ORANGE, lw=1, ls=":")
            ax_p.axvline(m["wpc"], color=ORANGE, lw=1, ls=":")
            ax_m.annotate("", xy=(m["wpc"], 0), xytext=(m["wpc"], -m["gm_db"]),
                          arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
            ax_m.text(m["wpc"] * 1.08, -m["gm_db"] / 2, f"GM = {m['gm_db']:.1f} dB", color=ORANGE, fontsize=9, va="center")
    ax_m.set_ylabel("magnitude (dB)")
    ax_p.set_ylabel("phase (deg)")
    ax_p.set_xlabel("frequency (rad/s)")
    for a in axs:
        a.grid(True, which="both", alpha=0.3)
    return axs


def nyquist_plot(ax, G, wmax=1e3, color=BLUE, label=None, xlim=None, ylim=None, **kw):
    """Nyquist curve (both branches) with the critical point ``-1`` and direction arrows."""
    c = lc.nyquist_curve(G, wmax=wmax, **kw)
    ax.plot(c.real, c.imag, color=color, lw=1.8, label=label)
    # arrows along the positive-frequency branch
    n = len(c)
    for frac in (0.62, 0.72):
        i = int(frac * n)
        d = c[i + 3] - c[i]
        if abs(d) > 0:
            ax.annotate("", xy=(c[i + 3].real, c[i + 3].imag), xytext=(c[i].real, c[i].imag),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
    ax.plot(-1, 0, "+", color=RED, ms=12, mew=2, zorder=6)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Re $L(j\\omega)$")
    ax.set_ylabel("Im $L(j\\omega)$")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    return ax
