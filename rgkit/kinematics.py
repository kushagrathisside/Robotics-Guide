"""Denavit–Hartenberg link transforms and serial-chain forward kinematics.

This small module is shared by several chapters (forward and inverse
kinematics, Jacobians, dynamics). It implements **Craig's modified
(proximal) convention** — the one used in the lectures and in the course
examination papers — and, for comparison, the **standard (distal)
convention** of Spong and others.

Modified (Craig):  ``i-1_i T = Rx(alpha_{i-1}) Dx(a_{i-1}) Rz(theta_i) Dz(d_i)``
Standard (Spong):  ``i-1_i T = Rz(theta_i) Dz(d_i) Dx(a_i) Rx(alpha_i)``

A *D-H table* is a list of rows ``(alpha, a, d, theta, kind)``. For the
modified convention ``alpha`` and ``a`` are ``alpha_{i-1}`` and ``a_{i-1}``;
for the standard convention they are ``alpha_i`` and ``a_i``. ``kind`` is
``"R"`` (revolute: the joint variable is added to ``theta``) or ``"P"``
(prismatic: the joint variable is added to ``d``). Angles are in radians.
"""

from __future__ import annotations

import numpy as np

from .transforms import rotx, rotz, make_T, transl

__all__ = ["dh_modified", "dh_standard", "fk_dh", "fk_tool", "planar_fk"]


def dh_modified(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Craig's link transform ``Rx(alpha) Dx(a) Rz(theta) Dz(d)`` (4x4)."""
    return make_T(rotx(alpha)) @ transl(a, 0, 0) @ make_T(rotz(theta)) @ transl(0, 0, d)


def dh_standard(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Standard (distal) link transform ``Rz(theta) Dz(d) Dx(a) Rx(alpha)`` (4x4)."""
    return make_T(rotz(theta)) @ transl(0, 0, d) @ transl(a, 0, 0) @ make_T(rotx(alpha))


def fk_dh(table, q=None, convention: str = "modified"):
    """Cumulative transforms ``[0_0T, 0_1T, ..., 0_nT]`` of a serial chain.

    ``q`` supplies one value per row (radians or metres); ``None`` means all zeros.
    """
    link = dh_modified if convention == "modified" else dh_standard
    q = np.zeros(len(table)) if q is None else np.asarray(q, dtype=float)
    Ts, T = [np.eye(4)], np.eye(4)
    for (alpha, a, d, theta, kind), qi in zip(table, q):
        T = T @ (link(alpha, a, d, theta + qi, ) if kind == "R" else link(alpha, a, d + qi, theta))
        Ts.append(T)
    return Ts


def fk_tool(table, q=None, convention: str = "modified", tool=None) -> np.ndarray:
    """Base-to-tool transform; ``tool`` is an optional fixed 4x4 flange-to-tool transform."""
    T = fk_dh(table, q, convention)[-1]
    return T if tool is None else T @ tool


def planar_fk(lengths, q):
    """Joint and tip positions of a planar nR arm: returns an ``(n+1, 2)`` array."""
    lengths, q = np.asarray(lengths, float), np.asarray(q, float)
    ang = np.cumsum(q)
    pts = np.zeros((len(lengths) + 1, 2))
    pts[1:, 0] = np.cumsum(lengths * np.cos(ang))
    pts[1:, 1] = np.cumsum(lengths * np.sin(ang))
    return pts
