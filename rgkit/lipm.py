"""Linear inverted pendulum model (LIPM), capture point and zero-moment point (ZMP).

The LIPM treats a legged robot as a point mass at constant height ``zc`` above the ground whose horizontal
motion is governed by the position ``p`` of the zero-moment point (centre of pressure):

.. math::  \\ddot x = \\omega^2 (x - p), \\qquad \\omega = \\sqrt{g / z_c} .

All functions are written for one horizontal coordinate (the sagittal ``x`` or the frontal ``y``); the two
decouple.  Symbols: ``x`` position of the centre of mass (COM), ``v`` its velocity, ``p`` ZMP position.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lipm_omega", "lipm_propagate", "capture_point", "orbital_energy", "zmp_lipm", "zmp_multibody",
    "periodic_gait", "simulate_steps", "dcm_foot_for_target",
]

G = 9.81


def lipm_omega(zc, g=G):
    """Natural frequency ``omega = sqrt(g / zc)`` (rad/s) of the LIPM."""
    return np.sqrt(g / zc)


def lipm_propagate(x, v, p, t, zc, g=G):
    """Exact solution after time ``t`` with the ZMP held at ``p``: returns ``(x(t), v(t))``.

    ``x(t) = p + (x0 - p) cosh(w t) + (v0 / w) sinh(w t)``,  ``v(t) = w (x0 - p) sinh(w t) + v0 cosh(w t)``.
    """
    w = lipm_omega(zc, g)
    c, s = np.cosh(w * t), np.sinh(w * t)
    return p + (x - p) * c + v / w * s, w * (x - p) * s + v * c


def capture_point(x, v, zc, g=G):
    """Instantaneous capture point (divergent component of motion) ``xi = x + v / omega``.

    It obeys ``xi' = omega (xi - p)``: the only unstable mode of the LIPM.  Stepping onto ``p = xi`` brings the
    COM to rest above the foot.
    """
    return x + v / lipm_omega(zc, g)


def orbital_energy(x, v, p, zc, g=G):
    """Conserved quantity ``E = v^2 / 2 - omega^2 (x - p)^2 / 2`` for a fixed ZMP ``p``."""
    w = lipm_omega(zc, g)
    return 0.5 * v ** 2 - 0.5 * w ** 2 * (x - p) ** 2


def zmp_lipm(x, xdd, zc, g=G):
    """ZMP implied by the LIPM for a given COM acceleration: ``p = x - zc xdd / g``."""
    return x - zc * xdd / g


def zmp_multibody(x, z, xdd, zdd, Ldot_y, mass, g=G):
    """ZMP (centre of pressure on flat ground at height 0) of a multi-body system in the sagittal plane.

    ``x, z`` -- COM position; ``xdd, zdd`` -- COM acceleration; ``Ldot_y`` -- rate of change of the angular
    momentum about the COM (``y`` component); ``mass`` -- total mass.  Derived from Newton's and Euler's
    equations: ``p = x - (z m xdd + Ldot_y) / (m (zdd + g))``.
    """
    return x - (z * mass * xdd + Ldot_y) / (mass * (zdd + g))


def periodic_gait(step_length, step_time, zc, g=G):
    """Symmetric periodic LIPM gait with the ZMP at each support foot.

    The COM starts a step at ``-s/2`` and ends it at ``+s/2`` relative to the support foot, with the same
    speed at both instants:  ``v = (w s / 2) coth(w T / 2)``.  Returns ``(x_start, v)``.
    """
    w = lipm_omega(zc, g)
    return -0.5 * step_length, 0.5 * w * step_length / np.tanh(0.5 * w * step_time)


def simulate_steps(x0, v0, feet, step_time, zc, g=G, samples_per_step=40):
    """COM trajectory when the ZMP sits at ``feet[k]`` during step ``k`` (each lasting ``step_time``).

    Returns arrays ``t, x, v, p`` (``p`` is the ZMP at each sample).
    """
    t_all, x_all, v_all, p_all = [], [], [], []
    x, v, t0 = x0, v0, 0.0
    tau = np.linspace(0.0, step_time, samples_per_step, endpoint=False)
    for p in feet:
        xs, vs = lipm_propagate(x, v, p, tau, zc, g)
        t_all.append(t0 + tau)
        x_all.append(xs)
        v_all.append(vs)
        p_all.append(np.full_like(tau, p))
        x, v = lipm_propagate(x, v, p, step_time, zc, g)
        t0 += step_time
    return (np.concatenate(t_all), np.concatenate(x_all), np.concatenate(v_all), np.concatenate(p_all))


def dcm_foot_for_target(x, v, xi_target, step_time, zc, g=G):
    """ZMP/foot position that moves the capture point from ``xi0 = x + v/w`` to ``xi_target`` in ``step_time``.

    From ``xi(T) = p + (xi0 - p) exp(w T)``  ->  ``p = (xi_target - exp(w T) xi0) / (1 - exp(w T))``.
    """
    w = lipm_omega(zc, g)
    e = np.exp(w * step_time)
    xi0 = x + v / w
    return (xi_target - e * xi0) / (1.0 - e)
