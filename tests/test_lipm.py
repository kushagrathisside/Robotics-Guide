"""Tests for rgkit.lipm: closed forms against numerical integration, capture point, periodic gait."""
import numpy as np
from scipy.integrate import solve_ivp

from rgkit import lipm


def test_propagate_matches_numerical_integration():
    zc, p = 0.8, 0.05
    w = lipm.lipm_omega(zc)
    sol = solve_ivp(lambda t, s: [s[1], w ** 2 * (s[0] - p)], [0, 0.7], [0.1, 0.3], rtol=1e-11, atol=1e-12)
    assert np.allclose(lipm.lipm_propagate(0.1, 0.3, p, 0.7, zc), sol.y[:, -1], atol=1e-8)


def test_orbital_energy_and_capture_point():
    zc, p = 0.8, 0.0
    t = np.linspace(0, 0.5, 11)
    x, v = lipm.lipm_propagate(0.05, 0.4, p, t, zc)
    E = lipm.orbital_energy(x, v, p, zc)
    assert np.ptp(E) < 1e-12
    xi = lipm.capture_point(x, v, zc)
    w = lipm.lipm_omega(zc)
    assert np.allclose(xi - p, (lipm.capture_point(0.05, 0.4, zc) - p) * np.exp(w * t))
    # stepping on the capture point brings the COM to rest above the foot
    xi0 = lipm.capture_point(0.05, 0.4, zc)
    xe, ve = lipm.lipm_propagate(0.05, 0.4, xi0, 3.0, zc)
    assert np.isclose(xe, xi0, atol=1e-4) and abs(ve) < 1e-3


def test_zmp_formulas_consistent():
    # LIPM: constant height, no angular momentum -> p = x - zc xdd / g
    zc, g, m = 0.7, 9.81, 3.0
    assert np.isclose(lipm.zmp_multibody(0.1, zc, 0.4, 0.0, 0.0, m, g), lipm.zmp_lipm(0.1, 0.4, zc, g))
    # static: ZMP below the COM
    assert np.isclose(lipm.zmp_multibody(0.1, zc, 0.0, 0.0, 0.0, m, g), 0.1)
    # the LIPM acceleration xdd = w^2 (x - p) is recovered
    p = 0.02
    xdd = lipm.lipm_omega(zc, g) ** 2 * (0.1 - p)
    assert np.isclose(lipm.zmp_lipm(0.1, xdd, zc, g), p)


def test_periodic_gait_repeats_each_step():
    s, T, zc = 0.3, 0.5, 0.8
    x0, v0 = lipm.periodic_gait(s, T, zc)
    x1, v1 = lipm.lipm_propagate(x0, v0, 0.0, T, zc)
    assert np.isclose(x1, s / 2) and np.isclose(v1, v0)
    feet = [k * s for k in range(6)]
    t, x, v, p = lipm.simulate_steps(x0, v0, feet, T, zc)
    # at the start of every step the COM is s/2 behind the support foot with the same speed
    for k in range(6):
        i = k * 40
        assert np.isclose(x[i] - feet[k], -s / 2, atol=1e-9) and np.isclose(v[i], v0)


def test_dcm_foot_hits_target():
    zc, T = 0.8, 0.4
    x, v, target = 0.0, 0.5, 0.35
    p = lipm.dcm_foot_for_target(x, v, target, T, zc)
    xe, ve = lipm.lipm_propagate(x, v, p, T, zc)
    assert np.isclose(lipm.capture_point(xe, ve, zc), target)
