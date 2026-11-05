"""Smoke tests for rgkit.control_plots: every helper draws without error on a headless backend."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from rgkit import control_plots as cp, linear_control as lc   # noqa: E402


def test_all_helpers_draw():
    G = lc.TF([1], [1, 3, 2, 0])
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    cp.draw_chain(axs[0, 0], ["Controller", "Motor\nand gearbox", "Link"], ["r", "u", "tau", "q"])
    cp.draw_loop(axs[0, 1], d="D(s)")
    cp.draw_loop(axs[0, 2], feedback=None)
    cp.root_locus_plot(axs[1, 0], G, np.concatenate([[0], np.logspace(-3, 2, 200)]), mark_gains=[1.0])
    cp.s_plane_regions(axs[1, 1], zeta_min=0.5, wn_max=6, sigma_min=1)
    cp.pz_map(axs[1, 1], [-1 + 1j, -1 - 1j], [-3])
    t = np.linspace(0, 5, 500)
    cp.annotate_step(axs[1, 2], t, lc.second_order(5, 0.4).step(t))
    plt.close(fig)
    fig, axs = plt.subplots(2, 1)
    cp.bode_plot(axs, lc.TF([10], [1, 6, 5, 0]), np.logspace(-2, 2, 200), margins=True,
                 asymptotes=[("K", 2), ("int", 1), ("lag", 1), ("lag", 5)])
    plt.close(fig)
    fig, ax = plt.subplots()
    cp.nyquist_plot(ax, lc.TF([5], [1, 3, 2, 0]), xlim=(-3, 1), ylim=(-2, 2))
    plt.close(fig)


def test_asymptotic_bode_is_close_to_exact_away_from_corners():
    G = lc.TF([10], np.polymul([1, 0], np.polymul([1, 1], [1, 5])))      # 2 / (s (s+1)(s/5+1))
    w = np.array([0.01, 100.0])
    asym = cp.asymptotic_bode([("K", 2), ("int", 1), ("lag", 1), ("lag", 5)], w)
    exact, _ = lc.bode(G, w)
    assert np.allclose(asym, exact, atol=0.2)
