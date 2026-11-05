import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rgkit import dh_plot as dp
from rgkit import kinematics as kin
from rgkit.plotting import new_axes3d

PUMA = [(0, 0, 0, 0, "R"), (-np.pi / 2, 0, 0, 0, "R"), (0, 0.4318, 0.15, 0, "R"),
        (-np.pi / 2, 0.0203, 0.4318, 0, "R"), (np.pi / 2, 0, 0, 0, "R"), (-np.pi / 2, 0, 0, 0, "R")]


def test_draw_dh_link_returns_the_link_transform():
    ax = new_axes3d(lim=1)
    T = dp.draw_dh_link(ax, 0.8, 0.5, 0.3, -0.6)
    assert np.allclose(T, kin.dh_modified(0.8, 0.5, 0.3, -0.6))
    plt.close("all")


def test_draw_arm_returns_frames_for_both_conventions():
    for table, conv in ((PUMA, "modified"), ([(r[0], r[1], r[2], r[3], r[4]) for r in PUMA], "standard")):
        ax = new_axes3d(lim=1)
        Ts = dp.draw_arm(ax, table, np.linspace(0.1, 0.6, 6), conv)
        assert len(Ts) == 7 and np.allclose(Ts[-1], kin.fk_dh(table, np.linspace(0.1, 0.6, 6), conv)[-1])
        dp.equal_limits(ax, ax._dh_points)
        plt.close("all")


def test_arc_ends_at_requested_angle():
    ax = new_axes3d(lim=1)
    pts = dp.draw_arc3d(ax, [0, 0, 0], [0, 0, 1], [1, 0, 0], np.pi / 2, 1.0)
    assert np.allclose(pts[:, -1], [0, 1, 0], atol=1e-12)
    plt.close("all")
