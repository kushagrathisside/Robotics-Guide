"""Common notebook configuration so that every chapter looks the same."""

from __future__ import annotations

import numpy as np


def setup_notebook(precision: int = 4):
    """Configure numpy printing and matplotlib style used in all notebooks."""
    import matplotlib as mpl

    np.set_printoptions(precision=precision, suppress=True, linewidth=100)
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "figure.figsize": (6.0, 4.5),
        "font.size": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "cm",
        "animation.embed_limit": 40.0,
    })


def interact(f, **widgets):
    """``ipywidgets.interact`` that degrades gracefully on GitHub.

    When notebooks are executed for publication (``RG_STATIC=1``) the
    function is called once with each widget's initial value, so the saved
    notebook shows a static figure instead of an empty widget area. In a
    live Jupyter or Colab session the usual sliders appear.
    """
    from .animation import is_static

    if is_static():
        f(**{k: getattr(v, "value", v) for k, v in widgets.items()})
        return None
    import ipywidgets

    return ipywidgets.interact(f, **widgets)
