"""rgkit — shared code for the Robotics-Guide notebooks.

* :mod:`rgkit.transforms` — tested reference implementations of rotations,
  homogeneous transforms, Euler angles, angle-axis and quaternions.
* :mod:`rgkit.plotting`   — drawing helpers for frames, vectors and bodies.
* :mod:`rgkit.animation`  — animations that also display on GitHub.
* :mod:`rgkit.mujoco_utils` — rendering MuJoCo models inside notebooks
  (imported on demand, because MuJoCo is optional).
"""

from . import transforms, plotting, animation  # noqa: F401
from .notebook import setup_notebook, interact  # noqa: F401

__version__ = "0.1.0"
