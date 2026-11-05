"""Shared constants for the Chapter 1 (Control-Systems Foundations) notebook sources."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repository root
from tools.nbbuild import md, code, SETUP, header, nav, build

COURSE = "Robot Motion Control"
CHAPTER = "Chapter 1 — Control-Systems Foundations"
CH_DIR = "RMC/01-control-systems-foundations"

NB = {
    1: ("1.1 Modelling a Robot Joint", "1.1-modelling-a-robot-joint.ipynb"),
    2: ("1.2 Time Response, Stability and Steady-State Error", "1.2-time-response-stability-and-steady-state-error.ipynb"),
    3: ("1.3 The Root-Locus Method", "1.3-root-locus.ipynb"),
    4: ("1.4 Frequency Response and Stability Margins", "1.4-frequency-response-and-stability-margins.ipynb"),
    5: ("1.5 PID Control", "1.5-pid-control.ipynb"),
    6: ("1.6 State-Space Design", "1.6-state-space-design.ipynb"),
}


def nav_for(k):
    """Navigation cell linking only to notebooks that exist on disk."""
    root = pathlib.Path(__file__).resolve().parents[3] / CH_DIR
    ok = lambda e: e if (e and (root / e[1]).exists()) else None
    return nav(ok(NB.get(k - 1)), ok(NB.get(k + 1)))


# lecture links: only items listed in docs/REFERENCES.md
BRUNTON = "[S. Brunton, *Control Bootcamp* (overview video)](https://www.youtube.com/watch?v=Pi7l8mMjYVE)"
TECHTALKS = "[MathWorks *MATLAB Tech Talks* — control systems](https://www.mathworks.com/videos/tech-talks/controls.html)"
NPTEL_RMC = "[NPTEL (IIT Palakkad), *Mechanics and Control of Robotic Manipulators*](https://nptel.ac.in/courses/112106304)"
NANDI2 = "Prof. G. C. Nandi, *Mathematical Foundation of Robotics — Execution Level …* (his second playlist, closest to this course)"
M6 = "[Maths primer M6 — Laplace transforms and linear systems](../../math-primer/M6-laplace-and-linear-systems.md)"
M5 = "[M5 — Calculus and differential equations](../../math-primer/M5-calculus-and-differential-equations.md)"
