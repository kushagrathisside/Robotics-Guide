"""Shared constants and helpers for the Chapter 5 notebook sources."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repository root
from tools.nbbuild import md, code, SETUP, header, nav, build

COURSE = "Mathematical Foundations of Robotics"
CHAPTER = "Chapter 5 — Velocity Kinematics, Jacobians, Singularities and Statics"
CH_DIR = "MFoR/05-velocity-kinematics-and-statics"

_INDEX = json.loads((pathlib.Path(__file__).resolve().parents[3] / "docs" / "video_index.json").read_text(encoding="utf-8"))
_LABEL = {"Nandi": "Prof. G. C. Nandi", "TE": "ThatsEngineering", "MR": "Modern Robotics"}


def V(prefix, source=None):
    """Markdown link of the unique entry of docs/video_index.json whose title starts with ``prefix``."""
    hits = [e for e in _INDEX if e["title"].startswith(prefix) and (source is None or e["source"] == source)]
    if len(hits) != 1:
        raise KeyError(f"video title prefix {prefix!r} matches {len(hits)} entries")
    e = hits[0]
    return f"[{_LABEL[e['source']]} — {e['title']}]({e['url']})"


NOTEBOOKS = [
    ("5.1", "5.1-velocity-of-rigid-bodies-and-propagation.ipynb", "5.1 Velocity of Rigid Bodies and Velocity Propagation"),
    ("5.2", "5.2-the-manipulator-jacobian.ipynb", "5.2 The Manipulator Jacobian"),
    ("5.3", "5.3-frames-and-the-analytical-jacobian.ipynb", "5.3 Jacobians in Other Frames and the Analytical Jacobian"),
    # planned (not yet written): ("5.4", "5.4-singularities-and-manipulability.ipynb", "5.4 Singularities and Manipulability"),
    # ("5.5", "5.5-statics-and-force-ellipsoids.ipynb", "5.5 Static Forces, Joint Torques and Force Ellipsoids"),
    # ("5.6", "5.6-redundancy-null-space-and-damped-least-squares.ipynb", "5.6 Redundancy, the Null Space and Damped Least Squares"),
]


def nav_for(sec):
    i = [n[0] for n in NOTEBOOKS].index(sec)
    prev = (NOTEBOOKS[i - 1][2], NOTEBOOKS[i - 1][1]) if i > 0 else None
    nxt = (NOTEBOOKS[i + 1][2], NOTEBOOKS[i + 1][1]) if i < len(NOTEBOOKS) - 1 else None
    return nav(prev, nxt)


def prereq(sec):
    i = [n[0] for n in NOTEBOOKS].index(sec)
    return f"[{NOTEBOOKS[i-1][2]}]({NOTEBOOKS[i-1][1]})" if i > 0 else None
