"""Shared constants and helpers of the Chapter 8 notebook sources (not a notebook)."""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COURSE = "Mathematical Foundations of Robotics"
CHAPTER = "Chapter 8 — Case Studies: Collaborative Arms, Industrial Arms and Humanoids"
CH_DIR = "MFoR/08-case-studies"

FILES = {
    "8.1": ("8.1-a-seven-joint-collaborative-arm.ipynb", "A Seven-Joint Collaborative Arm"),
    "8.2": ("8.2-an-industrial-six-axis-arm.ipynb", "An Industrial Six-Axis Arm: from Frames to Singularities"),
    "8.3": ("8.3-pick-and-place-with-a-ur5e.ipynb", "Pick and Place with a UR5e in MuJoCo"),
    "8.4": ("8.4-a-humanoid-robot.ipynb", "A Humanoid Robot: the ROBOTIS OP3"),
    "8.5": ("8.5-safety-and-human-robot-interaction.ipynb", "Safety and Human–Robot Interaction"),
}
ORDER = ["8.1", "8.2", "8.3", "8.4", "8.5"]


def prev_next(sec):
    i = ORDER.index(sec)
    prev = None if i == 0 else (f"{ORDER[i-1]} {FILES[ORDER[i-1]][1]}", FILES[ORDER[i-1]][0])
    nxt = None if i == len(ORDER) - 1 else (f"{ORDER[i+1]} {FILES[ORDER[i+1]][1]}", FILES[ORDER[i+1]][0])
    return prev, nxt


def link(sec):
    """Markdown link to a sibling notebook of this chapter."""
    return f"[{sec} {FILES[sec][1]}]({FILES[sec][0]})"


_INDEX = (ROOT / "docs" / "VIDEO_INDEX.md").read_text(encoding="utf-8")
_SECTIONS = {"Nandi": "Prof. G. C. Nandi", "TE": "ThatsEngineering", "MR": "Modern Robotics"}


def vid(prefix, who):
    """Markdown link to a video of docs/VIDEO_INDEX.md whose title starts with ``prefix`` (``who``: Nandi | TE | MR)."""
    section, hits = None, []
    for line in _INDEX.splitlines():
        if line.startswith("## "):
            section = line
            continue
        m = re.match(r"\|\s*(.+?)\s*\|\s*\[watch\]\((https://www\.youtube\.com/watch\?v=[\w-]{11})\)", line)
        if m and section and _SECTIONS[who] in section and m.group(1).startswith(prefix):
            hits.append((m.group(1), m.group(2)))
    if len(hits) != 1:
        raise ValueError(f"video title prefix {prefix!r} gives {len(hits)} hits in {who}")
    title, url = hits[0]
    label = {"Nandi": "Prof. G. C. Nandi", "TE": "ThatsEngineering", "MR": "Modern Robotics"}[who]
    return f"[{label}, {title}]({url})"
