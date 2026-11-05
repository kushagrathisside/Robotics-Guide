"""Constants and helpers shared by the Chapter 7 notebook sources (authoring-time only; not imported by the notebooks)."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
COURSE = "Mathematical Foundations of Robotics"
CHAPTER = "Chapter 7 — Trajectory Generation"
CH_DIR = "MFoR/07-trajectory-generation"

_INDEX = json.loads((ROOT / "docs" / "video_index.json").read_text(encoding="utf-8"))
_LABEL = {"MR": "Modern Robotics", "TE": "ThatsEngineering", "Nandi": "Prof. G. C. Nandi"}


def V(prefix, source=None):
    """Markdown link to the one video of ``docs/video_index.json`` whose title starts with ``prefix``."""
    hits = [e for e in _INDEX if e["title"].startswith(prefix) and (source is None or e["source"] == source)]
    if len(hits) != 1:
        raise ValueError(f"{prefix!r} matches {len(hits)} videos")
    e = hits[0]
    return f"[{_LABEL[e['source']]} {e['title']}]({e['url']})"


NO_LECTURE = ("*No lecture of the verified course index treats trajectory generation* — see the "
              "[chapter README](README.md) for what to watch instead")

PREREQ_CALC = ("Derivatives of polynomials and small linear systems "
               "([M5 Calculus](../../math-primer/M5-calculus-and-differential-equations.md), "
               "[M2 Matrices](../../math-primer/M2-matrices-and-linear-maps.md))")
