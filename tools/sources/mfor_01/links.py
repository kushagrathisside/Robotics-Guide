"""Lecture links for Chapter 1, resolved from docs/video_index.json (never typed by hand)."""
import json
import pathlib

_IDX = json.loads((pathlib.Path(__file__).resolve().parents[3] / "docs" / "video_index.json").read_text(encoding="utf-8"))
_WHO = {"Nandi": "Prof. G. C. Nandi", "MR": "Modern Robotics", "TE": "ThatsEngineering"}


def V(prefix):
    """Markdown link of the unique video whose title starts with ``prefix``."""
    hits = [e for e in _IDX if e["title"].startswith(prefix)]
    if len(hits) != 1:
        raise KeyError(f"{prefix!r} matches {len(hits)} videos")
    e = hits[0]
    return f"[{_WHO[e['source']]} — {e['title']}]({e['url']})"
