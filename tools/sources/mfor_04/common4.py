"""Shared helpers for the Chapter 4 notebook sources (not imported by the notebooks themselves)."""
import json, pathlib

_IDX = json.loads((pathlib.Path(__file__).resolve().parents[3] / "docs" / "video_index.json").read_text(encoding="utf-8"))


def V(prefix: str) -> str:
    """Markdown link for the unique entry of docs/video_index.json whose title starts with ``prefix``."""
    hits = [e for e in _IDX if e["title"].startswith(prefix)]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} video-index entries start with {prefix!r}: {[h['title'] for h in hits]}")
    e = hits[0]
    who = {"Nandi": "Prof. G. C. Nandi, ", "TE": "ThatsEngineering — ", "MR": "Modern Robotics "}[e["source"]]
    return f"[{who}{e['title']}]({e['url']})"
