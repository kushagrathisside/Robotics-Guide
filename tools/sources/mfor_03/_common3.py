"""Helpers shared by the Chapter 3 notebook sources (kept next to them; not part of the notebooks)."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

try:                                                   # the shared helper, if the maintainer has added it
    from tools.common import V                        # noqa: F401
except ImportError:                                    # fall back to the verified index in docs/video_index.json
    _INDEX = json.loads((ROOT / "docs" / "video_index.json").read_text(encoding="utf-8"))

    def V(prefix: str) -> str:
        """Markdown link to the one video of ``docs/video_index.json`` whose title starts with ``prefix``."""
        hits = [e for e in _INDEX if e["title"].startswith(prefix)]
        if len(hits) != 1:
            raise KeyError(f"{len(hits)} videos match {prefix!r}")
        return f"[{hits[0]['title']}]({hits[0]['url']})"
