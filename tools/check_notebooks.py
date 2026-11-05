"""Quality gate for the notebooks. Run:  python tools/check_notebooks.py [paths ...]

Exit status is non-zero if any notebook fails a *hard* check. Checks:

* the notebook is executed: no ``error`` outputs and every code cell has been run;
* required sections exist (learning objectives, contents, summary, exercises, further reading);
* size budget (default 700 KB) so that GitHub renders the file quickly;
* no machine-specific absolute paths (``/home/``, ``/tmp/``, ``C:\\``);
* every relative Markdown link points at an existing file;
* every ``youtube.com/watch?v=`` link is *known* (docs/video_index.json, docs/REFERENCES.md or
  docs/VIDEO_INDEX.md) — this blocks invented video links;
* a Colab badge and a navigation cell are present.

Warnings (printed, not fatal): fewer than 6 worked-example/verification code cells, no real-robot use,
no figure output.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_KB = 700
REQUIRED = [r"learning objectives", r"contents", r"summary", r"exercises", r"further reading"]

known_text = ""
for f in ("docs/REFERENCES.md", "docs/VIDEO_INDEX.md"):
    p = ROOT / f
    if p.exists():
        known_text += p.read_text(encoding="utf-8")
known_text += (ROOT / "docs/video_index.json").read_text(encoding="utf-8") if (ROOT / "docs/video_index.json").exists() else ""


def check(path: Path):
    errors, warnings = [], []
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    md = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    code = [c for c in cells if c["cell_type"] == "code"]
    size_kb = path.stat().st_size / 1024
    if size_kb > MAX_KB:
        errors.append(f"size {size_kb:.0f} KB exceeds {MAX_KB} KB")
    for i, c in enumerate(code):
        if c.get("execution_count") is None and "".join(c["source"]).strip():
            errors.append(f"code cell {i} was not executed")
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors.append(f"code cell {i} raised {o.get('ename')}: {str(o.get('evalue'))[:80]}")
    low = md.lower()
    for req in REQUIRED:
        if not re.search(rf"^#+\s.*{req}", low, re.M):
            errors.append(f"missing heading containing '{req}'")
    allsrc = md + "\n".join("".join(c["source"]) for c in code)
    for bad in ("/home/", "/tmp/", "C:\\\\", "/root/"):
        if bad in allsrc:
            errors.append(f"machine-specific path fragment {bad!r} in source")
    for m in re.finditer(r"\]\(([^)#\s]+)(#[^)]*)?\)", md):
        target = m.group(1)
        if re.match(r"[a-z]+://|mailto:", target):
            continue
        if not (path.parent / target).resolve().exists():
            errors.append(f"broken relative link: {target}")
    for m in re.finditer(r"youtube\.com/watch\?v=([\w-]{11})", allsrc):
        if m.group(1) not in known_text:
            errors.append(f"unknown video id {m.group(1)} (not in docs/video_index.json or REFERENCES.md)")
    if "colab-badge" not in md:
        errors.append("no Colab badge")
    if "Chapter contents" not in md:
        errors.append("no navigation cell (use tools.nbbuild.nav)")
    has_img = any("image/png" in o.get("data", {}) for c in code for o in c.get("outputs", []))
    if not has_img:
        warnings.append("no figure output")
    if "rgkit.robots" not in allsrc and "robots." not in allsrc and "mujoco" not in allsrc.lower():
        warnings.append("no MuJoCo / real-robot use (fine for pure-maths notebooks)")
    if len(code) < 6:
        warnings.append(f"only {len(code)} code cells")
    return errors, warnings, size_kb


def main(argv):
    paths = [Path(a).resolve() for a in argv] or sorted(list((ROOT / "MFoR").rglob("*.ipynb")) + list((ROOT / "RMC").rglob("*.ipynb")))
    failed = 0
    for p in paths:
        e, w, kb = check(p)
        status = "FAIL" if e else "ok  "
        print(f"{status} {p.relative_to(ROOT)}  ({kb:.0f} KB)")
        for x in e:
            print(f"      ERROR   {x}")
        for x in w:
            print(f"      warning {x}")
        failed += bool(e)
    print(f"\n{len(paths) - failed}/{len(paths)} notebooks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
