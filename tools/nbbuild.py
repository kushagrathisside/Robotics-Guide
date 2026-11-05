"""Tiny helper to author notebooks as Python lists of cells, then execute them."""
import os, sys, pathlib, textwrap
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

REPO = pathlib.Path(__file__).resolve().parents[1]
GH = "kushagrathisside/Robotics-Guide"
BRANCH = "main"


def md(s):
    return new_markdown_cell(textwrap.dedent(s).strip("\n"))


def code(s):
    return new_code_cell(textwrap.dedent(s).strip("\n"))


SETUP = r'''
# ── Setup ─────────────────────────────────────────────────────────────────────
# Makes the shared `rgkit` package importable, both locally and on Google Colab.
import sys, pathlib, subprocess
if "google.colab" in sys.modules:                                   # Colab
    if not pathlib.Path("Robotics-Guide").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/kushagrathisside/Robotics-Guide.git"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mujoco", "mediapy"], check=True)
    sys.path.insert(0, "Robotics-Guide")
else:                                                               # local Jupyter
    root = pathlib.Path.cwd().resolve()
    while not (root / "rgkit").is_dir() and root != root.parent:
        root = root.parent
    sys.path.insert(0, str(root))

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math, Markdown
import ipywidgets as widgets

from rgkit import setup_notebook, interact
from rgkit.plotting import (new_axes3d, draw_frame, draw_vector, draw_point, draw_box,
                            draw_frame_2d, draw_vector_2d, matrix_to_latex, AXIS_COLORS)
from rgkit.animation import animate, filmstrip

setup_notebook()
deg = np.deg2rad          # degrees -> radians
'''


def header(course, chapter, section, title, path, rows, abstract):
    colab = f"https://colab.research.google.com/github/{GH}/blob/{BRANCH}/{path}"
    table = "\n".join(f"| **{k}** | {v} |" for k, v in rows)
    return md(f"""
*{course} · {chapter}*

# {section} {title}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})

{abstract}

| | |
|:--|:--|
{table}
""")


def nav(prev=None, nxt=None, index="README.md"):
    parts = []
    parts.append(f"[← {prev[0]}]({prev[1]})" if prev else "")
    parts.append(f"[Chapter contents]({index})")
    parts.append(f"[{nxt[0]} →]({nxt[1]})" if nxt else "")
    return md("---\n\n" + " &nbsp;·&nbsp; ".join(p for p in parts if p))


def shrink_png(b64):
    """Quantise a PNG to an adaptive palette: ~3x smaller, visually identical for line art."""
    import base64, io
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    q = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
    buf = io.BytesIO(); q.save(buf, format="PNG", optimize=True)
    out = base64.b64encode(buf.getvalue()).decode()
    return out if len(out) < len(b64) else b64


def build(cells, rel_path, execute=True, timeout=600):
    out = REPO / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    if execute:
        os.environ.update(RG_STATIC="1", RG_SAVE_ASSETS="1", MUJOCO_GL="osmesa",
                          PYOPENGL_PLATFORM="osmesa", MPLBACKEND="module://matplotlib_inline.backend_inline")
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(out.parent)}})
        client.execute()
        for c in nb.cells:                       # keep diffs clean, shrink images
            if c.cell_type == "code":
                c.metadata.pop("execution", None)
                for o in c.get("outputs", []):
                    d = o.get("data", {})
                    if "image/png" in d:
                        d["image/png"] = shrink_png(d["image/png"])
    nbformat.write(nb, out)
    size = out.stat().st_size / 1024
    print(f"wrote {rel_path}  ({size:.0f} KB, {len(cells)} cells)")
    return out
