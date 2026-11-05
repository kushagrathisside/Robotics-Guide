# Brief for chapter authors (read fully before doing anything)

You are building **one chapter, end to end, alone**, of an open-source, textbook-quality Jupyter-notebook repository for two
IIIT Allahabad courses (MFoR — Mathematical Foundations of Robotics, which follows Craig's *Introduction to Robotics*; RMC —
Robot Motion Control). Other agents are building the other chapters at the same time in the same checkout. The repository is
for students of **any** college, not for exam cramming.

Repository: `/home/user/Robotics-Guide` (branch `claude/beautiful-fermat-e3ja9i`). **Do not run `git add`, `git commit`
or `git push`.** Do not touch files outside your allowed area (below).

## Read these first (in this order)
1. `/home/user/Robotics-Guide/docs/STYLE_GUIDE.md` — the authoring contract. Follow it exactly.
2. The three reference notebooks and their sources: `MFoR/02-spatial-descriptions-and-transformations/2.1…2.3*.ipynb` and
   `tools/sources/mfor_02/nb_2_{1,2,3}.py`. Imitate their depth, structure, tone, figures, MuJoCo cross-checks and exercises.
   (Read the `.py` sources rather than the huge `.ipynb` JSON.)
3. `SYLLABUS.md` (your chapter's notebook plan — you may refine titles/split/merge to 4–7 good notebooks, keeping the
   chapter's scope), `docs/REFERENCES.md`, `docs/VIDEO_INDEX.md`, `docs/MUJOCO.md`.
4. Library API (read the docstrings): `rgkit/transforms.py`, `rgkit/kinematics.py`, `rgkit/plotting.py`,
   `rgkit/animation.py`, `rgkit/notebook.py`, `rgkit/mujoco_utils.py`, `rgkit/robots.py`, `tools/nbbuild.py`,
   `tools/common.py`, `tools/check_notebooks.py`.
5. The instructor's notes PDF (optional but valuable for MFoR chapters — it shows what students of this course are expected to
   know, including past-paper question styles): `/root/.claude/uploads/6ab1f4d3-5244-53eb-9a10-41ffaf3c90f1/4f2fc9db-MFoR_MidSem_Notes_Detailed.pdf`
   (58 pages; the Read tool needs the `pages` parameter, max 20 pages per call; TOC on pages 5–7). Use it as a *guide to topics
   and as a source of verified numerical data*, never copy its prose or figures.

## Environment (already set up — do not install anything unless essential)
* Python with every dependency (numpy, scipy, sympy, matplotlib, ipywidgets, mujoco, nbclient, pytest, …):
  `PY=/tmp/claude-0/-home-user-Robotics-Guide/6ab1f4d3-5244-53eb-9a10-41ffaf3c90f1/scratchpad/venv/bin/python`
  (`uv pip install --python $PY <pkg>` if a package is truly missing; plain `pip` is very slow.)
* Always `export RG_CACHE=/tmp/claude-0/-home-user-Robotics-Guide/6ab1f4d3-5244-53eb-9a10-41ffaf3c90f1/scratchpad/rgcache`
  before building: the real robot models (2r, ur5e, panda, iiwa14, sawyer, op3) are already cached there.
* Headless rendering works with `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` (the builder sets this for notebooks; set it for
  your own scripts).
* **Build** a notebook: `cd /home/user/Robotics-Guide && PYTHONPATH=. RG_CACHE=… $PY tools/sources/<course>_<NN>/nb_<c>_<s>.py`
  (takes 10–90 s; it executes the notebook and writes it next to the others). **Check**: `$PY tools/check_notebooks.py <path>`.
  **Look at your figures**: `$PY -I tools/extract_figures.py <notebook.ipynb> <outdir>` writes every output PNG; open a few with
  the Read tool and fix cluttered, overlapping or misleading plots. Use `tools/sources/mfor_02/nb_2_3.py` as the template for
  a source file (the first lines show the required imports).
* Machine resources are shared with ~16 other builders: keep animations short (≤ 60 frames), avoid heavy loops, and do not
  render hundreds of MuJoCo frames per notebook (≤ 30 renders).

## Allowed area (write only here)
* `<Course>/<NN>-<slug>/` — your chapter folder (create it) with notebooks, `assets/` (GIFs), and `README.md`.
* `tools/sources/<course>_<NN>/` — the Python sources of your notebooks (create it; no `__init__.py` needed).
* `rgkit/<yourmodule>.py` — NEW tested library modules only (name them after your topic, e.g. `rgkit/jacobian.py`); with
  `tests/test_<yourmodule>.py`. Never edit existing files in `rgkit/`, `tools/`, `docs/`, `tests/` or root files, and never
  edit another chapter. If you believe a shared file has a bug or needs a feature, **describe it in your final report**.
* Temporary files go in your own scratch directory, not in the repository.

## Quality bar
* Formal textbook English, full paragraphs, original wording, correct maths, verified numbers (see the style guide).
* **Every worked example and exercise answer is verified by code in the notebook**; where a quantity exists in MuJoCo (poses,
  Jacobians, mass matrices, dynamics, control response), **cross-check against the simulator with a real robot model**.
* 4–7 notebooks of 2–3 hours of study each, in `<ch>.<section>-<slug>.ipynb`, with ≥ 2 figures, an animation or interactive
  explorer where motion helps, ≥ 5 worked examples, ≥ 6 exercises with answers, summary + common mistakes, further reading.
* Links: only from `docs/VIDEO_INDEX.md`/`docs/REFERENCES.md` (use `tools.common.V(...)` for the video index). Cite book
  chapters only at a level you are sure of. Never invent URLs, IDs, section numbers or equation numbers.
* The chapter `README.md`: overview, learning path, table of notebooks (title · topics · time · prerequisites), animation
  previews from `assets/`, scope notes, and a short "where the textbooks cover this" table (chapter-level only).
* Navigation cells must link only to notebooks that exist (previous/next within your chapter; the first notebook has no
  "previous"; the last has no "next"). The checker enforces this; also run it on the whole chapter at the end.

## Final report (keep it short, ≤ 25 lines)
List notebooks written (title, size), library modules/tests added, what you verified against MuJoCo, check results, anything
you could not verify or had to leave out, and any change you need in shared files. Do not paste notebook content.
