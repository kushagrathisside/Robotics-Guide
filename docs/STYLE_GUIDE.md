# Style guide and authoring contract

This guide defines what every notebook in the repository must look like. It is written for human contributors
and for the automated authors that build chapters in parallel. The three finished notebooks of
[Chapter 2](../MFoR/02-spatial-descriptions-and-transformations/) (2.1–2.3) are the reference implementation:
**when this guide is ambiguous, imitate them.**

## 1. The unit of work: a notebook

A chapter is a folder of notebooks plus a `README.md`. Each notebook teaches **one coherent topic in 2–3 hours of
study** and is written as a section of a formal textbook: a title, an abstract, learning objectives, prose that
builds intuition before formulae, derivations, figures that can be played with, worked examples, a summary,
exercises with answers, and pointers to further reading.

Notebooks are **generated** from a short Python source so that they can be re-executed and reviewed as text:

```
tools/sources/<course>_<NN>/nb_<chapter>_<section>.py     # source of one notebook
MFoR/<NN>-<chapter-slug>/<chapter>.<section>-<slug>.ipynb  # executed output (committed)
```

`<course>` is `mfor` or `rmc`. Build one notebook with `python tools/sources/mfor_02/nb_2_1.py` (see
`tools/nbbuild.py`) and check it with `python tools/check_notebooks.py <path>`.

## 2. Required structure (in this order)

| # | Cell | Content |
|:--|:--|:--|
| 1 | Header (`tools.nbbuild.header`) | course · chapter, numbered title, Colab badge, 3–5-sentence abstract, table of primary reading, parallel reading, lectures, prerequisites, study time |
| 2 | **Learning objectives** | 4–7 numbered, testable verbs ("derive…", "compute…", "explain…") |
| 3 | **Contents** | numbered list with in-page anchors |
| 4 | Setup code cell | `SETUP` from `tools.nbbuild` (do not edit; it makes `rgkit` importable locally and on Colab) |
| 5 | Sections | numbered `## 1. …`; each section: prose → definition/theorem → derivation or intuition → figure or animation → code the reader can change |
| 6 | Real-robot section(s) | see §5 |
| 7 | **Worked examples** | numbered `Example <ch>.<sec>.<n>`; full solution in prose **and** a code cell that recomputes every number and `assert`s it |
| 8 | **Summary** | a table of the key results, then a "Common mistakes" list |
| 9 | **Exercises** | `<ch>.<sec>.<n>`; 5–8 problems mixing derivation, numerical, programming and (where relevant) a robot-model exercise; each answer in a `<details><summary>Answer</summary>` block with the final numbers |
| 10 | **Further reading and lectures** | verified links only (see §6) |
| 11 | Navigation (`tools.nbbuild.nav`) | previous · chapter contents · next |

## 3. Writing standards

* **Voice.** Formal, complete English in full paragraphs, as in an Oxford or Cambridge textbook. No slang, no
  "let's", no filler. Define every symbol on first use. Use bullet lists and tables where they clarify (a
  comparison, a set of formulae, a procedure), not in place of explanation.
* **Originality.** Explain in your own words. Do not paste or closely paraphrase sentences, figures or worked
  problems from textbooks, slides or the instructor's notes; cite them instead. Standard equations and
  published *numerical data* (for example a robot's link dimensions) may be used with attribution.
* **Notation.** Craig's leading superscript/subscript notation (${}^{A}_{B}T$, ${}^{A}P_{BORG}$, $c_{12}$, $s_{23}$).
  Vectors bold in prose where helpful; unit vectors with hats; angles in **radians** in code and equations
  unless degrees are shown explicitly. State which D-H convention is in use whenever a D-H table appears
  (**modified / Craig** is the default of the course).
* **Mathematics.** GitHub renders `$…$` and `$$…$$`. Avoid macros GitHub's MathJax lacks (`\newcommand`,
  `\begin{align}` — use `\begin{aligned}`), keep a blank line before and after display maths, and do not put `|`
  inside maths in tables (use `\mid` or `\lvert`).
* **Numbers.** Every number that appears in the prose of a worked example or exercise answer must be
  recomputed in code in the same notebook. Round sensibly (3–4 significant figures). If you cannot verify a
  number, remove it.
* **Citations.** Cite books at **chapter or section level you are certain of**; if unsure, cite the chapter only.
  Never invent a section number, equation number, page, URL or video ID.

## 4. Figures, animation and interactivity

* Colours: **X red, Y green, Z blue** for axes, orange for the vector of interest, grey for construction lines.
  Always label axes and use `rgkit.plotting` (`draw_frame`, `draw_vector`, `new_axes3d`, …).
* Every notebook needs at least **two figures** and, where motion helps, **one animation**
  (`rgkit.animation.animate`) and **one interactive explorer** (`rgkit.notebook.interact` with `ipywidgets`
  sliders). In published notebooks the animation appears as a key-frame filmstrip and the explorer as a static
  figure at its default setting — both are automatic when `RG_STATIC=1`, which the builder sets.
* GIFs are written to `<chapter>/assets/` (≤ 500 KB each) and embedded in the chapter `README.md`.
* Keep each notebook ≤ **600 KB** (the checker fails at 700 KB): modest figure sizes, few frames, no large arrays
  printed.
* Use `np.set_printoptions` via `rgkit.setup_notebook()`; avoid printing `-0.` (add `+ 0.0`).

## 5. Real robots, MuJoCo and verification

* **Use real robot models** whenever a concept can be shown on one: `rgkit.robots.load_robot("ur5e" | "panda" |
  "2r" | "iiwa14" | "sawyer" | "op3")` loads a [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  model (downloaded on first use at a pinned commit, never copied into the repository). Show the model with
  `rgkit.mujoco_utils.render`, draw frames with `robots.frames_option`, and print attribution with
  `robots.describe(name)`. Toy models are acceptable only for minimal teaching examples (for instance a planar
  link drawn in matplotlib).
* **Cross-check against the simulator.** Where practical, compute a quantity by hand (D-H forward kinematics, a
  Jacobian, a mass matrix, a controller response) and **compare it with MuJoCo** in an `assert`/printed `True`.
  This is the repository's main defence against mistakes.
* Numerical results in the notebooks must be reproducible: seed random generators, avoid wall-clock-dependent
  output (timings may be printed only in clearly non-essential cells).
* Put reusable, tested logic into `rgkit/<module>.py` plus `tests/test_<module>.py`; keep notebook code readable
  (the reader is meant to modify it). Do not edit existing shared modules (`transforms`, `kinematics`, `plotting`,
  `animation`, `notebook`, `mujoco_utils`, `robots`); create a new module instead.

## 6. Sources, lectures and links

* Lecture links come from [`docs/VIDEO_INDEX.md`](VIDEO_INDEX.md) (verified from the instructor's notes) and
  [`docs/REFERENCES.md`](REFERENCES.md). `tools.common.V("<title prefix>")` returns a Markdown link from the
  index and raises if the title is ambiguous or missing.
* The three playlists of Prof. G. C. Nandi are: (1) *Mathematical Foundation of Robotics* (first course),
  (2) *Mathematical Foundation of Robotics — Execution Level …* (second course, closest to Robot Motion Control) and
  (3) *Introduction to Humanoid and Collaborative Robots* (third course: case studies). URLs of (2) and (3) are not yet
  verified; refer to them **by title** without a URL.
* Do not link to unauthorised copies of copyrighted books. Free and legal sources are listed in `REFERENCES.md`.
* Each notebook's header lists: *primary reading* (Craig or, for RMC, the control text), *parallel reading*
  (Lynch & Park, Spong et al., Siciliano et al., Corke, …), *lectures*, *prerequisites* and *study time*.

## 7. Repository etiquette for parallel authors

* Write **only** inside: your chapter folder, `tools/sources/<course>_<NN>/`, `tests/test_<yourmodule>.py`, and new
  `rgkit/<yourmodule>.py` files. Never edit another chapter, `docs/`, `README.md`, `SYLLABUS.md` or existing
  shared modules. If you need a change there, say so in your final report.
* Do **not** run `git commit`, `git push` or `git add`; the maintainer integrates the work.
* Every chapter folder has a `README.md`: chapter overview, learning path, a table of notebooks (title, topics, time,
  prerequisites), animation previews (`![…](assets/….gif)`), and a list of what the chapter does *not* cover.
* Before reporting finish: all notebooks build without errors, `python tools/check_notebooks.py <your notebooks>`
  passes, your tests pass, and every worked example and exercise answer has been verified numerically.
