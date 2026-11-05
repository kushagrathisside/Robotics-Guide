# Plan and catch-up notes

This folder exists so that you can **pick the work up again** if a session ends before the chapters are finished. It contains (1) the plan and the
instructions used to build chapters, (2) the research behind the syllabus, and (3) the status of every chapter. Finished chapters are in `MFoR/` and `RMC/`;
unfinished notebook sources are in `tools/sources/` (they have no built notebook yet).

## Files in `plan/`
| File | What it is |
|:--|:--|
| `STATUS.md` | what exists, per chapter, and what is left |
| `NEXT_STEPS.md` | the ordered to-do list and how to resume |
| `CHAPTER_BRIEFS.md` | the specific instructions for each chapter (scope, real-robot checks, sources) |
| `agent_brief.md` | the common authoring brief every chapter builder follows (environment paths are from the original session; adapt) |
| `research_report.md` | syllabus research: IIIT-A syllabus findings, consolidated topic trees, book chapter maps (items marked [mem]/? are unverified) |
| `SYLLABUS_snapshot.md` | the syllabus at the time of this snapshot |

## Decisions made by the repository owner (keep following them)
* **Publish chapter by chapter**: push a whole chapter, never a small section.
* **No AI co-author lines in commits** (no `Co-Authored-By`, no session lines). Credit to Claude appears **only in the README**.
* **One chapter at a time, one builder per chapter; no new parallel sub-agents** (to save usage).
* Use **real robot models** (MuJoCo Menagerie; Baxter from robosuite) wherever they teach better; check hand results against MuJoCo.
* Cover **Craig's chapters** (MFoR 1–7 + planned 9–11; Craig Ch. 9–11 are taught in RMC); Craig is the completeness reference in place of Prof. Nandi's playlists.

## Where things are
Everything is on the `main` branch: the library `rgkit`, tools, docs, maths primer, README, CI, the ten chapters started (see `STATUS.md`) and their
tested modules. Unfinished notebook sources: `tools/sources/mfor_01`, `mfor_08`, `rmc_01`, `rmc_08`.
