# Plan and catch-up notes (drafts branch)

This branch (`claude/plan-drafts`) exists so that you can **pick the work up again** if a session ends before the chapters are
finished. It contains (1) the plan and the instructions used to build chapters, (2) the research behind the syllabus, and (3) a
**snapshot of all in-progress chapter files at their real paths**, so any finished chapter can be merged into the main branch
(`claude/beautiful-fermat-e3ja9i`) by copying its folder.

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
* Cover Craig's chapters (MFoR 1–7 + planned 9–11; control chapters 9–11 live in RMC) and **everything in Prof. G. C. Nandi's playlists**
  (the playlist contents are not yet audited — YouTube was blocked in the build environment; paste the video titles to do it).
* Add complete sections on **Baxter** and **NAO** (MFoR Chapter 8).

## What is on the main branch
Foundation (library `rgkit`, tools, docs, maths primer, README, CI), Chapter 2 notebooks 2.1–2.3, and the syllabus.
Everything else is only here, in this snapshot.
