# Next steps (in order)

1. ~~Audit Prof. Nandi's playlists~~ — replaced by following **Craig's chapters** (`SYLLABUS.md`, section 3 and the Craig 9–11 table).
2. Finish chapters **one at a time**, in this order: MFoR 2 (2.6, 2.7) → 1 → 3 → 4 → 5 → 6 → 7 → **8 (Baxter, NAO)** → RMC 1 → 2 → 3 → 4 → 5 → 6 → 7.
   For each: write the missing notebooks following `docs/STYLE_GUIDE.md` and `agent_brief.md`; run `python tools/check_notebooks.py <chapter>`;
   run `pytest`; then **commit the whole chapter in one commit and push** (no AI co-author lines).
3. Add the planned chapters: MFoR 9 (mechanism design, closed chains), 10 (programming languages, off-line programming), 11 (motion planning).
4. Then RMC 8 and 9; MFoR 8 extras (industrial six-axis arm, humanoid OP3, safety).
5. Choose a licence; set `main` as the default branch and delete the old `claude/…` branches on GitHub.

## How to resume with Claude Code
Start a session on this repository (branch `main`) and ask:
and ask:
"Finish MFoR Chapter 4 following plan/CHAPTER_BRIEFS.md, plan/agent_brief.md and docs/STYLE_GUIDE.md; one chapter at a time, no sub-agents."
Python environment: `pip install -r requirements.txt`; build a notebook with `python tools/sources/<course>_<NN>/nb_<c>_<s>.py`.
