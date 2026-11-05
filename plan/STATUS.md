# Status snapshot

`python tools/check_notebooks.py` gives the live truth. At the time of this snapshot (one chapter finished, others stopped part-way
to save usage):

| Chapter | Notebooks that exist | Left to do |
|:--|:--|:--|
| **MFoR 1 Introduction, DOF** | 1.1–1.5 + README (**done, passes checker**) | — |
| **MFoR 2 Spatial descriptions** | 2.1–2.5 + README (**done, passes checker**) | 2.6 screw theory, 2.7 practice problems |
| MFoR 3 Forward kinematics | 3.1 (passes) | 3.2–3.5, README; modules `dh_tools`, `dh_plot`, `poe` exist |
| MFoR 4 Inverse kinematics | 4.1 | 4.2–4.6, README; module `ik` exists |
| MFoR 5 Jacobians and statics | 5.1 (passes), 5.2 | 5.3–5.5; README exists; module `jacobian` |
| MFoR 6 Dynamics | 6.1 (passes) | 6.2–6.5; README exists; module `dynamics` |
| MFoR 7 Trajectories | 7.1, 7.2 (pass) | 7.3, 7.4; module `trajectory` |
| MFoR 8 Case studies | none | **8.1 Baxter (complete study), 8.2 NAO (complete study)**, then others |
| MFoR 9 Mechanism design + closed chains | not started | planned (Craig Ch. 8; Lynch & Park closed chains) |
| MFoR 10 Programming languages + off-line programming | not started | planned (Craig Ch. 12–13) |
| MFoR 11 Motion planning | not started | planned (Craig 7.9; Lynch & Park motion planning) |
| RMC 1 Control foundations | none (module `linear_control` exists) | all |
| RMC 2 Independent-joint control | 2.1 | 2.2–2.5 |
| RMC 3 Lyapunov, model-based | 3.1 (README exists) | 3.2–3.4 |
| RMC 4 Robust, adaptive, operational space | none (module `robust_adaptive`) | all |
| RMC 5 Force and interaction | none (module `interaction_control`) | all |
| RMC 6 Teleoperation | none (module `teleop`) | all |
| RMC 7 Optimal, predictive, learning | 7.1 (passes; README exists) | 7.2–7.5 |
| RMC 8 Sensing, visual servoing | stopped | all (modules `sensing`, `visualservo`) |
| RMC 9 Underactuated, mobile | stopped | all (module `underactuated`) |

Most failures in the checker are only **dangling links** to notebooks that were never written, and missing chapter READMEs;
fix by writing the notebook or replacing the link with "(planned)".

## Known issues
* Completeness is checked against **Craig's chapters** (owner's instruction), not against Prof. Nandi's playlists; see the Craig 9–11 table in `SYLLABUS.md`.
* The 2.5 figures of the finished Chapter 2 were not visually inspected.
* IIIT-A syllabus text comes from search snippets, not the official PDFs.
* NAO has no MuJoCo model; build a skeleton from the published dimensions (see `CHAPTER_BRIEFS.md`).
