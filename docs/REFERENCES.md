# References and Learning Resources

This page lists the textbooks, lecture series, university courses, software and interactive resources that this guide draws on. It covers two courses:

- **MFoR**: *Mathematical Foundations of Robotics*. This is roughly a first course in manipulator robotics: spatial descriptions, transforms, forward and inverse kinematics, Jacobians and statics, dynamics, and trajectory generation.
- **RMC**: *Robot Motion Control*. This covers linear and independent-joint control, PD/PID, computed torque, nonlinear, adaptive and robust control, force and impedance control, master–slave teleoperation, and learning-based control.

The guide is written for students at any institution. Where a resource is tied to IIIT Allahabad (IIIT-A), it is marked as such.

**How to read the verification notes.**
- *Verified* means the URL and its description were seen during research, on the resource itself or in a search-engine result for that resource.
- *Unverified* means the item is believed to exist but could not be confirmed. No URL is given for an unverified item unless that URL was actually seen.
- Commercial textbooks are listed without download links. Free links appear only where the authors or publisher provide the book free of charge.

---

## 1. Core textbooks (kinematics, dynamics, manipulator control)

| # | Book | Edition used here | Serves | Chapters most relevant | Free legal copy |
|---|------|-------------------|--------|------------------------|-----------------|
| 1 | J. J. Craig, *Introduction to Robotics: Mechanics and Control* | 3rd ed. (Pearson/Prentice Hall, 2005; ISBN 978-0-201-54361-2). A 4th ed. (Pearson, 2017/2018; Global Edition ISBN 978-1-292-16495-3) uses the same 13-chapter structure. | MFoR (primary); RMC (Ch. 9–11) | MFoR: Ch. 2 Spatial descriptions and transformations; Ch. 3 Manipulator kinematics; Ch. 4 Inverse manipulator kinematics; Ch. 5 Jacobians: velocities and static forces; Ch. 6 Manipulator dynamics; Ch. 7 Trajectory generation; Appendices A–C. RMC: Ch. 9 Linear control of manipulators; Ch. 10 Nonlinear control of manipulators; Ch. 11 Force control of manipulators. | No (commercial). Publisher page: https://www.pearson.com/en-us/subject-catalog/p/introduction-to-robotics-mechanics-and-control/P200000003304/9780133489798 ; Library of Congress TOC (3rd ed.): https://catdir.loc.gov/catdir/toc/fy0604/2004275113.html |
| 2 | K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and Control* | Cambridge University Press, 2017 (ISBN 978-1-107-15630-2); "updated first edition" preprint dated 30 Dec 2019 | MFoR (screw theory / PoE view, numerical IK); RMC (Ch. 11) | Ch. 2 Configuration Space; Ch. 3 Rigid-Body Motions; Ch. 4 Forward Kinematics; Ch. 5 Velocity Kinematics and Statics; Ch. 6 Inverse Kinematics; Ch. 8 Dynamics of Open Chains; Ch. 9 Trajectory Generation; Ch. 11 Robot Control | **Yes**, the authors' free preprint: https://hades.mech.northwestern.edu/images/2/25/MR-v2.pdf ; book home: https://hades.mech.northwestern.edu/index.php/Modern_Robotics |
| 3 | M. W. Spong, S. Hutchinson and M. Vidyasagar, *Robot Modeling and Control* | 2nd ed., Wiley, 2020 (608 pp.) | MFoR and RMC | Part I, The Geometry of Robots (Ch. 2 Rigid Motions; Ch. 3 Forward Kinematics; then velocity and inverse kinematics). Part II, Dynamics and Motion Planning. Part III, Control of Manipulators (Ch. 8 Independent Joint Control; Ch. 9 Nonlinear and Multivariable Control; Ch. 10 Force Control; Ch. 11 Vision-Based Control). Part IV, Control of Underactuated Systems. Appendix C, Lyapunov Stability. | No (commercial). https://www.wiley.com/en-us/Robot+Modeling+and+Control,+2nd+Edition-p-9781119524045 |
| 4 | B. Siciliano, L. Sciavicco, L. Villani and G. Oriolo, *Robotics: Modelling, Planning and Control* | Springer, 2009 | MFoR and RMC (often used as the main control text) | Ch. 2 Kinematics; Ch. 3 Differential Kinematics and Statics; Ch. 4 Trajectory Planning; Ch. 7 Dynamics; **Ch. 8 Motion Control** (pp. 303–361); **Ch. 9 Force Control** (pp. 363–405); Ch. 10 Visual Servoing (pp. 407–467) | No (commercial; often available through university SpringerLink subscriptions). https://link.springer.com/book/10.1007/978-1-84628-642-1 |
| 5 | B. Siciliano, L. Villani, G. Oriolo and A. De Luca, *Foundations of Robotics* | Springer, 2025 (successor text now used in Sapienza's *Robotics 2*) | MFoR and RMC | n/a (new; chapter mapping not yet verified) | No. https://link.springer.com/book/10.1007/978-3-031-85523-8 |
| 6 | P. Corke, *Robotics, Vision and Control: Fundamental Algorithms in Python* | 3rd ed., Springer Tracts in Advanced Robotics vol. 146, 2023 (a MATLAB version also exists) | MFoR (orientation, quaternions, Python practice); RMC (Ch. 9) | Ch. 2 Representing Position and Orientation (quaternions); Ch. 3 Time and Motion; Ch. 7 Robot Arm Kinematics; Ch. 8 Manipulator Velocity; Ch. 9 Dynamics and Control | No (commercial). Book site and code: https://petercorke.com/rvc3p/home/ ; Springer: https://link.springer.com/book/10.1007/978-3-031-06469-2 |
| 7 | R. M. Murray, Z. Li and S. S. Sastry, *A Mathematical Introduction to Robotic Manipulation* | CRC Press, 1994 | MFoR (screw theory, PoE, Paden–Kahan subproblems); RMC (Ch. 4) | Ch. 2 Rigid Body Motion; Ch. 3 Manipulator Kinematics; Ch. 4 Robot Dynamics and Control; Ch. 5–6 multifingered hands; Ch. 7–8 nonholonomic systems; App. A Lie groups | **Yes.** Open-access listing at Taylor & Francis: https://www.taylorfrancis.com/books/oa-mono/10.1201/9781315136370/mathematical-introduction-robotic-manipulation-richard-murray-zexiang-li-shankar-sastry . The electronic edition notes that it is distributed from the author's site (http://www.cds.caltech.edu/~murray/mlswiki); a copy is hosted for UC Berkeley EECS C106A at https://pages.github.berkeley.edu/EECS-106/fa24-site/assets/MLS.pdf |
| 8 | R. Tedrake, *Robotic Manipulation: Perception, Planning, and Control* (MIT 6.4210 course notes) | Online, continuously updated | MFoR (kinematics in code), RMC (force/impedance control in simulation) | Online chapters | **Yes**: https://manipulation.mit.edu/ |
| 9 | R. Tedrake, *Underactuated Robotics* (MIT 6.8210 course notes) | Online, continuously updated | RMC (nonlinear, optimal and learning-based control) | Online chapters (pendulum, acrobot/cart-pole, LQR, trajectory optimisation, MPC, RL) | **Yes**: https://underactuated.mit.edu/ |
| 10 | Robotic Systems Lab, ETH Zürich, *Robot Dynamics* lecture notes | HS 2017 script | MFoR (kinematics, dynamics); RMC (operational-space and floating-base control) | Kinematics, differential kinematics, multibody dynamics, floating-base systems | **Yes**, as cited by third parties at https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf (the link itself was not opened during research; *unverified*) |
| 11 | D. K. Pratihar, *Fundamentals of Robotics* | Narosa | MFoR (Indian-syllabus style numericals) | Used as the text for the NPTEL course below | No (commercial) |

**Further reading for RMC (manipulator control monographs)**

| Book | Notes |
|------|-------|
| F. L. Lewis, D. M. Dawson and C. T. Abdallah, *Robot Manipulator Control: Theory and Practice*, 2nd ed., CRC Press, 2003 (ISBN 978-0-8247-4072-6) | Chapters: 1 Commercial Robot Manipulators; 2 Introduction to Control Theory; 3 Robot Dynamics; 4 Computed-Torque Control; 5 Robust Control; 6 Adaptive Control; 7 Advanced Control Techniques; 8 Neural Network Control; 9 Force Control; 10 Implementation and Software. A copy is hosted on the first author's group site: https://lewisgroup.uta.edu/FL%20books/Robot_Manipulator_Control_Theory_and_Practice_-_Frank_L.Lewis-%20small.pdf . Publisher: https://www.routledge.com/Robot-Manipulator-Control-Theory-and-Practice/Lewis-Dawson-Abdallah/p/book/9780824740726 |
| R. Kelly, V. Santibáñez and A. Loría, *Control of Robot Manipulators in Joint Space* (Springer) | Standard reference for PD + gravity compensation and Lyapunov proofs. *Unverified* (the listing was not found during research). |

---

## 2. Control-engineering textbooks (background for RMC)

| Book | Edition | Why it matters for RMC | Key chapters (verify against your edition) | Free legal copy |
|------|---------|------------------------|--------------------------------------------|-----------------|
| K. Ogata, *Modern Control Engineering* | 5th ed., Pearson (ISBN 978-0-13-615673-4). Listed in the IIIT-A course syllabus. | Transfer functions, transient response, root locus, frequency response, PID, state space | Ch. 1 Introduction; Ch. 2–4 Mathematical modelling; Ch. 5 Transient and steady-state response; Ch. 6 Root locus; Ch. 7 Frequency response; Ch. 8 PID controllers; Ch. 9–10 State-space analysis and design. *Chapter titles are from memory; Pearson confirms root locus in Ch. 6.* | No. https://www.pearson.com/en-us/subject-catalog/p/modern-control-engineering/P200000003713/9780136156734 |
| N. S. Nise, *Control Systems Engineering* | 8th ed., Wiley, 2019/2024 (ISBN 978-1-119-72140-6 for the 2024 printing) | The most readable first control text; good for second-order specifications and root-locus PID design | Modelling in the frequency and time domains; time response; stability; steady-state error; root locus and design via root locus; frequency response; design via state space; digital control. *Chapter numbers not verified.* | No. |
| G. F. Franklin, J. D. Powell and A. Emami-Naeini, *Feedback Control of Dynamic Systems* | Pearson | Appears among the IIIT-A syllabus references (the author list was truncated in the source) | — | No. |
| J.-J. E. Slotine and W. Li, *Applied Nonlinear Control* | Prentice Hall, 1991 | Lyapunov theory, feedback linearisation, sliding-mode control, adaptive control, and robot trajectory control (Ch. 9) | Part I: 1 Introduction; 2 Phase Plane Analysis; 3 Fundamentals of Lyapunov Theory; 4 Advanced Stability Theory; 5 Describing Function Analysis. Part II: 6 Feedback Linearization; 7 Sliding Control; 8 Adaptive Control; 9 Control of Multi-Input Physical Systems | No. |
| H. K. Khalil, *Nonlinear Systems* | 3rd ed., Prentice Hall, 2002 | Rigorous Lyapunov stability, passivity, feedback linearisation, sliding mode | Ch. 4 Lyapunov Stability; Ch. 13 Feedback Linearization; Ch. 14 Nonlinear Design Tools (14.1 sliding mode); passivity material expanded in this edition | No. Author page: https://www.egr.msu.edu/~khalil/NonlinearSystems/ |
| R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction* | 2nd ed., MIT Press, 2018 | Background for the RL-for-control part of RMC | — | **Yes**, author-hosted: http://incompleteideas.net/book/the-book-2nd.html |

**Key survey papers (RMC)**

- P. F. Hokayem and M. W. Spong, "Bilateral teleoperation: An historical survey," *Automatica* 42(12):2035–2057, 2006. DOI: https://doi.org/10.1016/j.automatica.2006.06.027
- J. Kober, J. A. Bagnell and J. Peters, "Reinforcement learning in robotics: A survey," *Int. J. Robotics Research* 32(11):1238–1274, 2013. DOI: https://doi.org/10.1177/0278364913495721 . Author PDF: https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Kober_IJRR_2013.pdf

---

## 3. Video lectures

### 3.1 Prof. G. C. Nandi (IIIT Allahabad)

Prof. Gora Chand Nandi is Head of the Centre of Intelligent Robotics at IIIT-A. His homepage is https://sites.google.com/iiita.ac.in/gcnandi/home and his study-materials page, https://sites.google.com/iiita.ac.in/gcnandi/study-materials, lists *Mathematical Foundation of Robotics (July 2024–December 2024)*.

| Playlist / video | URL | Status |
|---|---|---|
| (1) *Mathematical Foundation of Robotics* (29 lessons; first course, listed on his channel as a *Course*). L2–L25 cover the mid-sem topics (FK/IK, spatial and rotational transforms, Euler angles, gimbal lock, quaternions, D-H, PUMA 560); L26–L29 cover dynamics. | https://youtube.com/playlist?list=PLFWlHcAOSQboRkC6DtxKIqXmErq0oc7S3 | **Verified from the instructor's notes** (the URL is printed in the IIIT-A mid-sem notes, p. 58). YouTube itself could not be opened from the research environment. |
| A single video, "Mathematical Foundation of Robotics, Lecture-4 (Basics of Forward and Inverse Dynamics)". Its description links to a Google Sites course page. | https://www.youtube.com/watch?v=dJA9kR1MVXM | Seen in search results. That it belongs to Prof. Nandi's series is *probable but unverified*. |
| (2) **Mathematical Foundation of Robotics — Execution Level …** (12 videos; second course, RMC-oriented). Identified from a screenshot of Prof. Nandi's YouTube playlists page supplied by the repository owner; the full title is truncated there. | — | **Title and size confirmed by the owner; URL to be added.** |
| (3) **Introduction to Humanoid and Collaborative Robots** (16 videos; third course: case studies, including the Baxter collaborative robot and humanoids). Identified from the same screenshot. The video "Introduction to Humanoid Robots, Lecture-1" (https://www.youtube.com/watch?v=dNqMxiBGIio) is probably its first lecture. | — | **Title and size confirmed by the owner; URL to be added.** |

> **Action for maintainers:** please add the exact URLs of playlists (2) and (3) from Prof. Nandi's YouTube channel or his study-materials page once confirmed.

### 3.2 Other lecture series

| Series | Course fit | URL | Status |
|---|---|---|---|
| ThatsEngineering, *Robotics* (15 videos; follows Craig's notation; good for numericals) | MFoR | https://youtube.com/playlist?list=PLZaGkBteQK3HQFSWDM7-yRQWTd86DeDIY | Verified from the instructor's notes (p. 58); not opened on YouTube |
| Northwestern, *Modern Robotics, All Videos* (Lynch; short videos per book section) | MFoR + RMC (Ch. 11) | https://www.youtube.com/playlist?list=PLggLP4f-rq02vX0OQQ5vrCxbJrzamYDfx ; index by chapter: https://hades.mech.northwestern.edu/index.php/Modern_Robotics_Videos | Verified (search result) |
| Stanford CS223A, *Introduction to Robotics* (O. Khatib, 2008) | MFoR + RMC (operational-space and force control) | YouTube: https://www.youtube.com/playlist?list=PL65CC0384A1798ADF ; Stanford Engineering Everywhere: https://see.stanford.edu/course/cs223a | Verified (search result) |
| Sapienza, *Robotics 1*, Prof. A. De Luca | MFoR | https://www.youtube.com/playlist?list=PLAQopGWlIcyaqDBW1zSKx7lHfVcOmWSWt | Verified (search result) |
| Sapienza, *Robotics 2*, Prof. A. De Luca (dynamics, adaptive and robust control, impedance, hybrid force/velocity) | RMC | https://www.youtube.com/playlist?list=PLAQopGWlIcya6LnIF83QlJTqvpYmJXnDm ; course page: https://www.diag.uniroma1.it/deluca/rob2_en.php | Verified (search result) |
| MIT 6.832/6.8210, *Underactuated Robotics* (Tedrake) | RMC (nonlinear, optimal, learning control) | Example lecture: https://www.youtube.com/watch?v=pZ5-yS4X0wg ; notes: https://underactuated.mit.edu/ | Individual videos verified; no single playlist URL confirmed |
| University of Michigan, *ROB 501: Mathematics for Robotics* (J. Grizzle) | MFoR (linear-algebra background) | https://www.youtube.com/playlist?list=PLdPQZLMHRjDIzO99aE7yAtdOHSVHMXfYH | Verified (search result) |
| S. Brunton, *Control Bootcamp* | RMC (state space, controllability, LQR, observers, robustness) | Overview video: https://www.youtube.com/watch?v=Pi7l8mMjYVE | Individual videos verified; playlist URL not confirmed |
| B. Douglas / MathWorks, *MATLAB Tech Talks*: Understanding PID Control, Understanding MPC, Robust Control, Learning-Based Control | RMC | https://www.mathworks.com/videos/tech-talks/controls.html ; channel: https://www.youtube.com/user/ControlLectures/playlists | Verified (search result) |
| QUT Robot Academy (P. Corke; 200+ short lessons) | MFoR | Described at https://robohub.org/the-robot-academy-an-open-online-robotics-education-resource ; the site is robotacademy.net.au | Verified (secondary source) |

### 3.3 NPTEL / SWAYAM (India)

| Course | Instructor | URL | Fit |
|---|---|---|---|
| *Robotics* (IIT Kharagpur): kinematics, dynamics, control schemes, robot intelligence, motion planning, biped walking | Prof. D. K. Pratihar | https://nptel.ac.in/courses/112105249 | MFoR + intro RMC |
| *Robotics: Basics and Selected Advanced Concepts* (IISc): D-H, direct/inverse kinematics of serial and parallel robots, workspace, Lagrangian dynamics, linear and model-based control | Prof. Ashitava Ghosal | https://nptel.ac.in/courses/112108298 | MFoR + RMC |
| *Introduction to Robotics* (IIT Madras): includes computed-torque control and reinforcement learning | Prof. T. Asokan (with B. Ravindran, K. Vasudevan) | https://nptel.ac.in/courses/107106090 | MFoR + RMC |
| *Mechanics and Control of Robotic Manipulators* (IIT Palakkad; 8 weeks; MATLAB/Simulink exercises) | Prof. Santhakumar Mohan | https://nptel.ac.in/courses/112106304 ; run page: https://onlinecourses.nptel.ac.in/noc23_me79/preview ; sample lecture "Introduction to robot motion control": https://www.youtube.com/watch?v=x9LAwYTZptM | RMC |
| *Wheeled Mobile Robots* | Prof. S. Mohan and Prof. T. Asokan | https://nptel.ac.in/courses/112106298 | Extension |

---

## 4. Open courses at top universities

| University | Code | Title | Textbook(s) | URL |
|---|---|---|---|---|
| Stanford | CS223A / ME320 | Introduction to Robotics | Winter 2026: Siciliano, Sciavicco, Villani & Oriolo (2009). Historically Khatib's own course reader with Craig. | https://cs.stanford.edu/group/manips/teaching/cs223a/ |
| Stanford | CS327A | Advanced Robotic Manipulation (operational-space formulation, unified motion/force control, redundancy, whole-body control) | Papers / Khatib notes | https://khatib.stanford.edu/teaching/cs327a/index.html |
| Northwestern | ME 449 | Robotic Manipulation | Lynch & Park, *Modern Robotics* | https://hades.mech.northwestern.edu/index.php/ME_449_Robotic_Manipulation |
| Northwestern / Coursera | — | *Modern Robotics: Mechanics, Planning, and Control* Specialization (6 courses) | Lynch & Park | https://www.coursera.org/specializations/modernrobotics |
| Carnegie Mellon | 16-384 | Robot Kinematics and Dynamics (rigid-body motion, FK, Jacobian, IK, D-H, extended from 2D to 3D) | Spong, Hutchinson & Vidyasagar (one offering; *unverified for all terms*) | https://cmu-hkn.github.io/CMU-ECE-CS-Guide/electives/16384 |
| Carnegie Mellon | 16-711 | Kinematics, Dynamic Systems and Control (screw theory, Paden–Kahan, Lagrange, independent-joint control, feedback linearisation, operational-space and impedance control, LQR, MPC) | Murray–Li–Sastry (recommended); Lynch & Park | https://www.cs.cmu.edu/~cga/kdc/description.html ; https://www.cs.cmu.edu/~hgeyer/Teaching_16-711.html |
| Carnegie Mellon | 16-745 | Optimal Control and Reinforcement Learning (LQR, DP, convex MPC, iLQR/DDP, LQG, RL) | Lecture notes | https://optimalcontrol.ri.cmu.edu/ |
| MIT | 2.12 | Introduction to Robotics (planar/spatial kinematics, multibody dynamics, control design, actuators, sensors) | H. Asada's lecture notes | https://web.mit.edu/2.12/ ; OCW (Fall 2005): https://ocw.mit.edu/courses/2-12-introduction-to-robotics-fall-2005/pages/syllabus/ |
| MIT | 6.4210 | Robotic Manipulation | Tedrake, online notes | https://manipulation.mit.edu/ |
| MIT | 6.8210 | Underactuated Robotics | Tedrake, online notes | https://underactuated.mit.edu/ |
| ETH Zürich | 151-0851 | Robot Dynamics | RSL lecture notes | Lecture script (third-party copy of HS 2017 notes): https://www.scribd.com/document/434126638/RD-HS2017script ; unofficial MATLAB exercises (Fall 2020): https://github.com/ivanalberico/Robot-Dynamics-ETH |
| UC Berkeley | EECS C106A | Introduction to Robotics (FK/IK, Jacobian, force relations, dynamics, control, vision) | Murray, Li & Sastry | https://www2.eecs.berkeley.edu/Courses/EECSC106A/ |
| UC Berkeley | EECS C106B | Robotic Manipulation and Interaction (grasping, nonholonomic planning and control, SLAM) | Murray, Li & Sastry | https://www2.eecs.berkeley.edu/Courses/EECSC106B |
| Georgia Tech | ECE 4560 | Introduction to Automation and Robotics (manipulator kinematics, dynamics, control) | Not confirmed | https://syllabus.gatech.edu/node/21010 |
| Georgia Tech | ME 4451 | Robotics | Siciliano, Sciavicco & Villani (recommended) | https://syllabus.gatech.edu/sites/default/files/2026-04/ME-4451-Syllabus_template-2026.pdf |
| TU München | (ICS) | Multi-sensory Based Robot Dynamic Manipulation (D-H, Euler–Lagrange, model-based and operational-space control) | Corke; Craig (recommended) | https://ce.cit.tum.de/ics/teaching/winter-term-2022-23/multi-sensory-based-robot-dynamic-manipulation |
| Univ. of Illinois | ME 446 / ECE 489 / SE 422 | Robot Dynamics and Control (independent-joint control, inverse dynamics, force/impedance; hands-on lab) | Spong, Hutchinson & Vidyasagar; Lynch & Park (optional) | https://courses.grainger.illinois.edu/ME446/sp2021 |
| Politecnico di Milano | — | Control of Industrial Robots (computed-torque feedforward, PD + gravity, inverse dynamics, robust and adaptive control, impedance/admittance, hybrid force/position) | — | https://rocco.faculty.polimi.it/cir/index.html |
| Sapienza Univ. of Rome | Robotics 1 / Robotics 2 | Robotics 1 (kinematics, trajectory planning); Robotics 2 (dynamics, adaptive and robust control, compliance/impedance, hybrid force/velocity) | Siciliano et al.; now *Foundations of Robotics* (2025) | https://www.diag.uniroma1.it/deluca/rob2_en.php |
| IIIT Allahabad | IT 4020 | Robot Motion Control (dynamics-based modelling, trajectory planning, master–slave, PD/PID, computed torque, nonlinear control, sensors) | Ogata; Craig; Siciliano et al.; Lynch & Park; Corke; Franklin et al. (as listed in the syllabus document) | https://aaa.iiita.ac.in/PDF1/M.Tech%20Curriculum%202025.pdf (not opened; content known from search snippets only) |
| IIIT Allahabad | IT 4030 | Foundation(s) of Robotics / Mathematical Foundations of Robotics | Craig; Lynch & Park; Corke (per the instructor's notes) | https://aaa.iiita.ac.in/PDF1/M.Tech%20Curriculum%202025.pdf |

---

## 5. Open-source software and notebooks

| Tool | What it is for | URL | Install |
|---|---|---|---|
| Modern Robotics code library | Reference implementations of every algorithm in Lynch & Park (Python, MATLAB, Mathematica) | Homepage: http://modernrobotics.org/ | `pip install modern-robotics` |
| Robotics Toolbox for Python (P. Corke) | D-H and ETS robot models, FK/IK, Jacobians, dynamics, trajectories; ships many standard arms (e.g. PUMA 560, Panda) | https://github.com/petercorke/robotics-toolbox-python ; docs: https://petercorke.github.io/robotics-toolbox-python/ | `pip install roboticstoolbox-python` |
| Spatial Math Toolbox for Python | SO(3)/SE(3), quaternions, twists, Euler/RPY conversions | https://github.com/rai-opensource/spatialmath-python ; docs: https://rai-opensource.github.io/spatialmath-python/ | `pip install spatialmath-python` |
| RVC3 notebooks | Jupyter notebooks for every chapter of *Robotics, Vision and Control* 3rd ed. | https://github.com/petercorke/RVC3-python | `pip install rvc3python` |
| Swift | Browser-based 3D visualiser used by the Robotics Toolbox | https://github.com/jhavl/swift | `pip install swift-sim` |
| Pinocchio | Fast rigid-body dynamics (RNEA, ABA, CRBA) with analytical derivatives | https://github.com/stack-of-tasks/pinocchio ; docs: http://stack-of-tasks.github.io/pinocchio/ | `pip install pin` or `conda install pinocchio -c conda-forge`. **Note:** the PyPI package named `pinocchio` is an unrelated project. |
| Drake (MIT / Toyota Research Institute) | Multibody dynamics, optimisation, controller design; used by MIT 6.4210/6.8210 | https://drake.mit.edu ; tutorials README: https://github.com/RobotLocomotion/drake/blob/master/tutorials/README.md | `pip install drake` |
| `manipulation`, `underactuated` | Companion Python packages for Tedrake's two online books | https://manipulation.mit.edu/ ; https://underactuated.mit.edu/ | `pip install manipulation` / `pip install underactuated` |
| MuJoCo (Google DeepMind) | Physics simulator for contact-rich control and RL; official Colab tutorials are linked from the README | https://github.com/google-deepmind/mujoco ; docs: https://mujoco.readthedocs.io/ | `pip install mujoco` |
| dm_control | Control-suite environments built on MuJoCo | https://github.com/google-deepmind/dm_control | `pip install dm-control` |
| PyBullet | Lightweight physics simulation for robotics and RL | https://github.com/bulletphysics/bullet3 | `pip install pybullet` |
| Python Control Systems Library | Transfer functions, root locus, Bode, state space, LQR (RMC background) | https://python-control.org ; https://github.com/python-control/python-control | `pip install control` |
| CasADi | Algorithmic differentiation and nonlinear optimisation (MPC, trajectory optimisation) | http://casadi.org ; https://github.com/casadi/casadi | `pip install casadi` |
| do-mpc | Model predictive control and moving-horizon estimation toolbox | https://www.do-mpc.com | `pip install do-mpc` |
| Gymnasium | Standard RL environment API | https://gymnasium.farama.org ; https://github.com/Farama-Foundation/Gymnasium | `pip install gymnasium` |
| Stable-Baselines3 | Reference RL algorithms (PPO, SAC, TD3, ...) | https://github.com/DLR-RM/stable-baselines3 | `pip install stable-baselines3` |
| SymPy | Symbolic derivation of D-H transforms, Jacobians and Lagrangian dynamics | https://sympy.org | `pip install sympy` |

**Notebook collections and course repositories**

| Repository | Content |
|---|---|
| https://github.com/dynamics-simulation-16-715/lecture-notebooks-2021 | CMU 16-715 *Advanced Robot Dynamics and Simulation* lecture notebooks (archived) |
| https://www.github.com/boisgera/control-engineering-with-python | *Control Engineering with Python*: dynamical systems, Lagrangian mechanics, mobile-robot control (runs on Binder) |
| https://github.com/TheProjectsGuy/RDC21-EC4.401 | IIIT Hyderabad *Robotics: Dynamics and Control* student materials (kinematics, Jacobians, dynamics) |
| https://www.github.com/jotaraul/jupyter-notebooks-for-robotics-courses | Mobile-robotics notebooks (University of Málaga) |
| https://github.com/alessandro-giusti/teaching-notebooks | Notebooks for teaching robotics and machine learning |

---

## 6. Online interactive resources

| Resource | Use | URL |
|---|---|---|
| Modern Robotics wiki (book, videos, code, errata, Coursera links) | Self-study companion for Lynch & Park | https://hades.mech.northwestern.edu/index.php/Modern_Robotics |
| Modern Robotics book pages at Northwestern | Per-chapter resources | https://modernrobotics.northwestern.edu/nu-gm-book-resource/ |
| IIT Kharagpur Virtual Lab, *Mechanisms and Robotics* | Browser experiments: FK and IK of the PUMA 560, Movemaster, KGP 50, VAL robot teaching | https://mr-iitkgp.vlabs.ac.in/ ; experiment list: https://mr-iitkgp.vlabs.ac.in/List%20of%20experiments.html |
| IIT Kharagpur Virtual Lab, *Kinematics and Dynamics of Mechanisms* | Mechanism kinematics background | https://kdm-iitkgp.vlabs.ac.in/ |
| MIT Robotic Manipulation, interactive notes (Meshcat/Deepnote examples) | Interactive kinematics, IK, force control | https://manipulation.mit.edu/ |
| MIT Underactuated Robotics, interactive notes | Pendulum, cart-pole, LQR, trajectory optimisation, RL | https://underactuated.mit.edu/ |
| Drake tutorials (Jupyter, view without an account) | Multibody plant, systems, optimisation | https://github.com/RobotLocomotion/drake/blob/master/tutorials/README.md |
| MATLAB Tech Talks, Control Systems | Intuition for PID, MPC, robust and learning-based control | https://www.mathworks.com/videos/tech-talks/controls.html |

---

*Last researched: November 2026. Report broken links or missing resources by opening an issue.*
