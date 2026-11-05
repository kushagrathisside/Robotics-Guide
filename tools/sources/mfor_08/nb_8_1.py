import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common8 import *
from tools.nbbuild import md, code, SETUP, header, nav, build

SEC = "8.1"
FN, TITLE = FILES[SEC]
prev, nxt = prev_next(SEC)

cells = [
header(COURSE, CHAPTER, SEC, TITLE, f"{CH_DIR}/{FN}", [
    ("Primary reading", "Craig, 3rd ed., Ch. 3 (link parameters, frame assignment, the modified convention) and Ch. 4 (solvability)"),
    ("Parallel reading", "Siciliano et al., Ch. 2–3 (kinematics; redundant manipulators) · Lynch & Park, Ch. 4–6 · Spong et al., Ch. 3 · Corke, *RVC*, Ch. 7"),
    ("Lectures", f"{vid('Frame assignment', 'TE')} · {vid('D-H tables', 'TE')} · {vid('Lecture 17', 'Nandi')} · Prof. Nandi's third playlist, *Introduction to Humanoid and Collaborative Robots* (referred to by title only)"),
    ("Prerequisites", "[2.3 Homogeneous Transformations](../02-spatial-descriptions-and-transformations/2.3-homogeneous-transformations.ipynb); the Denavit–Hartenberg recipe of Chapter 3 (recalled below)"),
    ("Study time", "about 3 hours"),
], r"""
A six-joint arm can bring its tool to any pose in its workspace, but it has no freedom left over: for a given pose
the joint angles are fixed up to a handful of discrete alternatives. **Collaborative arms** — the Baxter and Sawyer of
Rethink Robotics, the Franka Emika Panda, the KUKA LBR iiwa — add a *seventh* joint, and with it a continuous family of
joint configurations for every tool pose: the elbow can move while the hand stays put. This notebook models a
Baxter-style arm with the modified Denavit–Hartenberg table of the course examinations, checks that table and its
link transforms symbolically, and explains the redundancy. It then reads the kinematic structure of three real
seven-joint arms (Sawyer, Panda, iiwa) from their MuJoCo models, derives their D-H tables automatically, states
honestly where D-H frames and the model's own frames differ, and moves each arm along its self-motion with the
tool pose held fixed."""),

md(r"""
## Learning objectives

After working through this notebook you should be able to:

1. Explain why seven joints give a one-parameter family of solutions for a six-dimensional task, and state the dimension of the null space of the Jacobian.
2. Read a modified D-H table with a **joint offset** (the $\theta_2 + 90^\circ$ of a Baxter-style arm), write each link transform, and verify the algebra symbolically.
3. Compute the forward kinematics of a seven-joint arm, including the pose of the elbow relative to the shoulder, and check it on a planar special case.
4. Compute the self-motion direction (the null vector of $J$), move the arm along it with the tool pose fixed, and define the **elbow swivel angle**.
5. Read the joint axes of a real arm from a MuJoCo model, decide whether its shoulder and wrist are spherical, and derive a D-H table from the axes.
6. Compare a D-H forward kinematics with a simulator's, explain the constant frame offsets that remain, and find the range of self-motion that joint limits allow by numerical optimisation.

## Contents

1. [Why seven joints?](#1.-Why-seven-joints?)
2. [A Baxter-style arm: the modified D-H table](#2.-A-Baxter-style-arm:-the-modified-D-H-table)
3. [Forward kinematics, symbolically and numerically](#3.-Forward-kinematics,-symbolically-and-numerically)
4. [Redundancy: the null space and the elbow swivel](#4.-Redundancy:-the-null-space-and-the-elbow-swivel)
5. [Three real seven-joint arms in MuJoCo](#5.-Three-real-seven-joint-arms-in-MuJoCo)
6. [Reading the kinematic structure from a model](#6.-Reading-the-kinematic-structure-from-a-model)
7. [Deriving a D-H table automatically — and the frame offsets that remain](#7.-Deriving-a-D-H-table-automatically-—-and-the-frame-offsets-that-remain)
8. [Self-motion of the real arms by numerical optimisation](#8.-Self-motion-of-the-real-arms-by-numerical-optimisation)
9. [Worked examples](#9.-Worked-examples)
10. [Summary, common mistakes and exercises](#10.-Summary)
"""),

code(SETUP + r'''
import sympy as sp
import mujoco
from rgkit.transforms import rotx, rotz, make_T, inv_T, transl, is_transform
from rgkit.kinematics import fk_dh, fk_tool
from rgkit import mujoco_utils as mju, robots
from rgkit import case_studies as cs
'''),

md(r"""
## 1. Why seven joints?

A rigid body in space has **six** degrees of freedom — three for the position of a point and three for the
orientation — so the pose of a tool is a point in the six-dimensional group $SE(3)$. An arm with $n$ joints maps its
$n$ joint variables onto that space. If $n=6$ and the arm is away from a singularity, the map is locally one-to-one:
a small change of the tool pose demands one definite small change of the joints, and, as Chapter 4 shows, a given pose
is reached by a *finite* number of joint vectors (at most eight for an arm with a spherical wrist, at most sixteen for
a general six-joint arm). If $n=7$ the map goes from a seven-dimensional space to a six-dimensional one; at a generic
configuration its Jacobian $J(\mathbf q)\in\mathbb R^{6\times 7}$ has rank $6$, so its **null space** has dimension
$n-6=1$. Joint velocities in the null space,

$$
\dot{\mathbf q}=s\,\mathbf n(\mathbf q),\qquad J(\mathbf q)\,\mathbf n(\mathbf q)=\mathbf 0,
$$

move the arm without moving the tool at all. Following them traces a curve of configurations that all reach the same
pose, the **self-motion manifold**; it is one-dimensional here. The arm is said to be *kinematically redundant* for the
six-dimensional task, with **degree of redundancy** $r=n-6=1$.

> **Redundancy is relative to the task.** A seven-joint arm is redundant for a six-dimensional pose task ($r=1$), more
> redundant for a three-dimensional position task ($r=4$), and not redundant at all for a task that constrains seven
> quantities. A six-joint arm drawing on a table (a two-dimensional task) is redundant too. The word describes the
> difference between the number of joints and the dimension of what we ask of them.

The extra freedom is useful. It lets the controller keep the elbow clear of an obstacle or of the person standing
next to the robot, steer the joints away from their limits, avoid singularities, or choose between postures that
require different torques. A human arm is commonly modelled with seven degrees of freedom — three at the shoulder, one
at the elbow, one for forearm rotation and two at the wrist — and the commercial collaborative arms follow that pattern: a
shoulder with three intersecting or nearly intersecting axes, an elbow, and a wrist with three. The price is that the
inverse-kinematics problem no longer has an answer: it has a *family* of answers, and some **additional rule** must pick
one of them (Chapter 5 treats the pseudoinverse, which is one such rule).
"""),

md(r"""
## 2. A Baxter-style arm: the modified D-H table

The Baxter robot of Rethink Robotics has two seven-joint arms. In the course papers its left arm is modelled with
the following table, written in the **modified (Craig) convention**,
${}^{i-1}_{\;\;i}T = R_X(\alpha_{i-1})\,D_X(a_{i-1})\,R_Z(\theta_i)\,D_Z(d_i)$, with the row
$(\alpha_{i-1},\ a_{i-1},\ d_i,\ \theta_i)$ listing the quantities of the link $i-1\to i$:

| $i$ | $\alpha_{i-1}$ | $a_{i-1}$ | $d_i$ | $\theta_i$ | joint |
|:-:|:-:|:-:|:-:|:-:|:--|
| 1 | $0$ | $0$ | $0$ | $\theta_1$ | S0 — shoulder roll |
| 2 | $-90^\circ$ | $L_1$ | $0$ | $\theta_2+90^\circ$ | S1 — shoulder pitch |
| 3 | $90^\circ$ | $0$ | $L_2$ | $\theta_3$ | E0 — elbow roll |
| 4 | $-90^\circ$ | $L_3$ | $0$ | $\theta_4$ | E1 — elbow pitch |
| 5 | $90^\circ$ | $0$ | $L_4$ | $\theta_5$ | W0 — wrist roll |
| 6 | $-90^\circ$ | $L_5$ | $0$ | $\theta_6$ | W1 — wrist pitch |
| 7 | $90^\circ$ | $0$ | $0$ | $\theta_7$ | W2 — wrist roll |

The base frame $\{BL\}$ sits $L_0$ below the shoulder, so ${}^{BL}_{\;\;0}T = D_Z(L_0)$, and the gripper frame $\{GL\}$ is a distance
$L_6$ beyond the last joint along $\hat Z_7$, ${}^{7}_{GL}T = D_Z(L_6)$. The pattern of the table is the pattern of the arm:
alternating **roll** joints (rotation about the link's own axis: S0, E0, W0, W2) and **pitch** joints (rotation about an axis across the
link: S1, E1, W1), with link twists of $\pm90^\circ$ between them. The lengths $L_2$ (upper arm), $L_4$ (forearm) and $L_6$
(wrist to gripper) are the long dimensions and appear as the **offsets $d$** along the roll axes; the short lengths $L_1$, $L_3$,
$L_5$ are the small **sideways offsets** at the shoulder, elbow and wrist and appear as $a$.

**The joint offset.** Row 2 has $\theta_2+90^\circ$ instead of $\theta_2$. This is a *joint offset*: the D-H frames are placed
where the recipe puts them, but the manufacturer's zero for the joint angle ($\theta_2=0$) is chosen with the arm pointing straight out.
The constant $90^\circ$ reconciles the two, and it is added *inside* the link transform, so
$\cos(\theta_2+90^\circ)=-\sin\theta_2$ and $\sin(\theta_2+90^\circ)=\cos\theta_2$ replace the usual entries.

The cell below builds the seven link transforms symbolically from the general formula and compares them with the
matrices worked out by hand from the table (each entry is simple because $\alpha=0,\pm90^\circ$). The arithmetic is verified
by `simplify` returning the zero matrix.
"""),

code(r'''
q = sp.symbols("q1:8")                       # joint variables theta_1..theta_7
L = sp.symbols("L1:7", positive=True)        # L1..L6; L0 is a pure translation of the base
half = sp.pi / 2

def link_T(alpha, a, d, theta):
    """Craig's modified link transform Rx(alpha) Dx(a) Rz(theta) Dz(d), symbolic."""
    ca, sa, ct, st = sp.cos(alpha), sp.sin(alpha), sp.cos(theta), sp.sin(theta)
    return sp.Matrix([[ct, -st, 0, a],
                      [st * ca, ct * ca, -sa, -sa * d],
                      [st * sa, ct * sa, ca, ca * d],
                      [0, 0, 0, 1]])

rows = [(0, 0, 0, q[0]),
        (-half, L[0], 0, q[1] + half),       # joint offset of +90 degrees
        (half, 0, L[1], q[2]),
        (-half, L[2], 0, q[3]),
        (half, 0, L[3], q[4]),
        (-half, L[4], 0, q[5]),
        (half, 0, 0, q[6])]
T_link = [sp.simplify(link_T(*r)) for r in rows]

c, s = [sp.cos(x) for x in q], [sp.sin(x) for x in q]
# the same seven matrices, written by hand from the table
by_hand = [
    sp.Matrix([[c[0], -s[0], 0, 0], [s[0], c[0], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    sp.Matrix([[-s[1], -c[1], 0, L[0]], [0, 0, 1, 0], [-c[1], s[1], 0, 0], [0, 0, 0, 1]]),
    sp.Matrix([[c[2], -s[2], 0, 0], [0, 0, -1, -L[1]], [s[2], c[2], 0, 0], [0, 0, 0, 1]]),
    sp.Matrix([[c[3], -s[3], 0, L[2]], [0, 0, 1, 0], [-s[3], -c[3], 0, 0], [0, 0, 0, 1]]),
    sp.Matrix([[c[4], -s[4], 0, 0], [0, 0, -1, -L[3]], [s[4], c[4], 0, 0], [0, 0, 0, 1]]),
    sp.Matrix([[c[5], -s[5], 0, L[4]], [0, 0, 1, 0], [-s[5], -c[5], 0, 0], [0, 0, 0, 1]]),
    sp.Matrix([[c[6], -s[6], 0, 0], [0, 0, -1, 0], [s[6], c[6], 0, 0], [0, 0, 0, 1]]),
]
for i, (T, H) in enumerate(zip(T_link, by_hand), start=1):
    print(f"link {i-1}->{i}: general formula == hand-written matrix:", sp.simplify(T - H) == sp.zeros(4, 4))
display(Math(r"{}^{1}_{2}T=" + sp.latex(T_link[1]) + r"\qquad {}^{2}_{3}T=" + sp.latex(T_link[2])))
'''),

md(r"""
The second matrix shows the effect of the offset: the first row is $[-s_2,\ -c_2,\ 0,\ L_1]$. Without the offset it would be
$[c_2,\ -s_2,\ 0,\ L_1]$, and the arm's "straight out" posture would correspond to $\theta_2=-90^\circ$ instead of $0$. The
third column of ${}^{2}_{3}T$ is $(0,-1,0)^{\mathsf T}$: the roll axis $\hat Z_3$ of the elbow points along $-\hat Y_2$, which is why the upper-arm length
$L_2$ appears as $-L_2$ in the $y$ entry of the translation column.
"""),

md(r"""
## 3. Forward kinematics, symbolically and numerically

The pose of the gripper is the product of the nine transforms,

$$
{}^{BL}_{GL}T={}^{BL}_{\;\;0}T\;{}^{0}_{1}T\;{}^{1}_{2}T\;{}^{2}_{3}T\;{}^{3}_{4}T\;{}^{4}_{5}T\;{}^{5}_{6}T\;{}^{6}_{7}T\;{}^{7}_{GL}T .
$$

A question the course papers ask is the pose of the **elbow relative to the shoulder**. The shoulder pitch frame is $\{2\}$ and the
elbow pitch frame is $\{4\}$, so the answer is ${}^{2}_{4}T={}^{2}_{3}T\,{}^{3}_{4}T$. Multiplying the two matrices above gives

$$
{}^{2}_{4}T=\begin{bmatrix} c_3c_4 & -c_3s_4 & -s_3 & L_3c_3\\ s_4 & c_4 & 0 & -L_2\\ s_3c_4 & -s_3s_4 & c_3 & L_3s_3\\ 0&0&0&1\end{bmatrix},
$$

and the interpretation is direct. The elbow lies a distance $L_2$ down the *negative* $\hat Y_2$ axis (the upper arm), displaced by the
small offset $L_3$ along $\hat X_3$ turned through the elbow-roll angle $\theta_3$: the offset swings about the upper-arm axis as the
elbow rolls. The elbow pitch $\theta_4$ affects only the orientation of $\{4\}$, not its origin. The cell confirms the product symbolically.
"""),

code(r'''
T24 = sp.simplify(T_link[2] * T_link[3])
T24_text = sp.Matrix([[c[2]*c[3], -c[2]*s[3], -s[2], L[2]*c[2]],
                      [s[3], c[3], 0, -L[1]],
                      [s[2]*c[3], -s[2]*s[3], c[2], L[2]*s[2]],
                      [0, 0, 0, 1]])
print("2_4T equals the matrix quoted in the text:", sp.simplify(T24 - T24_text) == sp.zeros(4, 4))

# full symbolic gripper pose, then lambdify
L0_s = sp.Symbol("L0", positive=True)
T_BL_GL = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, L0_s], [0, 0, 0, 1]])
for T in T_link:
    T_BL_GL = T_BL_GL * T
T_BL_GL = T_BL_GL * sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, L[5]], [0, 0, 0, 1]])
fk_sym = sp.lambdify((q, (L0_s,) + L), T_BL_GL, "numpy")

# representative dimensions of a human-scale arm (rounded, illustrative; consult the manufacturer for a real robot)
dims = dict(L0=0.27, L1=0.07, L2=0.36, L3=0.07, L4=0.37, L5=0.01, L6=0.37)
h = np.pi / 2
def baxter_table(L1, L2, L3, L4, L5):
    """Numeric modified-D-H table of the Baxter-style arm (rows: alpha, a, d, theta, kind)."""
    return [(0, 0, 0, 0, "R"), (-h, L1, 0, h, "R"), (h, 0, L2, 0, "R"), (-h, L3, 0, 0, "R"),
            (h, 0, L4, 0, "R"), (-h, L5, 0, 0, "R"), (h, 0, 0, 0, "R")]
table = baxter_table(dims["L1"], dims["L2"], dims["L3"], dims["L4"], dims["L5"])
T_base = transl(0, 0, dims["L0"]); T_flange_tool = transl(0, 0, dims["L6"])
fk_num = lambda qq: T_base @ fk_dh(table, qq)[-1] @ T_flange_tool

rng = np.random.default_rng(0)
err = max(np.abs(fk_num(qq) - np.array(fk_sym(qq, tuple(dims.values())), float)).max() for qq in rng.uniform(-2, 2, (50, 7)))
print("symbolic product vs numeric rgkit D-H code, max difference over 50 random poses:", f"{err:.1e}")

T0 = fk_num(np.zeros(7))
display(Math(matrix_to_latex(T0, 3, r"{}^{BL}_{GL}T(\mathbf 0)")))
'''),

md(r"""
At the zero configuration the gripper is at $(L_1+L_2+L_4+L_6,\ 0,\ L_0-L_3-L_5)$: the arm points straight forward and is
slightly lowered by the elbow and wrist offsets. The next cell draws the arm at this configuration and at a second posture, using
only the frame origins and a few frames; the frame of the elbow $\{4\}$ and the wrist $\{6\}$ are marked.
"""),

code(r'''
def skeleton(table, qq, T_base=T_base, tool=T_flange_tool):
    """Origins of base, frames {0}..{7} and the tool, with the full list of frame poses."""
    Ts = [T_base @ T for T in fk_dh(table, qq)] + [T_base @ fk_dh(table, qq)[-1] @ tool]
    return Ts

def draw_arm(ax, Ts, color="#444", frames=(2, 4, 6, 8), flen=0.08, label=True):
    P = np.array([T[:3, 3] for T in Ts])
    P = np.vstack([[0, 0, 0], P])
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-o", color=color, lw=4, ms=5, alpha=0.7, solid_capstyle="round")
    names = {2: "{S1}", 4: "{E1}", 6: "{W1}", 8: "tool"}
    for i in frames:
        draw_frame(ax, Ts[i], length=flen, name=None)
        if label:
            ax.text(*(Ts[i][:3, 3] + [0.02, 0.02, 0.04]), names.get(i, str(i)), fontsize=9)

fig = plt.figure(figsize=(10.5, 4.8))
for k, (qq, title) in enumerate([(np.zeros(7), r"$\mathbf q=\mathbf 0$ (arm straight out)"),
                                  (np.array([deg(30), deg(-30), deg(45), deg(60), deg(0), deg(30), deg(0)]), r"$\mathbf q=(30,-30,45,60,0,30,0)^\circ$")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    new_axes3d(ax, lim=[(-0.1, 1.0), (-0.5, 0.5), (0.0, 0.9)], title=title, elev=20, azim=-60)
    draw_arm(ax, skeleton(table, qq))
plt.tight_layout(); plt.show()
'''),

md(r"""
**A check on a special case.** If the three roll joints and the last wrist joint are held at zero ($\theta_1=\theta_3=\theta_5=\theta_6=\theta_7=0$), only the
pitch joints S1 and E1 move, and the whole arm swings in the vertical $x$–$z$ plane. Symbolic multiplication of the chain then gives
(with $\theta_{24}=\theta_2+\theta_4$)

$$
x=L_1+L_2\cos\theta_2-L_3\sin\theta_2+(L_4+L_6)\cos\theta_{24}-L_5\sin\theta_{24},\qquad
z=L_0-L_2\sin\theta_2-L_3\cos\theta_2-(L_4+L_6)\sin\theta_{24}-L_5\cos\theta_{24},
$$

with $y=0$. This is a planar two-link arm whose links $L_2$ and $L_4+L_6$ are each tilted by a small angle: the offset
$L_3$ (or $L_5$) at the end of a link rotates the link's effective direction. The notebook confirms the formula numerically in Example 8.1.3.
"""),

md(r"""
## 4. Redundancy: the null space and the elbow swivel

With seven joints the geometric Jacobian is $6\times7$. For the D-H chain it is built column by column from the frames:
column $i$ is $[\hat{\mathbf z}_i\times(\mathbf p_e-\mathbf p_i);\ \hat{\mathbf z}_i]$ for a revolute joint, with $\hat{\mathbf z}_i$ the axis of frame $\{i\}$
and $\mathbf p_i$ its origin. The function `rgkit.case_studies.jacobian_dh` does exactly that, and Chapter 5 derives it.
The singular-value decomposition $J=U\Sigma V^{\mathsf T}$ exposes the structure: six positive singular values and a
seventh, zero direction — the last column of $V$ — which spans the null space.

Moving along that direction by small steps, and removing the second-order drift with a few damped-least-squares
iterations, follows the self-motion curve. A convenient scalar to describe where on the curve the arm is the
**swivel angle** $\phi$: draw the line from the shoulder to the wrist, take the circle that the elbow could trace around that line (the elbow stays at fixed
distances from both end points, which is the geometry of a triangle with fixed sides), and measure the angle of the elbow around that circle from the upward direction.
If the shoulder (the three axes S0, S1, E0) and the wrist (W0, W1, W2) were exactly spherical joints, the self-motion would *be* a rotation of the elbow about the
shoulder–wrist line, and the swivel angle would be the parameter of the motion.
"""),

code(r'''
q_ref = np.array([0.3, -0.4, 0.5, 0.9, -0.3, 0.6, 0.1])
J = cs.jacobian_dh(table, q_ref, tool=T_flange_tool)       # geometric Jacobian at the gripper, base frame
U, sv, Vt = np.linalg.svd(J)
print("Jacobian shape:", J.shape, "  singular values:", sv.round(4), "  rank:", np.linalg.matrix_rank(J))
n = cs.null_vector(J)
print("null vector n =", n.round(4), "   |J n| =", f"{np.linalg.norm(J @ n):.1e}")

# self-motion: 150 small steps along the null space, tool pose held fixed
fk_g = lambda qq: T_base @ fk_dh(table, qq)[-1] @ T_flange_tool          # shoulder-to-gripper chain
jac_g = lambda qq: cs.jacobian_dh(table, qq, tool=T_flange_tool, base=T_base)
path = cs.self_motion(fk_g, jac_g, q_ref, n_steps=150, ds=0.03)
T_start = fk_g(q_ref)
drift = max(np.linalg.norm(cs.pose_error(fk_g(qq), T_start)) for qq in path)
print(f"largest pose error along the self-motion path: {drift:.1e}   (joint-space length {150*0.03:.1f} rad)")

def swivel(table_, qq, T_base=T_base):
    Ts = [T_base @ T for T in fk_dh(table_, qq)]
    return cs.swivel_angle(Ts[2][:3, 3], Ts[4][:3, 3], Ts[6][:3, 3])     # shoulder S1, elbow E1, wrist W1

phi, radius, d_sw = np.array([swivel(table, qq) for qq in path]).T
phi = np.unwrap(phi)
# the same arm with the three small offsets removed: an exact spherical-revolute-spherical arm
table0 = baxter_table(0.0, dims["L2"], 0.0, dims["L4"], 0.0)
fk0 = lambda qq: T_base @ fk_dh(table0, qq)[-1] @ T_flange_tool
jac0 = lambda qq: cs.jacobian_dh(table0, qq, tool=T_flange_tool, base=T_base)
path0 = cs.self_motion(fk0, jac0, q_ref, n_steps=150, ds=0.03)
phi0, radius0, _ = np.array([swivel(table0, qq) for qq in path0]).T
phi0 = np.unwrap(phi0)
print(f"elbow-circle radius with offsets    : {radius.min():.4f} .. {radius.max():.4f} m   (varies by {100*np.ptp(radius)/radius.mean():.1f} %)")
print(f"elbow-circle radius without offsets : {radius0.min():.6f} .. {radius0.max():.6f} m   (constant)")
'''),

code(r'''
# the self-motion of the Baxter-style arm: several postures with the SAME gripper pose, and the swivel geometry
fig = plt.figure(figsize=(11, 4.6))
ax = fig.add_subplot(1, 2, 1, projection="3d")
new_axes3d(ax, lim=[(-0.1, 1.0), (-0.5, 0.5), (0.0, 0.9)], title="Same gripper pose, five elbow positions", elev=20, azim=-55)
cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
for k, col in zip(np.linspace(0, 150, 5).astype(int), cols):
    draw_arm(ax, skeleton(table, path[k]), color=col, frames=(8,) if k else (2, 4, 6, 8), label=(k == 0))
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(np.rad2deg(phi), 1000 * radius, color="#1f77b4", label="with offsets $L_1,L_3,L_5$")
ax2.plot(np.rad2deg(phi0), 1000 * radius0, color="#ff7f0e", label="offsets removed")
ax2.set_xlabel("swivel angle $\\phi$ (deg)"); ax2.set_ylabel("distance of the elbow from the shoulder–wrist line (mm)")
ax2.set_title("The elbow moves on a circle only if the shoulder and wrist are spherical", fontsize=10)
ax2.legend(); ax2.set_ylim(0, 200)
plt.tight_layout(); plt.show()
'''),

md(r"""
Two lessons. First, the null-space motion works: 150 steps change the joint vector by a large amount (the joint-space path length is
$4.5$ rad) while the gripper pose changes by less than $10^{-15}$. Second, the geometry is only *approximately* a circle for the
Baxter-style arm, because the small offsets $L_1$, $L_3$, $L_5$ spoil the spherical symmetry; with the offsets removed (an arm whose
shoulder axes intersect in one point, and whose wrist axes intersect in another) the elbow stays on an exact circle of
constant radius. Real arms sit between the two cases, as Section 8 shows.

The animation shows the self-motion as a film: the elbow swings, the hand does not move.
"""),

code(r'''
fig = plt.figure(figsize=(5.4, 5.0))
ax = fig.add_subplot(111, projection="3d")
sel = np.concatenate([np.arange(0, 151, 5), np.arange(150, -1, -5)])

def update(k):
    ax.cla()
    new_axes3d(ax, lim=[(-0.1, 1.0), (-0.5, 0.5), (0.0, 0.9)], title=rf"swivel $\phi$ = {np.rad2deg(phi[k]):6.1f}°   (hand fixed)", elev=20, azim=-55)
    draw_arm(ax, skeleton(table, path[k]), color="#d62728", frames=(2, 4, 6, 8), label=False, flen=0.07)

animate(fig, update, sel, interval=70, gif="assets/8.1-self-motion.gif", n_keyframes=4)
'''),

md(r"""
The explorer below lets you choose a point along the self-motion. The sliders show the swivel angle and the corresponding joint vector; the
tool pose in the title does not change.
"""),

code(r'''
def explore(k=0):
    fig = plt.figure(figsize=(10.5, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    new_axes3d(ax, lim=[(-0.1, 1.0), (-0.5, 0.5), (0.0, 0.9)], elev=20, azim=-55,
               title=rf"$\phi={np.rad2deg(phi[k]):.0f}^\circ$,  tool at {np.round(fk_g(path[k])[:3, 3], 3)}")
    draw_arm(ax, skeleton(table, path[k]), color="#d62728", frames=(2, 4, 6, 8))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(np.arange(1, 8), np.rad2deg(path[k]), color="#4c72b0")
    ax2.set_xlabel("joint $i$"); ax2.set_ylabel(r"$\theta_i$ (deg)"); ax2.set_ylim(-200, 200)
    ax2.set_title("joint angles at this point of the self-motion", fontsize=10)
    plt.tight_layout(); plt.show()

interact(explore, k=widgets.IntSlider(min=0, max=150, step=10, value=60, description="step"))
'''),

md(r"""
## 5. Three real seven-joint arms in MuJoCo

The same structure — a spherical-ish shoulder, an elbow, a spherical-ish wrist — appears in the commercial collaborative arms. The models
below come from MuJoCo Menagerie (see [docs/MUJOCO.md](../../docs/MUJOCO.md)): the Rethink Robotics **Sawyer** (the single-arm successor in
Baxter's family), the **Franka Emika Panda** and the **KUKA LBR iiwa 14**. In each model, joint axes and link frames are the
manufacturer's description of the robot, so everything we compute from them is a statement about the real machine's *design*. (The Panda model
carries a two-finger gripper; we use only the seven arm joints.)
"""),

code(r'''
ARM = {   # name -> (last arm body, tool body or site, kind)
    "sawyer": ("right_l6", "attachment_site", "site"),
    "panda": ("link7", "hand", "body"),
    "iiwa14": ("link7", "attachment_site", "site"),
}
SHOULDER, ELBOW, WRIST = 1, 3, 5          # which joint axis (0-based) marks the shoulder, elbow and wrist points

class Arm7:
    """A seven-joint MuJoCo arm: forward kinematics of the tool, joint limits and the swivel geometry."""
    def __init__(self, name):
        self.name = name
        self.m, self.d = robots.load_robot(name, scene="light")
        last, tool, kind = ARM[name]
        self.jids = cs.chain_joint_ids(self.m, last)
        self.n = len(self.jids)
        self.lo, self.hi = self.m.jnt_range[self.jids, 0], self.m.jnt_range[self.jids, 1]
        self.tool_id = (self.m.site(tool) if kind == "site" else self.m.body(tool)).id
        self.kind = kind
        self.q_home = self.m.key_qpos[0][:self.n].copy()

    def kinematics(self, q):
        self.d.qpos[:self.n] = q
        mujoco.mj_kinematics(self.m, self.d)

    def fk(self, q):
        self.kinematics(q)
        if self.kind == "site":
            return make_T(self.d.site_xmat[self.tool_id].reshape(3, 3), self.d.site_xpos[self.tool_id])
        return make_T(self.d.xmat[self.tool_id].reshape(3, 3), self.d.xpos[self.tool_id])

    def swivel(self, q):
        self.kinematics(q)
        a = [self.d.xanchor[self.jids[i]].copy() for i in (SHOULDER, ELBOW, WRIST)]
        return cs.swivel_angle(*a)

    def render(self, q, camera, width=420, height=360):
        self.d.qpos[:self.n] = q
        mujoco.mj_forward(self.m, self.d)
        return mju.render(self.m, self.d, camera=camera, width=width, height=height)

arms = {k: Arm7(k) for k in ARM}
for k, a in arms.items():
    print(f"{a.m.model.name if hasattr(a.m, 'model') else k:8s}  arm joints: {a.n}   total qpos: {a.m.nq}   actuators: {a.m.nu}   "
          f"mass of the whole model: {a.m.body_subtreemass[1]:.1f} kg")
for k in arms:
    print(robots.describe(k))
'''),

code(r'''
cams = {"sawyer": dict(azimuth=140, elevation=-15, distance=2.0, lookat=(0, 0, 0.6)),
        "panda": dict(azimuth=140, elevation=-15, distance=1.9, lookat=(0.2, 0, 0.5)),
        "iiwa14": dict(azimuth=140, elevation=-15, distance=1.9, lookat=(0.2, 0, 0.55))}
imgs = [arms[k].render(arms[k].q_home, cams[k]) for k in arms]
filmstrip(imgs, ["Rethink Sawyer", "Franka Emika Panda", "KUKA LBR iiwa 14"])
'''),

md(r"""
## 6. Reading the kinematic structure from a model

A MuJoCo model stores every body's pose relative to its parent and, for each joint, an axis in the frame of its body. Putting the arm at the
**zero configuration** ($\mathbf q=\mathbf 0$) and calling `mj_forward` makes the simulator compute, for every joint, the **axis direction**
$\boldsymbol\omega_i$ and a **point on the axis** $\mathbf p_i$ (the joint anchor) in the world frame. These $2n$ vectors *are* the kinematic description of the arm
(they are the screw axes of the product-of-exponentials formula of Chapter 3); D-H parameters are one way of encoding them.

The first thing the axes reveal is the arm's family. Three axes that meet in one point form a **spherical joint**, in the kinematic sense. An arm with a spherical
shoulder and a spherical wrist separated by an elbow has the simplest geometry for inverse kinematics, and the
residual below measures how close three consecutive axes come to meeting: it is the root-mean-square distance of the three lines from their best common point (zero if
they intersect).
"""),

code(r'''
def axis_label(w):
    i = int(np.argmax(np.abs(w)))
    return ("+-"[w[i] < 0] + "xyz"[i]) if abs(w[i]) > 0.999 else "(" + ",".join(f"{x:.2f}" for x in w) + ")"

def meet_residual(om, pt):
    """RMS distance of the lines (om_k, pt_k) from their best common point."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for w, p in zip(om, pt):
        P = np.eye(3) - np.outer(w, w)
        A += P; b += P @ p
    x = np.linalg.solve(A, b)
    return float(np.sqrt(np.mean([np.sum((np.eye(3) - np.outer(w, w)) @ (x - p)**2) for w, p in zip(om, pt)])))

axes0 = {}
print(f"{'robot':8s} {'axes at q = 0 (world frame)':36s} {'shoulder 1-2-3':>15s} {'wrist 5-6-7':>12s}")
for k, a in arms.items():
    a.d.qpos[:] = 0; mujoco.mj_forward(a.m, a.d)
    om, pt = cs.joint_axes(a.m, a.d, a.jids)
    axes0[k] = (om.copy(), pt.copy())
    print(f"{k:8s} {' '.join(axis_label(w) for w in om):36s} {1000*meet_residual(om[:3], pt[:3]):12.2f} mm {1000*meet_residual(om[4:], pt[4:]):9.2f} mm")
'''),

md(r"""
The **iiwa** is the cleanest case: its shoulder axes and its wrist axes each intersect exactly (residual zero), so it is a true
spherical–revolute–spherical arm, and the elbow circle of Section 4 should be exact. The **Panda** also has a spherical shoulder, but
its wrist is offset (the seventh axis does not pass through the point where axes 5 and 6 meet). The **Sawyer**, like Baxter, has
small offsets at all three groups of joints, so neither group is exactly spherical. The sizes of the residuals (millimetres to tens of millimetres) are
the quantitative version of "approximately" in Section 4.
"""),

md(r"""
## 7. Deriving a D-H table automatically — and the frame offsets that remain

Given the axes at the zero configuration, Craig's frame-assignment recipe can be executed mechanically — `rgkit.case_studies.dh_from_axes`
does so. For every pair of consecutive axes it finds their **common normal**: the direction $\hat X_i$, the length $a_i$ (the distance between the
axes) and the twist $\alpha_i$ (the angle between them about $\hat X_i$); the origin $O_i$ is where the normal leaves axis $i$; $d_i$ and the offset of $\theta_i$ then follow
from where the previous normal arrived and from the angle between consecutive $\hat X$ axes. Frame $\{0\}$ is made to coincide with $\{1\}$ at zero, and the last
$\hat X$ is chosen so that the last joint offset vanishes, exactly as in the recipe. Two frames cannot be generated by the recipe because the recipe does not fix them:
the **base frame** (the table's frame $\{0\}$ differs from the model's world frame by a fixed transform ${}^{W}_{0}T$) and the **tool frame** (the last D-H frame $\{7\}$
differs from the model's tool frame by a fixed transform ${}^{7}_{\;T}T$). Both are constant, so they can be found at $\mathbf q=\mathbf 0$ and then
*tested* at other configurations; if the table is right the test holds for every $\mathbf q$:

$$
{}^{W}_{T}T(\mathbf q)\;=\;{}^{W}_{0}T\;\;{}^{0}_{7}T_{\text{D-H}}(\mathbf q)\;\;{}^{7}_{\;T}T .
$$
"""),

code(r'''
fits = {}
rng = np.random.default_rng(1)
for k, a in arms.items():
    om, pt = axes0[k]
    fit = cs.dh_from_axes(om, pt)
    T_model0 = a.fk(np.zeros(a.n))
    T_7T = inv_T(fit.T_base_0 @ fk_dh(fit.table, np.zeros(a.n))[-1]) @ T_model0     # fixed, found at q = 0
    errs = []
    for qq in rng.uniform(a.lo, a.hi, (200, a.n)):
        errs.append(np.abs(a.fk(qq) - fit.T_base_0 @ fk_dh(fit.table, qq)[-1] @ T_7T).max())
    fits[k] = (fit, T_7T)
    print(f"{k:7s} modified D-H table derived from the model  -> max |model FK - D-H FK| over 200 random poses: {max(errs):.1e}")

def show_table(fit):
    tex = r"\begin{array}{c|cccc} i & \alpha_{i-1} & a_{i-1}\,(\mathrm{m}) & d_i\,(\mathrm{m}) & \theta_i \text{ offset}\\ \hline "
    for i, (al, ai, di, th, _) in enumerate(fit.table, start=1):
        tex += f"{i} & {np.rad2deg(al):.0f}^\\circ & {ai:.4f} & {di:.4f} & {np.rad2deg(th):.0f}^\\circ \\\\ "
    return tex + r"\end{array}"
display(Markdown("**Panda** — derived table (modified convention):"))
display(Math(show_table(fits["panda"][0])))
display(Math(matrix_to_latex(fits["panda"][0].T_base_0, 3, r"{}^{W}_{0}T") + r",\qquad " + matrix_to_latex(fits["panda"][1], 3, r"{}^{7}_{\;T}T")))
'''),

md(r"""
The derived Panda table contains the numbers $0.316$, $0.0825$, $0.384$ and $0.088$ metres: the length of the upper arm, the
elbow offset (which appears twice, with the same size before and after the elbow), the length of the forearm and the wrist offset. The height
$0.333$ m of the shoulder above the base is absorbed in ${}^{W}_{0}T$. These are the link dimensions that manufacturers list, and anyone who
compares the table above with the one in Franka's documentation will find the same dimensions, though perhaps with different **signs and joint offsets** — a
D-H table is *not unique*: flipping the direction of $\hat X_i$ changes the sign of $a_i$ and adds $180^\circ$ to $\theta_i$, and when two axes
intersect the sign of $\hat X_i=\pm\hat Z_i\times\hat Z_{i+1}$ is a free choice. Our generator always picks the positive direction. What is **not** a matter of
convention is the pose the table produces, and that agrees with the simulator to rounding error.

**The frame offsets, honestly.** Three kinds of constant transform separate "the D-H description" from "the model's description":

1. the world-to-base offset ${}^{W}_{0}T$ above (here: the shoulder height and a rotation about $\hat Z$ by $180^\circ$, because the recipe sets $\hat X_0 = \hat Z_1\times\hat Z_2$);
2. the tool offset ${}^{7}_{\;T}T$ (flange, hand or site);
3. the offset of **every intermediate link**: the model's body frame for link $i$ is not the D-H frame $\{i\}$. The cell below measures it. It
is constant over $\mathbf q$ — the two frames are rigidly attached to the same link — and its size says how far the manufacturer's link frame is from the frame that
the recipe puts on that link.
"""),

code(r'''
def link_frame_offsets(name):
    a = arms[name]; fit, _ = fits[name]
    bodies = [a.m.jnt_bodyid[j] for j in a.jids]                 # the body that carries joint i = link i
    off = []
    rng = np.random.default_rng(5)
    for i, b in enumerate(bodies, start=1):
        C = []
        for qq in rng.uniform(a.lo, a.hi, (6, a.n)):
            a.kinematics(qq)
            T_model = make_T(a.d.xmat[b].reshape(3, 3), a.d.xpos[b])
            T_dh = fit.T_base_0 @ fk_dh(fit.table, qq)[i]
            C.append(inv_T(T_dh) @ T_model)
        spread = max(np.abs(c - C[0]).max() for c in C)
        Rn = C[0][:3, :3]
        ang = np.arccos(np.clip((np.trace(Rn) - 1) / 2, -1, 1))
        off.append((i, np.linalg.norm(C[0][:3, 3]), np.rad2deg(ang), spread))
    return off

print("Offset between the model's link frame and the D-H frame of the same link (Panda):")
print(f"{'link':>5s} {'|translation| (mm)':>20s} {'rotation (deg)':>16s} {'variation over q':>18s}")
for i, tr, ang, sp_ in link_frame_offsets("panda"):
    print(f"{i:5d} {1000*tr:20.1f} {ang:16.1f} {sp_:18.1e}")
'''),

md(r"""
The last column is zero to rounding error: each offset is a fixed property of the link, not of the configuration. The offsets themselves are
not zero: the manufacturer's link frames are chosen for the CAD model and the inertial data, not for the D-H recipe. This is the situation
whenever a D-H table is compared with a real controller or simulator — the *joint axes* agree (as they must), the *link frames* do not — and it is why a robot's
documentation lists its D-H parameters together with the base and flange frames to which they refer.

**An alternative that has no frame offsets.** The product-of-exponentials formula of Chapter 3 uses the axes directly:
${}^{W}_{T}T(\mathbf q)=e^{[\mathcal S_1]q_1}\cdots e^{[\mathcal S_7]q_7}\,M$, with $\mathcal S_i=(\boldsymbol\omega_i;\,-\boldsymbol\omega_i\times\mathbf p_i)$ and $M$ the tool pose at zero. It needs only the
data of the model's own frames.
"""),

code(r'''
for k, a in arms.items():
    om, pt = axes0[k]
    S = cs.screw_axes(om, pt); M0 = a.fk(np.zeros(a.n))
    e = max(np.abs(a.fk(qq) - cs.poe_fk(S, M0, qq)).max() for qq in np.random.default_rng(2).uniform(a.lo, a.hi, (100, a.n)))
    print(f"{k:7s} product of exponentials from the model's axes vs MuJoCo: max difference {e:.1e}")
'''),

md(r"""
## 8. Self-motion of the real arms by numerical optimisation

For the Baxter-style arm the self-motion was traced by integrating the null vector. Here we do something different, in order to see a second way to solve
the same problem and to honour the joint limits of the real machines. The tool pose is held fixed by **six equality constraints**
$\text{pose error}(\mathbf q)=\mathbf 0$ and the elbow is prescribed by a **seventh** constraint on the swivel angle,
$\phi(\mathbf q)=\phi^\ast$. Among the configurations that satisfy all seven we take the one nearest to the previous
configuration, i.e. we solve

$$
\min_{\mathbf q}\ \lVert\mathbf q-\mathbf q_{\text{prev}}\rVert^{2}\quad\text{subject to}\quad
\text{pose error}(\mathbf q)=\mathbf 0,\ \ \sin(\phi(\mathbf q)-\phi^\ast)=0,\ \ \mathbf q_{\min}\le\mathbf q\le\mathbf q_{\max},
$$

with SLSQP (`scipy.optimize.minimize`), for a sequence of target angles $\phi^\ast$ marching away from the starting value. The forward kinematics is
MuJoCo's. The march stops when no feasible configuration exists: the **swivel range** that the joint limits permit.
"""),

code(r'''
from scipy.optimize import minimize

START = {"panda": [0.3, 0.5, -0.2, -1.9, 0.3, 2.0, -0.6],
         "iiwa14": [0.3, 0.7, -0.2, -1.4, 0.3, 0.6, 0.1],
         "sawyer": [0.3, -1.1, 0.2, 2.0, 0.3, 0.7, 3.3]}

def swivel_march(a, q0, direction, step=np.deg2rad(8), max_steps=60):
    """March the swivel angle in steps, holding the tool pose, respecting joint limits. Returns lists of (phi, q, radius)."""
    T0 = a.fk(q0); phi0 = a.swivel(q0)[0]
    q, out = q0.copy(), []
    for k in range(1, max_steps):
        target = phi0 + direction * step * k
        cons = [{"type": "eq", "fun": lambda x: cs.pose_error(a.fk(x), T0)},
                {"type": "eq", "fun": lambda x, t=target: np.array([np.sin(a.swivel(x)[0] - t)])}]
        r = minimize(lambda x: np.sum((x - q) ** 2), q, constraints=cons, bounds=list(zip(a.lo, a.hi)),
                     method="SLSQP", options=dict(maxiter=300, ftol=1e-14))
        if not r.success or np.linalg.norm(cs.pose_error(a.fk(r.x), T0)) > 1e-7:
            break
        q = r.x
        out.append((np.rad2deg(a.swivel(q)[0] - np.rad2deg(0) - phi0 * 0), q.copy(), a.swivel(q)[1]))
    return out, T0, phi0

results = {}
for k, a in arms.items():
    q0 = np.array(START[k])
    assert np.all(q0 > a.lo) and np.all(q0 < a.hi)
    fwd, T0, phi0 = swivel_march(a, q0, +1)
    bwd, _, _ = swivel_march(a, q0, -1)
    seq = [(np.rad2deg(a.swivel(q0)[0]), q0, a.swivel(q0)[1])] + fwd + bwd
    seq.sort(key=lambda t: t[0])
    errs = [np.linalg.norm(cs.pose_error(a.fk(s[1]), T0)) for s in seq]
    results[k] = dict(seq=seq, T0=T0, err=max(errs))
    ph = [s[0] for s in seq]; rad = [s[2] for s in seq]
    print(f"{k:7s} swivel range {min(ph):7.1f}° .. {max(ph):6.1f}°  (width {max(ph)-min(ph):5.1f}°)   "
          f"elbow-circle radius {min(rad)*1000:5.0f}..{max(rad)*1000:5.0f} mm   max tool-pose error {max(errs):.1e}")
'''),

md(r"""
All three arms keep the tool pose to better than $10^{-7}$ while the elbow swings through a wide arc, and the *width* of the arc differs: the iiwa's elbow circle has an exactly constant radius,
and the offsets of the Panda and the Sawyer make their radii vary. How far the elbow can swivel is limited by joint limits: wherever one of the seven angles reaches its range, the march stops.
The figures show the joint angles of the Panda as functions of the swivel, then the Panda rendered at three swivel angles with a hand that does not move.
"""),

code(r'''
seq = results["panda"]["seq"]
phis = np.array([s[0] for s in seq]); Q = np.array([s[1] for s in seq])
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for i in range(7):
    axs[0].plot(phis, np.rad2deg(Q[:, i]), label=f"$q_{i+1}$")
axs[0].set_xlabel("swivel angle $\\phi$ (deg)"); axs[0].set_ylabel("joint angle (deg)"); axs[0].legend(ncol=4, fontsize=8)
axs[0].set_title("Panda: joint angles along the self-motion (tool pose fixed)", fontsize=10)
for k, col in zip(arms, ("#1f77b4", "#ff7f0e", "#2ca02c")):
    s = results[k]["seq"]
    axs[1].plot([t[0] for t in s], [1000 * t[2] for t in s], "o-", ms=3, color=col, label=k)
axs[1].set_xlabel("swivel angle $\\phi$ (deg)"); axs[1].set_ylabel("elbow-circle radius (mm)")
axs[1].set_title("Radius of the elbow circle along the self-motion", fontsize=10); axs[1].legend()
plt.tight_layout(); plt.show()

a = arms["panda"]
picks = [0, len(seq) // 2, len(seq) - 1]
cam = dict(azimuth=150, elevation=-15, distance=1.8, lookat=(0.25, 0, 0.5))
imgs = [a.render(seq[i][1], cam) for i in picks]
filmstrip(imgs, [f"Panda, swivel {seq[i][0]:.0f}°, hand at {np.round(a.fk(seq[i][1])[:3, 3], 3)}" for i in picks])
'''),

md(r"""
## 9. Worked examples

**Example 8.1.1 — the link transform with a joint offset.** For the Baxter-style table take $L_1=0.07$ m and $\theta_2=30^\circ$. Write
${}^{1}_{2}T$ and show that it differs from the transform without the offset.

*Solution.* With $\theta_2+90^\circ=120^\circ$: $\cos120^\circ=-0.5$, $\sin120^\circ=0.866$. The link transform
$R_X(-90^\circ)D_X(L_1)R_Z(120^\circ)$ has first row $[-\sin\theta_2,\ -\cos\theta_2,\ 0,\ L_1]=[-0.5,\ -0.866,\ 0,\ 0.07]$, second row $[0,0,1,0]$, third row
$[-\cos\theta_2,\ \sin\theta_2,\ 0,\ 0]=[-0.866,\ 0.5,\ 0,\ 0]$. Without the offset the first row would be $[\cos30^\circ,\ -\sin30^\circ,\ 0,\ L_1]=[0.866,\ -0.5,\ 0,\ 0.07]$.

**Example 8.1.2 — the elbow relative to the shoulder.** With $L_2=0.36$ m, $L_3=0.07$ m, $\theta_3=30^\circ$ and $\theta_4=60^\circ$ find the position of the elbow frame $\{4\}$ in the shoulder-pitch frame $\{2\}$.

*Solution.* From the last column of ${}^{2}_{4}T$: $(L_3\cos\theta_3,\ -L_2,\ L_3\sin\theta_3)=(0.0606,\ -0.36,\ 0.035)$ m. The elbow pitch $\theta_4$ does not enter the position.

**Example 8.1.3 — the planar special case.** For $\theta_2=20^\circ$, $\theta_4=50^\circ$ and the other joints zero, find the gripper position with the dimensions of the notebook and $L_0=0.27$ m.

*Solution.* With $\theta_{24}=70^\circ$: $x=L_1+L_2\cos20^\circ-L_3\sin20^\circ+(L_4+L_6)\cos70^\circ-L_5\sin70^\circ$ and
$z=L_0-L_2\sin20^\circ-L_3\cos20^\circ-(L_4+L_6)\sin70^\circ-L_5\cos70^\circ$. The code evaluates both and compares with the full product.

**Example 8.1.4 — a null-space velocity.** At $\mathbf q_{\text{ref}}=(0.3,-0.4,0.5,0.9,-0.3,0.6,0.1)$ rad, give the joints the velocity $\dot{\mathbf q}=0.2\,\mathbf n$ rad/s, where $\mathbf n$ is the
unit null vector. How fast does the gripper move, and how fast does the elbow frame $\{4\}$ move?

*Solution.* The gripper twist is $J\dot{\mathbf q}=0.2\,J\mathbf n=\mathbf 0$. The elbow, however, moves: its linear velocity is the upper $3\times7$ block of the Jacobian *of frame $\{4\}$*, i.e. built from the first four joints only,
applied to $\dot{\mathbf q}$. The code evaluates it.

**Example 8.1.5 — counting degrees of freedom of the real models.** Count the joints and the rank of the Jacobian of the Sawyer model at a generic configuration, and state the degree of redundancy for the six-dimensional pose task and for the three-dimensional position task.

*Solution.* Seven revolute joints, Jacobian of rank six, so $r=7-6=1$ for the pose task and $r=7-3=4$ for the position task.

**Example 8.1.6 — reading the Panda's link dimensions.** The Panda table derived in Section 7 contains $a=0.0825$ m. Find this number in the model's own body tree, without D-H.

*Solution.* The elbow body's offset from its parent, taken from the model, is the vector $(0.0825, 0, 0)$ m in the parent's frame: the same length, found where the XML puts it.
"""),

code(r'''
# Example 8.1.1
th2, L1v = deg(30), 0.07
T12 = np.array(sp.N(T_link[1].subs({q[1]: th2, L[0]: L1v})), float)
assert np.allclose(T12[0], [-0.5, -0.8660, 0, L1v], atol=1e-4) and np.allclose(T12[2], [-0.8660, 0.5, 0, 0], atol=1e-4)
T12_no_offset = fk_dh([(-h, L1v, 0, th2, "R")], [0.0])[-1]
assert np.allclose(T12_no_offset[0], [0.8660, -0.5, 0, L1v], atol=1e-4)
print("Ex 8.1.1: first row with offset", T12[0].round(3), " without offset", T12_no_offset[0].round(3), " ->  OK")

# Example 8.1.2
sub = {q[2]: deg(30), q[3]: deg(60), L[1]: 0.36, L[2]: 0.07}
p24 = np.array(sp.N(T24[:3, 3].subs(sub)), float)
assert np.allclose(p24, [0.07 * np.cos(deg(30)), -0.36, 0.07 * np.sin(deg(30))])
print("Ex 8.1.2: elbow in the shoulder-pitch frame:", p24.round(4), " ->  OK")

# Example 8.1.3: planar special case against the full product of 9 transforms
t2, t4 = deg(20), deg(50)
t24 = t2 + t4
d = dims
x = d["L1"] + d["L2"] * np.cos(t2) - d["L3"] * np.sin(t2) + (d["L4"] + d["L6"]) * np.cos(t24) - d["L5"] * np.sin(t24)
z = d["L0"] - d["L2"] * np.sin(t2) - d["L3"] * np.cos(t2) - (d["L4"] + d["L6"]) * np.sin(t24) - d["L5"] * np.cos(t24)
Tfull = fk_num(np.array([0, t2, 0, t4, 0, 0, 0.0]))
assert np.allclose(Tfull[:3, 3], [x, 0.0, z])
print(f"Ex 8.1.3: gripper at ({x:.4f}, 0, {z:.4f}) m  -> equals the full chain: True")

# Example 8.1.4
Jr = cs.jacobian_dh(table, q_ref, tool=T_flange_tool)
nref = cs.null_vector(Jr); qdot = 0.2 * nref
assert np.allclose(Jr @ qdot, 0, atol=1e-12)
Ts4 = fk_dh(table, q_ref)
z_i = [Ts4[i + 1][:3, 2] for i in range(4)]; p_i = [Ts4[i + 1][:3, 3] for i in range(4)]
Jelbow = np.column_stack([np.cross(z_i[i], p_i[3] - p_i[i]) for i in range(4)])          # frame {4} origin, first four joints
v_elbow = Jelbow @ qdot[:4]
print(f"Ex 8.1.4: gripper twist = {np.linalg.norm(Jr @ qdot):.1e};  elbow speed = {np.linalg.norm(v_elbow)*100:.2f} cm/s")
assert np.linalg.norm(v_elbow) > 1e-3

# Example 8.1.5
a = arms["sawyer"]
a.d.qpos[:a.n] = START["sawyer"]; mujoco.mj_forward(a.m, a.d)
Jm = cs.site_jacobian(a.m, a.d, "attachment_site")[:, :a.n]
assert a.n == 7 and np.linalg.matrix_rank(Jm) == 6
print(f"Ex 8.1.5: Sawyer joints = {a.n}, rank J = {np.linalg.matrix_rank(Jm)}, redundancy: pose task {a.n-6}, position task {a.n-3}")

# Example 8.1.6
pm = arms["panda"].m
print("Ex 8.1.6: Panda body offsets from the model (link, offset from parent):",
      {pm.body(i).name: pm.body_pos[i].round(4).tolist() for i in range(1, 9) if abs(pm.body_pos[i]).max() > 0.08 and abs(pm.body_pos[i]).max() < 0.1})
assert np.isclose(abs(fits["panda"][0].table[3][1]), 0.0825, atol=1e-4) and np.isclose(np.abs(pm.body_pos[pm.body("link4").id][0]), 0.0825)
'''),

md(r"""
## 10. Summary

| Result | Statement |
|:--|:--|
| Redundancy | $n$ joints, $m$-dimensional task: $r=n-m$. Seven joints and a pose task: $r=1$, null space of $J$ of dimension $1$ |
| Self-motion | $\dot{\mathbf q}=s\,\mathbf n(\mathbf q)$, $J\mathbf n=\mathbf 0$; follow it with small steps plus drift correction |
| Swivel angle | $\phi$ = angle of the elbow around the shoulder–wrist line; the elbow moves on an exact circle only if the shoulder and the wrist are spherical (the iiwa) |
| Joint offset | $\theta_2+90^\circ$ is added inside the link transform; it moves the zero of the joint, not the frames |
| Elbow w.r.t. shoulder | ${}^{2}_{4}T={}^{2}_{3}T\,{}^{3}_{4}T$; origin $(L_3c_3,\ -L_2,\ L_3s_3)$ |
| D-H from a model | common normals of consecutive axes give $a_i,\alpha_i$; the table is **not unique**; base, tool and every link frame differ from the model's by constants |
| Product of exponentials | uses the axes directly and has no frame-offset ambiguity |

### Common mistakes

* Forgetting the **joint offset** when the table says $\theta_2+90^\circ$, so that the zero posture is wrong while every link transform looks plausible.
* Treating a redundant arm as having "no inverse kinematics": it has *too many* solutions, and a secondary criterion must choose.
* Reading the *index* of a D-H table row as a *link number*: row $i$ is the transform from $\{i-1\}$ to $\{i\}$ and contains $\alpha_{i-1}$, $a_{i-1}$, $d_i$ and $\theta_i$ (modified convention).
* Assuming a robot's body frames are its D-H frames: the joint axes agree, the link frames generally do not.
* Using the elbow height instead of the swivel angle as the self-motion parameter near the planar posture, where the height is stationary and the parameter degenerates.

## Exercises

**8.1.1** Derive ${}^{1}_{2}T$ of the Baxter-style table from $R_X(-90^\circ)D_X(L_1)R_Z(\theta_2+90^\circ)D_Z(0)$ and show that it reduces to the matrix of Section 2.

<details><summary>Answer</summary>

$R_X(-90^\circ)=\begin{bmatrix}1&0&0\\0&0&1\\0&-1&0\end{bmatrix}$ and $R_Z(\theta_2+90^\circ)=\begin{bmatrix}-s_2&-c_2&0\\c_2&-s_2&0\\0&0&1\end{bmatrix}$, so the rotation block is
$\begin{bmatrix}-s_2&-c_2&0\\0&0&1\\-c_2&s_2&0\end{bmatrix}$; the translation is $R_X(-90^\circ)(L_1,0,0)^{\mathsf T}=(L_1,0,0)^{\mathsf T}$. The verification cell below confirms it symbolically.
</details>

**8.1.2** What is the degree of redundancy of (a) a seven-joint arm positioning a point in the plane, (b) a seven-joint arm holding a tool pose *and* a prescribed elbow height, (c) a planar 3R arm positioning its tip in the plane?

<details><summary>Answer</summary>

(a) the task has dimension 2 ($r=5$) if the arm is planar in the sense that its motion is confined to a plane; for a spatial arm the *plane* constraint is a restriction of the task, not of the arm, so be explicit about the task space; (b) $7-7=0$: the swivel angle is the missing constraint and the elbow height removes the freedom (up to finitely many solutions); (c) $3-2=1$.
</details>

**8.1.3** A planar 3R arm with link lengths $1,\ 0.8,\ 0.5$ m is at $\mathbf q=(30^\circ,\ 45^\circ,\ -60^\circ)$. Find the null vector of the $2\times3$ tip-position Jacobian and verify that a step of $0.05$ rad along it moves the tip by less than $2\times10^{-3}$ m.

<details><summary>Answer</summary>

The code in the verification cell finds $\mathbf n\approx(0.43,\,-0.82,\,0.38)^{\mathsf T}$ up to sign; the first-order tip motion is zero and the second-order displacement of a $0.05$ rad step is about $9\times10^{-4}$ m.
</details>

**8.1.4** Show that for the offset-free Baxter-style arm ($L_1=L_3=L_5=0$) the elbow frame $\{4\}$ stays at the distance $L_2$ from $\{2\}$ and at the distance $L_4$ from the wrist frame $\{6\}$ for every $\mathbf q$, and compute the radius of the elbow circle for $L_2=0.36$ m, $L_4=0.37$ m and $\theta_4=90^\circ$.

<details><summary>Answer</summary>

The origin of $\{4\}$ is at $(0,-L_2,0)$ in $\{2\}$ whatever $\theta_3,\theta_4$, and $\{6\}$ is $L_4$ from $\{4\}$ along $\hat Z_5$, so the triangle shoulder–elbow–wrist has constant sides $L_2$, $L_4$ and
$d_{sw}=\sqrt{L_2^2+L_4^2-2L_2L_4\cos\psi}$, with $\psi=180^\circ-\theta_4$ the interior elbow angle. The elbow is therefore a fixed point of a rigid triangle that rotates about the side $d_{sw}$; its radius is the height of the triangle over that side,
$r=L_2L_4\sin\psi/d_{sw}$. For $\theta_4=90^\circ$: $d_{sw}=\sqrt{0.36^2+0.37^2}=0.5162$ m and $r=0.36\cdot0.37/0.5162=0.2580$ m.
</details>

**8.1.5** Use the same formula for the iiwa model. Its upper arm and forearm lengths are the $d_3$ and $d_5$ of its derived table. Compute the radius of its elbow circle at the starting configuration of Section 8 and compare it with the value measured from the model.

<details><summary>Answer</summary>

The derived iiwa table gives $d_3=0.42$ m and $d_5=0.40$ m; with $\theta_4=-1.4$ rad the interior angle is $\psi=\pi-1.4$, $d_{sw}=0.5337$ m and $r=0.2639$ m, equal to the radius measured along the whole march (0.2639 m, constant).
</details>

**8.1.6** (Programming.) For the Panda find the swivel range (in degrees) available when the starting configuration is replaced by $\mathbf q=(0.3,\ 0.9,\ -0.2,\ -1.6,\ 0.3,\ 1.8,\ -0.6)$ rad. Is the range larger or smaller than from the configuration of Section 8? Explain in one sentence.

<details><summary>Answer</summary>

Run `swivel_march` for this start; the result is reported in the verification cell. The range depends on how close each joint sits to its limit when the march begins: a start nearer to a limit gives a smaller range on that side.
</details>
"""),

code(r'''
# verification of the exercise answers
T12_h = sp.Matrix([[-s[1], -c[1], 0, L[0]], [0, 0, 1, 0], [-c[1], s[1], 0, 0], [0, 0, 0, 1]])
assert sp.simplify(T_link[1] - T12_h) == sp.zeros(4, 4)
print("8.1.1 verified symbolically")

# 8.1.3
Ls = np.array([1.0, 0.8, 0.5]); q3r = np.deg2rad([30, 45, -60])
def tip_jac(qq):
    a = np.cumsum(qq)
    J = np.zeros((2, 3))
    for i in range(3):
        J[0, i] = -sum(Ls[j] * np.sin(a[j]) for j in range(i, 3)); J[1, i] = sum(Ls[j] * np.cos(a[j]) for j in range(i, 3))
    return J
n3 = cs.null_vector(tip_jac(q3r))
tip = lambda qq: np.array([np.sum(Ls * np.cos(np.cumsum(qq))), np.sum(Ls * np.sin(np.cumsum(qq)))])
disp = np.linalg.norm(tip(q3r + 0.05 * n3) - tip(q3r))
print("8.1.3 null vector", n3.round(2), " tip displacement for a 0.05 rad step:", f"{disp:.1e}", " (< 2e-3):", disp < 2e-3)
assert disp < 2e-3 and np.allclose(tip_jac(q3r) @ n3, 0, atol=1e-12)

# 8.1.4
L2v, L4v = 0.36, 0.37
t0 = baxter_table(0, L2v, 0, L4v, 0)
for qq in np.random.default_rng(3).uniform(-2, 2, (5, 7)):
    qq[3] = deg(90)
    Tq = fk_dh(t0, qq)
    assert np.isclose(np.linalg.norm(Tq[4][:3, 3] - Tq[2][:3, 3]), L2v) and np.isclose(np.linalg.norm(Tq[6][:3, 3] - Tq[4][:3, 3]), L4v)
    assert np.isclose(cs.swivel_angle(Tq[2][:3, 3], Tq[4][:3, 3], Tq[6][:3, 3])[1], L2v * L4v / np.hypot(L2v, L4v))
print("8.1.4 radius =", round(L2v * L4v / np.hypot(L2v, L4v), 4), "m  verified on 5 random configurations")

# 8.1.5
tab_iiwa = fits["iiwa14"][0].table
d3, d5 = tab_iiwa[2][2], tab_iiwa[4][2]
q4 = START["iiwa14"][3]; psi = np.pi - abs(q4)
dsw = np.sqrt(d3**2 + d5**2 - 2 * d3 * d5 * np.cos(psi)); r_formula = d3 * d5 * np.sin(psi) / dsw
r_model = np.mean([t[2] for t in results["iiwa14"]["seq"]])
print(f"8.1.5 d3={d3:.2f}, d5={d5:.2f}: formula r={r_formula:.4f} m, model r={r_model:.4f} m")
assert np.isclose(r_formula, r_model, atol=1e-5)

# 8.1.6
a = arms["panda"]; q_alt = np.array([0.3, 0.9, -0.2, -1.6, 0.3, 1.8, -0.6])
assert np.all(q_alt > a.lo) and np.all(q_alt < a.hi)
f2, _, _ = swivel_march(a, q_alt, +1); b2, _, _ = swivel_march(a, q_alt, -1)
ph_alt = [t[0] for t in f2 + b2] + [np.rad2deg(a.swivel(q_alt)[0])]
ph_ref = [s[0] for s in results["panda"]["seq"]]
print(f"8.1.6 swivel range from the alternative start: {max(ph_alt)-min(ph_alt):.0f} deg (Section 8 start: {max(ph_ref)-min(ph_ref):.0f} deg)")
'''),

md(r"""
## Further reading and lectures

* Craig, *Introduction to Robotics*, Ch. 3 (frame assignment and the modified convention) and Ch. 4 (solvability, including manipulators with more than six joints).
* Siciliano, Sciavicco, Villani and Oriolo, *Robotics: Modelling, Planning and Control*, Ch. 2–3: kinematics and the treatment of redundant manipulators.
* Lynch and Park, *Modern Robotics*, Ch. 4 (product of exponentials) and Ch. 6 (inverse kinematics, including redundant arms).
* Lectures: """ + f"{vid('Frame assignment', 'TE')}; {vid('Creating transformation', 'TE')}; {vid('Worked example: frame', 'TE')}; {vid('4.1.1', 'MR')}." + r"""
* Prof. G. C. Nandi's third playlist, *Introduction to Humanoid and Collaborative Robots*, treats the Baxter collaborative robot as a case study (referred to by title only).
* Robot models: Rethink Sawyer, Franka Emika Panda and KUKA LBR iiwa from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (see `robots.describe`).
"""),

nav(prev, nxt),
]

if __name__ == "__main__":
    build(cells, f"{CH_DIR}/{FN}")
