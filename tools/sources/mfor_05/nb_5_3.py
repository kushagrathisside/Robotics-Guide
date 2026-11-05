from ch5common import *

SEC = "5.3"
FN = "5.3-frames-and-the-analytical-jacobian.ipynb"
cells = [
header(COURSE, CHAPTER, SEC, "Jacobians in Other Frames and the Analytical Jacobian", f"{CH_DIR}/{FN}", [
    ("Primary reading", "Craig, 3rd ed., Ch. 5, §5.6 (Jacobians in different frames), §5.10 (Cartesian transformation of velocities and forces) and Ch. 2 §2.8 (Euler-angle representations); Appendix B for the angle-rate relations"),
    ("Parallel reading", "Lynch & Park §5.1 (space versus body Jacobian) · Siciliano et al. Ch. 3 (analytical Jacobian, representation singularities) · Spong et al., the chapter on the manipulator Jacobian (angular velocity and Euler angles) · Corke *RVC* Ch. 8"),
    ("Lectures", f"{V('5.1.1 Space Jacobian')} · {V('5.1.2 Body Jacobian')} · {V('Fixed and Euler angle representation')} (Euler-angle background) · {V('Ch. 5 Velocity kinematics and statics')}"),
    ("Prerequisites", "[5.2 The Manipulator Jacobian](5.2-the-manipulator-jacobian.ipynb)"),
    ("Study time", "about 3 hours"),
], r"""
A Jacobian is always a Jacobian *of a particular point, described in a particular frame*. Changing the frame of description multiplies it by
a block-diagonal rotation; changing the reference point mixes the angular rows into the linear rows; and moving to a different task
space — the rates of three Euler angles instead of the angular velocity vector — replaces $\boldsymbol\omega$ by $E^{-1}\boldsymbol\omega$, a
matrix that can fail to exist even when the robot is perfectly healthy. This notebook sets out those three operations, shows how they are used
(tool-frame Jacobians, tool offsets, camera frames, the $6\times6$ velocity transformation), contrasts the **geometric** with the **analytical**
Jacobian, and checks everything on a UR5e in MuJoCo."""),

md(r"""
## Learning objectives

1. Express a Jacobian in any frame, and explain why rank, determinant and singular configurations do not depend on that frame.
2. Move the Jacobian to another point of the tool, and use the $6\times6$ transformation that relates the velocities of two frames fixed to one body.
3. Derive the matrix $E$ that maps Euler-angle (or fixed-angle) rates to angular velocity for the common angle sets, and find its singularities.
4. Compute the analytical Jacobian $J_a=\mathrm{diag}(I,E^{-1})J$ and explain the difference between a *representation* singularity and a *kinematic* singularity.
5. Choose between the geometric and the analytical Jacobian for a given task.
6. Check the frame bookkeeping of a real six-axis arm against MuJoCo.

## Contents

1. [Why more than one frame?](#1.-Why-more-than-one-frame?)
2. [Changing the frame of description](#2.-Changing-the-frame-of-description)
3. [Changing the reference point](#3.-Changing-the-reference-point)
4. [The 6 × 6 velocity transformation between frames](#4.-The-6-%C3%97-6-velocity-transformation-between-frames)
5. [Geometric and analytical Jacobians](#5.-Geometric-and-analytical-Jacobians)
6. [Representation singularities](#6.-Representation-singularities)
7. [A real robot: frames and angle rates on the UR5e](#7.-A-real-robot:-frames-and-angle-rates-on-the-UR5e)
8. [Worked examples](#8.-Worked-examples)
9. [Summary, common mistakes and exercises](#9.-Summary)
"""),

code(SETUP + r'''
from rgkit.transforms import (rotx, roty, rotz, make_T, inv_T, skew, unskew, transl, euler_to_rot, rpy_to_rot, rot_to_rpy,
                              rot_to_euler_zyz, euler_zyz_to_rot)
from rgkit.kinematics import fk_dh, fk_tool, planar_fk
from rgkit import jacobian as jc
import sympy as sp
'''),

md(r"""
## 1. Why more than one frame?

Three different people may want the "same" Jacobian. The *programmer* of a pick-and-place routine thinks in the **base frame**: "move the tool 5 cm in $x$". The
*operator* teaching a path with a hand controller thinks in the **tool frame**: "advance along the tool's own $z$ axis". The *designer of a force controller* needs
velocities and forces in a **constraint frame** attached to a surface or to the grasped part. The motion is the same; the numbers differ because the axes differ.

There are three independent choices hidden in the phrase "the Jacobian":

| Choice | Options | Effect on the matrix |
|:--|:--|:--|
| Frame of description of $\mathbf v,\boldsymbol\omega$ | base $\{0\}$, tool $\{T\}$, any frame $\{B\}$ | left-multiplication by $\mathrm{diag}(R,R)$ |
| Reference point on the tool | tool origin, wrist, a point at the end of a held object | left-multiplication by $\begin{bmatrix}I&-[\mathbf r]_\times\\0&I\end{bmatrix}$ |
| Orientation coordinates | angular velocity $\boldsymbol\omega$, rates of Euler angles | left-multiplication of the lower block by $E^{-1}$ |

The first two are *exact, invertible* relabellings of the same physics. The third is also invertible **except** at special orientations. We take them in turn.
"""),

md(r"""
## 2. Changing the frame of description

A free vector is converted between frames by a rotation only (Notebook 2.3). Applying that rule to both the linear and the angular velocity of the same point,

$$
{}^{B}\!\begin{bmatrix}\mathbf v\\ \boldsymbol\omega\end{bmatrix}=\begin{bmatrix}{}^{B}_{A}R&0\\0&{}^{B}_{A}R\end{bmatrix}{}^{A}\!\begin{bmatrix}\mathbf v\\ \boldsymbol\omega\end{bmatrix}
\quad\Longrightarrow\quad
\boxed{{}^{B}J(\mathbf q)=\begin{bmatrix}{}^{B}_{A}R&0\\0&{}^{B}_{A}R\end{bmatrix}{}^{A}J(\mathbf q)}
$$

(Craig, §5.6, in the notation ${}^{i}J$ for the Jacobian described in frame $\{i\}$). In particular the Jacobian in the **tool frame**
$\{T\}$ is ${}^{T}J=\mathrm{diag}({}^{0}_{T}R^{\mathsf T},\,{}^{0}_{T}R^{\mathsf T})\,{}^{0}J$. Because the rotation is invertible, ${}^{B}J$ and ${}^{A}J$ have
the **same rank, the same singular configurations and (for a square $J$) the same determinant**. A singular configuration is a property of the arm and the chosen point, not of the frame.

**Example — the planar two-link arm in the tool frame.** From the symbolic propagation of Notebook 5.1, the tip velocity in the frame $\{3\}$ fixed to the last link is
${}^{3}\mathbf v=(L_1s_2\dot\theta_1,\ L_1c_2\dot\theta_1+L_2(\dot\theta_1+\dot\theta_2))$, i.e.

$$
{}^{3}J=\begin{bmatrix}L_1s_2&0\\L_1c_2+L_2&L_2\end{bmatrix},\qquad \det{}^{3}J=L_1L_2s_2,
$$

which is visibly simpler than ${}^{0}J$ — it depends on $\theta_2$ only, because from the tool's point of view the shoulder angle $\theta_1$ merely rotates the whole picture. The code verifies this by rotating ${}^{0}J$ with sympy.
"""),

code(r'''
t1, t2, L1s, L2s = sp.symbols("theta1 theta2 L1 L2", real=True)
J0 = sp.Matrix([[-L1s * sp.sin(t1) - L2s * sp.sin(t1 + t2), -L2s * sp.sin(t1 + t2)],
                [L1s * sp.cos(t1) + L2s * sp.cos(t1 + t2), L2s * sp.cos(t1 + t2)]])
R03 = sp.Matrix([[sp.cos(t1 + t2), -sp.sin(t1 + t2)], [sp.sin(t1 + t2), sp.cos(t1 + t2)]])        # planar ^0_3 R
J3 = sp.simplify(R03.T * J0)
display(Math(r"{}^{3}J = R^{\mathsf T}\,{}^{0}J = " + sp.latex(J3)))
display(Math(r"\det {}^{3}J=" + sp.latex(sp.simplify(J3.det())) + r"=\det {}^{0}J"))
'''),

md(r"""
**Numerical check on a spatial chain.** For a random six-joint chain we obtain ${}^{T}J$ in two independent ways: by rotating the base-frame Jacobian, and by Craig's propagation
(`jacobian_by_propagation`), which computes it directly in the tool frame without ever forming the base-frame matrix.
"""),

code(r'''
rng = np.random.default_rng(21)
tab = [(rng.uniform(-3, 3), rng.uniform(-.4, .4), rng.uniform(-.4, .4), rng.uniform(-3, 3), "R") for _ in range(6)]
tool = make_T(rotz(0.3) @ roty(0.2), [0.02, 0.0, 0.15])
q = rng.uniform(-1, 1, 6)
T = fk_tool(tab, q, tool=tool)
J_base = jc.jacobian_geometric(tab, q, tool=tool)
J_tool = jc.jacobian_in_frame(J_base, T[:3, :3].T)                    # ^T J = diag(R^T, R^T) ^0 J
print("tool-frame Jacobian = propagation result:", np.allclose(J_tool, jc.jacobian_by_propagation(tab, q, tool)))
print("same rank and determinant of the 6x6 matrix:", np.linalg.matrix_rank(J_base) == np.linalg.matrix_rank(J_tool),
      np.isclose(np.linalg.det(J_base), np.linalg.det(J_tool)))
print("singular values are NOT invariant (a rotation of the output only):",
      np.allclose(jc.singular_values(J_base), jc.singular_values(J_tool)))
'''),

md(r"""
The last line deserves a remark. A rotation $\mathrm{diag}(R,R)$ is orthogonal, so it *does* leave the singular values of the whole $6\times6$ matrix unchanged
— the printed value is `True`. What changes under a non-orthogonal operation (reference-point shift, Section 3) are the singular values; what never changes under any invertible left-multiplication is the *rank*.
"""),

md(r"""
## 3. Changing the reference point

The linear velocity depends on *which point of the tool* is meant, the angular velocity does not (Notebook 5.1). If point $P'$ is rigidly attached to the tool
at the displacement $\mathbf r=\overrightarrow{PP'}$ from $P$ (components taken in the frame in which $J$ is expressed), then $\mathbf v'=\mathbf v+\boldsymbol\omega\times\mathbf r=\mathbf v-[\mathbf r]_\times\boldsymbol\omega$ and $\boldsymbol\omega'=\boldsymbol\omega$, so

$$
\boxed{J_{P'}=\begin{bmatrix}I&-[\mathbf r]_\times\\0&I\end{bmatrix}J_{P}}.
$$

Two practical uses: (i) **tool offsets** — a screwdriver or a gripper adds a constant displacement and the Jacobian of its tip follows from that of the flange; (ii) the **wrist-centre Jacobian** used in
the singularity analysis of Notebook 5.4. The shift is not an orthogonal transformation, so it changes singular values and manipulability; the angular rows of the matrix are untouched. The next cell
confirms the rule against MuJoCo on the UR5e, with three different "points of the tool":

* the `attachment_site` (flange) via `mj_jacSite`;
* the origin of the `wrist_3_link` body via `mj_jacBody` — this point lies on the axis of joint 6, $0.1$ m before the flange;
* a tool tip $0.15$ m beyond the flange along its $z$ axis, via `mj_jac(point=...)`, and equivalently by adding $\mathrm{Trans}(0,0,0.15)$ to the D-H tool transform.
"""),

code(r'''
from rgkit import robots, mujoco_utils as mju
import mujoco

model, data = robots.load_robot("ur5e", scene="bare")
site = "attachment_site"
q = np.array([0.4, -1.2, 1.5, -0.9, 1.1, 0.5])
data.qpos[:] = q; mujoco.mj_forward(model, data)
p_site = data.site_xpos[model.site(site).id].copy()
R_site = data.site_xmat[model.site(site).id].reshape(3, 3).copy()

J_site = jc.mujoco_jacobian(model, data, site=site)
J_body = jc.mujoco_jacobian(model, data, body="wrist_3_link")
p_body = data.xpos[model.body("wrist_3_link").id]
tip = p_site + R_site @ np.array([0, 0, 0.15])
J_tip = jc.mujoco_jacobian(model, data, body="wrist_3_link", point=tip)

print("shift rule  J(wrist_3 origin) from J(flange):", np.allclose(jc.shift_reference_point(J_site, p_body - p_site), J_body, atol=1e-12))
print("shift rule  J(tool tip)       from J(flange):", np.allclose(jc.shift_reference_point(J_site, tip - p_site), J_tip, atol=1e-12))
J_dh_tip = jc.jacobian_in_frame(jc.jacobian_geometric(jc.UR5E_DH, q, tool=jc.UR5E_TOOL @ transl(0, 0, 0.15)), jc.UR5E_BASE_IN_WORLD[:3, :3])
print("D-H Jacobian with a 0.15 m longer tool = MuJoCo mj_jac at that point:", np.allclose(J_dh_tip, J_tip, atol=1e-12))
print("angular rows are identical for all three points:", np.allclose(J_site[3:], J_body[3:]) and np.allclose(J_site[3:], J_tip[3:]))
print("distance flange -> wrist_3 origin:", np.linalg.norm(p_body - p_site).round(4), "m")
'''),

md(r"""
## 4. The $6\times6$ velocity transformation between frames

Combining Sections 2 and 3 gives Craig's *Cartesian transformation of velocities* (§5.10). Let $\{A\}$ and $\{B\}$ be two frames **fixed to the same rigid body**, with ${}^{A}_{B}T=(R,\ \mathbf p)$,
and let $({}^{B}\mathbf v_B,{}^{B}\boldsymbol\omega)$ be the velocity of the body point at the origin of $\{B\}$ and the angular velocity of the body, both described in $\{B\}$. The velocity of the
body point at the origin of $\{A\}$, described in $\{A\}$, is

$$
\begin{bmatrix}{}^{A}\mathbf v_A\\ {}^{A}\boldsymbol\omega\end{bmatrix}
=\underbrace{\begin{bmatrix}R&[\mathbf p]_\times R\\ 0&R\end{bmatrix}}_{X({}^{A}_{B}T)}
\begin{bmatrix}{}^{B}\mathbf v_B\\ {}^{B}\boldsymbol\omega\end{bmatrix}.
$$

*Derivation.* The angular velocity is simply rotated, ${}^{A}\boldsymbol\omega=R\,{}^{B}\boldsymbol\omega$. The body point at the origin of $\{A\}$ is displaced from the origin of $\{B\}$ by ${}^{B}\mathbf r=-R^{\mathsf T}\mathbf p$, so
${}^{B}\mathbf v_A={}^{B}\mathbf v_B+{}^{B}\boldsymbol\omega\times{}^{B}\mathbf r$; rotating into $\{A\}$ and using $R(\mathbf a\times\mathbf b)=R\mathbf a\times R\mathbf b$ gives
${}^{A}\mathbf v_A=R\,{}^{B}\mathbf v_B-{}^{A}\boldsymbol\omega\times\mathbf p=R\,{}^{B}\mathbf v_B+[\mathbf p]_\times R\,{}^{B}\boldsymbol\omega$. $\square$

With $\mathbf p=\mathbf 0$ the matrix reduces to $\mathrm{diag}(R,R)$ of Section 2; with $R=I$ it reduces to the point shift of Section 3, the displacement being $-\mathbf p$. It is implemented as
`rgkit.jacobian.velocity_transform(T_AB)`, and Notebook 5.5 will show that the matrix for **forces** is its inverse transpose. Typical use: the tool carries a camera whose frame $\{C\}$ is known relative to the tool, ${}^{T}_{C}T$, and we want the camera's own velocity — the 6-vector
$X({}^{C}_{T}T)\,{}^{T}\mathcal V$, or equivalently the Jacobian ${}^{C}J=X({}^{C}_{T}T)\,{}^{T}J$ (a "visual servoing" Jacobian).

The cell below mounts a virtual camera on the UR5e flange and derives its twist two ways: from the $6\times6$ transformation applied to the tool twist, and from the Jacobian of the camera point and
orientation obtained by adding the camera mount to the D-H tool transform.
"""),

code(r'''
T_tool_cam = make_T(rotx(-np.pi / 2) @ rotz(0.2), [0.04, 0.0, 0.08])            # camera frame {C} in the flange/tool frame {T}
qd = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.25])

J_T = jc.jacobian_by_propagation(jc.UR5E_DH, q, jc.UR5E_TOOL)                  # ^T J : in the tool frame
twist_T = J_T @ qd                                                             # (^T v, ^T w) of the tool origin
twist_C = jc.velocity_transform(inv_T(T_tool_cam)) @ twist_T                   # ^C(v, w) of the camera origin
J_C = jc.jacobian_by_propagation(jc.UR5E_DH, q, jc.UR5E_TOOL @ T_tool_cam)     # camera Jacobian straight from the table
print("camera twist from  X(^C_T T) ^T V :", twist_C.round(5))
print("camera twist from  ^C J qdot      :", (J_C @ qd).round(5))
print("agree:", np.allclose(twist_C, J_C @ qd))
print("and ^C J = X ^T J as matrices:", np.allclose(jc.velocity_transform(inv_T(T_tool_cam)) @ J_T, J_C))
'''),

md(r"""
## 5. Geometric and analytical Jacobians

So far the bottom three rows of $J$ gave the **angular velocity** $\boldsymbol\omega$. That is a physically meaningful vector but *it is not the time derivative of any set of three
orientation coordinates*. Suppose, instead, that we describe the orientation of the tool by three angles $\boldsymbol\phi=(\phi_1,\phi_2,\phi_3)$ — roll–pitch–yaw, Euler angles, anything — and
collect the pose coordinates $\mathbf x=(\mathbf p,\boldsymbol\phi)$. The **analytical Jacobian** is the honest derivative $J_a=\partial\mathbf x/\partial\mathbf q$. It is related to the geometric Jacobian through the
**angle-rate matrix** $E(\boldsymbol\phi)$ defined by $\boldsymbol\omega=E(\boldsymbol\phi)\,\dot{\boldsymbol\phi}$:

$$
\begin{bmatrix}\dot{\mathbf p}\\ \dot{\boldsymbol\phi}\end{bmatrix}=\underbrace{\begin{bmatrix}I&0\\0&E(\boldsymbol\phi)^{-1}\end{bmatrix}}_{\text{when }E\text{ is invertible}}\begin{bmatrix}\mathbf v\\ \boldsymbol\omega\end{bmatrix}
\quad\Longrightarrow\quad
J_a=\begin{bmatrix}I&0\\0&E^{-1}\end{bmatrix}J.
$$

**Deriving $E$.** Write the rotation as a product of three elementary rotations; the angular velocity is the sum of three vectors, one for each angle (Notebook 5.1, Section 5): each angle rate multiplies the
axis about which that angle turns, *as it stands after the rotations that precede it*.

* **Craig's X–Y–Z fixed angles** (roll $\gamma$ about $\hat X$, pitch $\beta$ about $\hat Y$, yaw $\alpha$ about $\hat Z$; $R=R_Z(\alpha)R_Y(\beta)R_X(\gamma)$). The yaw axis is the fixed $\hat Z$; the pitch axis is $\hat Y$ after yaw, $R_Z(\alpha)\hat Y$; the roll axis is $\hat X$ after yaw and pitch, $R_Z(\alpha)R_Y(\beta)\hat X$:

$$
\boldsymbol\omega=\dot\gamma\begin{bmatrix}c_\alpha c_\beta\\ s_\alpha c_\beta\\-s_\beta\end{bmatrix}+\dot\beta\begin{bmatrix}-s_\alpha\\c_\alpha\\0\end{bmatrix}+\dot\alpha\begin{bmatrix}0\\0\\1\end{bmatrix},
\qquad E_{xyz}=\begin{bmatrix}c_\alpha c_\beta&-s_\alpha&0\\ s_\alpha c_\beta&c_\alpha&0\\-s_\beta&0&1\end{bmatrix},\ \ \det E_{xyz}=\cos\beta.
$$

* **Z–Y–Z Euler angles** ($R=R_Z(\alpha)R_Y(\beta)R_Z(\gamma)$; each rotation about an axis of the *moving* frame): the three axes are $\hat Z$, $R_Z(\alpha)\hat Y$, $R_Z(\alpha)R_Y(\beta)\hat Z$, so

$$
E_{zyz}=\begin{bmatrix}0&-s_\alpha&c_\alpha s_\beta\\0&c_\alpha&s_\alpha s_\beta\\1&0&c_\beta\end{bmatrix},\qquad \det E_{zyz}=-\sin\beta .
$$

The matrix is described in the base frame; the same construction in the *rotating* frame gives $E_b=R^{\mathsf T}E$. `rgkit.jacobian.euler_rate_matrix(angles, order, frame)` builds $E$ for any of the 24 conventions of `euler_to_rot`.
The code first confirms the two matrices above against numerical differentiation of the rotation matrix and with sympy, then applies $E^{-1}$ to a given angular velocity.
"""),

code(r'''
al, be, ga = sp.symbols("alpha beta gamma", real=True)
Rz_s = lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])
Ry_s = lambda a: sp.Matrix([[sp.cos(a), 0, sp.sin(a)], [0, 1, 0], [-sp.sin(a), 0, sp.cos(a)]])
Rx_s = lambda a: sp.Matrix([[1, 0, 0], [0, sp.cos(a), -sp.sin(a)], [0, sp.sin(a), sp.cos(a)]])
ez, ey, ex = sp.Matrix([0, 0, 1]), sp.Matrix([0, 1, 0]), sp.Matrix([1, 0, 0])
E_xyz = sp.Matrix.hstack(Rz_s(al) * Ry_s(be) * ex, Rz_s(al) * ey, ez)               # columns for (gamma_dot, beta_dot, alpha_dot)
E_zyz = sp.Matrix.hstack(ez, Rz_s(al) * ey, Rz_s(al) * Ry_s(be) * ez)               # columns for (alpha_dot, beta_dot, gamma_dot)
display(Math(r"E_{xyz}=" + sp.latex(E_xyz) + r",\quad \det=" + sp.latex(sp.simplify(E_xyz.det()))))
display(Math(r"E_{zyz}=" + sp.latex(E_zyz) + r",\quad \det=" + sp.latex(sp.simplify(E_zyz.det()))))

ang = np.array([deg(30), deg(60), deg(-45)])
E_num = jc.euler_rate_matrix(ang, "zyz", "current")
print("library E_zyz = sympy E_zyz:", np.allclose(E_num, np.array(E_zyz.subs({al: ang[0], be: ang[1], ga: ang[2]}), dtype=float)))
E_num2 = jc.euler_rate_matrix([deg(20), deg(35), deg(50)], "xyz", "fixed")           # (gamma, beta, alpha)
print("library E_xyz = sympy E_xyz:", np.allclose(E_num2, np.array(E_xyz.subs({ga: deg(20), be: deg(35), al: deg(50)}), dtype=float)))

omega = np.array([0.2, -0.1, 0.3])
rates = np.linalg.solve(E_num, omega)
h = 1e-6
w_check = unskew(((euler_to_rot(ang + h * rates, "zyz") - euler_to_rot(ang - h * rates, "zyz")) / (2 * h)) @ euler_to_rot(ang, "zyz").T)
print("angular velocity", omega, " ->  Z-Y-Z angle rates", rates.round(4), " -> back to omega:", w_check.round(5))
'''),

md(r"""
**Which Jacobian when?** The geometric Jacobian is the natural one for *velocity and force* relations — it is what MuJoCo, KDL, Pinocchio and most controllers use, and the one that
$\tau=J^{\mathsf T}F$ needs. The analytical Jacobian is the natural one when the task is *specified* by the angles themselves (for example, "hold roll at $0^\circ$ and let pitch go from $10^\circ$ to $50^\circ$ at a given rate") or when one
differentiates a cost that is written in angles. Orientation *errors* in control are best expressed as a rotation vector ($\theta\hat K$ of the relative rotation), whose rate is $\omega$, so that no $E$ is
needed; Notebook 5.6 uses this.
"""),

md(r"""
## 6. Representation singularities

$E$ is invertible only when $\det E\neq0$: for fixed XYZ angles when $\cos\beta\neq0$, and for ZYZ Euler angles when $\sin\beta\neq0$. At $\beta=\pm90^\circ$ (XYZ) the roll and yaw axes become parallel: two of the three
columns of $E$ coincide, the matrix has rank 2, and **no angular velocity along the lost direction can be written as angle rates**. This is the gimbal lock of Chapter 2 seen at velocity level. Two features distinguish it from a
physical singularity:

1. **It belongs to the coordinates, not the robot.** A rigid body can rotate perfectly well about any axis at pitch $90^\circ$; only the *parametrisation* fails. A different angle set (ZYZ instead of XYZ) is regular at the same orientation.
2. **It makes the analytical Jacobian blow up while the geometric Jacobian stays finite.** For a modest angular velocity near $\beta=90^\circ$ the angle rates become arbitrarily large.

The experiment rotates a body about the *fixed* $x$ axis at a constant $1$ rad/s, starting from XYZ angles $(\gamma,\beta,\alpha)=(20^\circ,\beta_0,30^\circ)$, so that the true angular velocity has constant magnitude, and computes
$\dot{\boldsymbol\phi}=E^{-1}\boldsymbol\omega$ along the way. Try the slider: for starting pitch near $90^\circ$ the rates shoot up although nothing dramatic happens to the body.
"""),

code(r'''
def rpy_rates_demo(beta0_deg=60.0):
    w = np.array([1.0, 0.0, 0.0])                                        # constant angular velocity about the fixed x axis
    R0 = rpy_to_rot(deg(20), deg(beta0_deg), deg(30))
    t = np.linspace(0, 1.2, 241)
    ang, rates, conds = [], [], []
    for ti in t:
        R = rot_axis_angle_x(ti) @ R0
        g, b, a = rot_to_rpy(R)
        E = jc.euler_rate_matrix([g, b, a], "xyz", "fixed")
        rates.append(np.linalg.lstsq(E, w, rcond=1e-12)[0]); ang.append([g, b, a]); conds.append(np.linalg.cond(E))
    ang, rates = np.degrees(ang), np.array(rates)
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5))
    axs[0].plot(t, ang[:, 1], color="#1f77b4"); axs[0].axhline(90, color="k", ls=":"); axs[0].set_ylim(-95, 95)
    axs[0].set_title(r"pitch $\beta$ (deg)"); axs[0].set_xlabel("t (s)")
    for k, (lab, col) in enumerate(zip([r"$\dot\gamma$ (roll)", r"$\dot\beta$ (pitch)", r"$\dot\alpha$ (yaw)"], ["#d62728", "#2ca02c", "#9467bd"])):
        axs[1].plot(t, rates[:, k], color=col, label=lab)
    axs[1].set_ylim(-12, 12); axs[1].legend(fontsize=8); axs[1].set_title(r"angle rates for $|\omega|=1$ rad/s"); axs[1].set_xlabel("t (s)")
    axs[2].semilogy(t, conds, color="#ff7f0e"); axs[2].set_title(r"condition number of $E$"); axs[2].set_xlabel("t (s)")
    plt.tight_layout(); plt.show()

def rot_axis_angle_x(a):
    return rotx(a)

interact(rpy_rates_demo, beta0_deg=widgets.FloatSlider(value=60.0, min=-85, max=85, step=1, description="β0 (deg)"))
'''),

md(r"""
In the default setting the body passes close to $\beta=90^\circ$ and the roll and yaw rates oscillate with spikes while $\lVert\boldsymbol\omega\rVert$ stays at $1$ rad/s: the *same physical motion* needs unbounded angle rates in the XYZ description.
A real manipulator suffers from this whenever a task is specified in Euler angles and the commanded orientation passes through a gimbal-lock pose — one reason why robot controllers
store orientations as quaternions or rotation matrices and command angular velocity.

**Terminology.** A configuration where $J_g$ loses rank is a **kinematic** (or *manipulator*) singularity — Notebook 5.4. A configuration where only $E$ loses rank is a **representation** (or *algorithmic*) singularity. The analytical Jacobian
inherits both; the geometric Jacobian only the first.
"""),

md(r"""
## 7. A real robot: frames and angle rates on the UR5e

Two checks with the UR5e of the earlier notebooks:

1. **Frames.** MuJoCo's Jacobian is described in the world frame. Rotating it by the transpose of the flange orientation gives the tool-frame Jacobian, which must equal the one obtained from our D-H table by propagation.
2. **Angle rates.** From MuJoCo's world Jacobian and the measured orientation we form $J_a=\mathrm{diag}(I,E^{-1})J$ for roll–pitch–yaw of the flange and compare it with *finite differences* of the quantities $(\mathbf p,\gamma,\beta,\alpha)$ read from the simulator. The agreement shows that
the analytical Jacobian really is the derivative of the angle coordinates. Then we search for a configuration where the flange's $x$ axis is vertical, i.e. the pitch is $\pm90^\circ$, and compare the size of $J_g$ with $J_a$ there.
"""),

code(r'''
def pose_mj(qq):
    data.qpos[:] = qq; mujoco.mj_kinematics(model, data)
    sid = model.site(site).id
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()

# ---- (1) tool-frame Jacobian
q = np.array([0.4, -1.2, 1.5, -0.9, 1.1, 0.5])
p, R = pose_mj(q); mujoco.mj_forward(model, data)
J_world = jc.mujoco_jacobian(model, data, site=site)
J_tool_mj = jc.jacobian_in_frame(J_world, R.T)
J_tool_dh = jc.jacobian_by_propagation(jc.UR5E_DH, q, jc.UR5E_TOOL)
print("tool-frame Jacobian: MuJoCo (rotated) vs D-H propagation, max difference:", np.abs(J_tool_mj - J_tool_dh).max().round(14))

# ---- (2) analytical Jacobian for roll-pitch-yaw = X-Y-Z fixed angles
def pose_params(qq):
    p, R = pose_mj(qq); g, b, a = rot_to_rpy(R)
    return np.r_[p, g, b, a]
Ja_fd = jc.jacobian_position_numeric(pose_params, q, h=1e-6)
g, b, a = rot_to_rpy(R)
Ja = jc.analytical_jacobian(J_world, [g, b, a], "xyz", "fixed")
print("analytical Jacobian from  diag(I, E^-1) J  vs finite differences of (p, roll, pitch, yaw):", np.abs(Ja - Ja_fd).max().round(8))
print("roll, pitch, yaw (deg) at q:", np.degrees([g, b, a]).round(2))
'''),

code(r'''
# ---- search for a configuration whose flange pitch is close to 90 degrees
rng = np.random.default_rng(2)
best, best_val = None, 0
for _ in range(4000):
    qq = rng.uniform(-np.pi, np.pi, 6)
    Rq = (jc.UR5E_BASE_IN_WORLD @ fk_tool(jc.UR5E_DH, qq, tool=jc.UR5E_TOOL))[:3, :3]
    if abs(Rq[2, 0]) > best_val:
        best, best_val = qq, abs(Rq[2, 0])
res_pitch = np.degrees(np.arcsin(best_val))
print(f"best of 4000 random configurations: |sin(pitch)| = {best_val:.5f} -> pitch = {res_pitch:.2f} deg")

# refine with a few Newton-like steps on the pitch so that it is close to (but not exactly at) 90 degrees
from scipy.optimize import minimize
fun = lambda qq: 1 - abs((jc.UR5E_BASE_IN_WORLD @ fk_tool(jc.UR5E_DH, qq, tool=jc.UR5E_TOOL))[2, 0]) + 1e-4 * np.sum((qq - best) ** 2)
qn = minimize(fun, best, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=4000)).x
p, R = pose_mj(qn); mujoco.mj_forward(model, data)
Jg = jc.mujoco_jacobian(model, data, site=site)
g, b, a = rot_to_rpy(R)
E = jc.euler_rate_matrix([g, b, a], "xyz", "fixed")
Ja = jc.analytical_jacobian(Jg, [g, b, a], "xyz", "fixed")
print(f"pitch = {np.degrees(b):.3f} deg, det E = cos(pitch) = {np.linalg.det(E):.2e}, cond(E) = {np.linalg.cond(E):.1e}")
print(f"largest entry of the geometric Jacobian J_g : {np.abs(Jg).max():.3f}")
print(f"largest entry of the analytical Jacobian J_a: {np.abs(Ja).max():.1f}")
print("J_g is still well conditioned there:", f"cond(J_g) = {np.linalg.cond(Jg):.2f}")
'''),

md(r"""
The robot is nowhere near a *kinematic* singularity — the geometric Jacobian is perfectly conditioned — yet the analytical Jacobian, which maps joint rates to the rates of the roll and yaw angles, has entries of order $10^{3}$ or more. A controller that commanded Euler-angle rates would
ask for enormous joint speeds here, for no physical reason.
"""),

md(r"""
## 8. Worked examples

**Example 5.3.1 — tool-frame Jacobian of the two-link arm.** For $L_1=1.0$, $L_2=0.8$, $\theta=(30^\circ,60^\circ)$ find ${}^{0}J$ and ${}^{3}J$ and the tip velocity in each frame for $\dot{\boldsymbol\theta}=(1.0,0.5)$ rad/s.

*Solution.* ${}^{0}J=\begin{bmatrix}-1.3&-0.8\\0.8660&0\end{bmatrix}$ ($s_1=0.5$, $s_{12}=1$, $c_1=0.8660$, $c_{12}=0$), so ${}^{0}\mathbf v=(-1.7,\ 0.8660)$. The tool frame is turned $90^\circ$; from the formula of Section 2,
${}^{3}J=\begin{bmatrix}L_1s_2&0\\L_1c_2+L_2&L_2\end{bmatrix}=\begin{bmatrix}0.8660&0\\1.3&0.8\end{bmatrix}$ and ${}^{3}\mathbf v=(0.8660,\ 1.7)$. The speed $1.908$ m/s is the same in both frames, and both determinants equal $L_1L_2s_2=0.6928$.

**Example 5.3.2 — a longer tool.** A rod of $0.3$ m is clamped along the second link so that the tool point moves $0.3$ m beyond the tip. Find the new Jacobian from the old one.

*Solution.* In the base frame the displacement is $\mathbf r=0.3(c_{12},s_{12},0)=(0,0.3,0)$ at $\theta_{12}=90^\circ$. Then $J'=\begin{bmatrix}I&-[\mathbf r]_\times\\0&I\end{bmatrix}J$ and, restricted to the planar rows,
$\mathbf v'=\mathbf v+\omega\hat z\times\mathbf r=\mathbf v+\omega(-0.3,0)$, i.e. $J'_{1,:}=J_{1,:}-0.3\,(1,1)$. The result equals the Jacobian of an arm with $L_2'=1.1$ m: $\begin{bmatrix}-1.6&-1.1\\0.8660&0\end{bmatrix}$.

**Example 5.3.3 — a camera on the tool.** The tool (UR5e flange) moves with the twist $({}^{T}\mathbf v,{}^{T}\boldsymbol\omega)=((0.1,0,0.2),(0,0.5,0))$ in its own frame. A camera sits at the point $(0.05,0,0.1)$ m of the tool frame with the axes parallel to the tool's axes. Find the camera origin's velocity.

*Solution.* With $R=I$ and ${}^{T}_{C}\mathbf p=(0.05,0,0.1)$ the camera velocity in $\{T\}$ axes is ${}^{T}\mathbf v+{}^{T}\boldsymbol\omega\times{}^{T}\mathbf p=(0.1,0,0.2)+(0,0.5,0)\times(0.05,0,0.1)=(0.1+0.05,\ 0,\ 0.2-0.025)=(0.15,\ 0,\ 0.175)$ m/s, using $(0,0.5,0)\times(0.05,0,0.1)=(0.05,0,-0.025)$. The angular velocity is unchanged.

**Example 5.3.4 — angular velocity to angle rates.** The tool orientation is described by Z–Y–Z Euler angles $(\alpha,\beta,\gamma)=(30^\circ,60^\circ,-45^\circ)$ and has the base-frame angular velocity $\boldsymbol\omega=(0.2,-0.1,0.3)$ rad/s. Find $(\dot\alpha,\dot\beta,\dot\gamma)$.

*Solution.* From $E_{zyz}$ with $c_\alpha=0.8660$, $s_\alpha=0.5$, $c_\beta=0.5$, $s_\beta=0.8660$: solving $E\dot{\boldsymbol\phi}=\boldsymbol\omega$ gives $\dot{\boldsymbol\phi}=(0.2289,\ -0.1866,\ 0.1423)$ rad/s. (The determinant is $-\sin\beta=-0.8660$, so the problem is well posed.)

**Example 5.3.5 — gimbal lock.** For XYZ fixed angles at pitch $\beta=90^\circ$, roll $\gamma=0.3$ rad, yaw $\alpha=0.5$ rad, which angular velocities can be produced by angle rates?

*Solution.* $E_{xyz}$ has columns $(0,0,-1)$, $(-s_\alpha,c_\alpha,0)$ and $(0,0,1)$: rank 2. The reachable angular velocities form the plane spanned by $\hat z$ and $(-s_\alpha,c_\alpha,0)=(-0.4794,\ 0.8776,\ 0)$; any $\boldsymbol\omega$ with a component along $(c_\alpha,s_\alpha,0)=(0.8776,\ 0.4794,\ 0)$ cannot be represented, although the body can of course rotate about that axis.
"""),

code(r'''
# Example 5.3.1
J0 = jc.planar_jacobian([1.0, 0.8], [deg(30), deg(60)]); qd = np.array([1.0, 0.5])
R03 = rotz(deg(90))[:2, :2]
J3 = R03.T @ J0
print("5.3.1  ^0J =\n", J0.round(4), "\n       ^3J =\n", J3.round(4))
print("       ^0v =", (J0 @ qd).round(4), " ^3v =", (J3 @ qd).round(4), " speeds:", np.linalg.norm(J0 @ qd).round(4), np.linalg.norm(J3 @ qd).round(4))
assert np.allclose(J3, [[0.8660, 0], [1.3, 0.8]], atol=1e-4) and np.allclose(J0 @ qd, [-1.7, 0.8660], atol=1e-4)
assert np.isclose(np.linalg.det(J0), 0.6928, atol=1e-4) and np.isclose(np.linalg.det(J3), np.linalg.det(J0))

# Example 5.3.2
r = 0.3 * np.array([np.cos(deg(90)), np.sin(deg(90))])
J6 = np.vstack([J0, [1, 1]])                                         # add the angular row (omega_z)
# planar shift: v' = v + omega x r  with omega = omega_z z-hat:  vx' = vx - omega r_y,  vy' = vy + omega r_x
J_shifted = np.vstack([J6[0] - r[1] * J6[2], J6[1] + r[0] * J6[2]])
J_L11 = jc.planar_jacobian([1.0, 1.1], [deg(30), deg(60)])
print("5.3.2  J' =\n", J_shifted.round(4), "\n       arm with L2 = 1.1:\n", J_L11.round(4))
assert np.allclose(J_shifted, J_L11) and np.allclose(J_L11, [[-1.6, -1.1], [0.8660, 0]], atol=1e-4)

# Example 5.3.3
v_T, w_T, p_C = np.array([0.1, 0, 0.2]), np.array([0, 0.5, 0]), np.array([0.05, 0, 0.1])
v_cam = v_T + np.cross(w_T, p_C)
X = jc.velocity_transform(inv_T(make_T(np.eye(3), p_C)))             # ^C_T T = inverse of ^T_C T
print("5.3.3  camera velocity:", v_cam.round(4), "  via X:", (X @ np.r_[v_T, w_T])[:3].round(4))
assert np.allclose(v_cam, [0.15, 0, 0.175]) and np.allclose((X @ np.r_[v_T, w_T])[:3], v_cam)

# Example 5.3.4
ang = np.array([deg(30), deg(60), deg(-45)]); E = jc.euler_rate_matrix(ang, "zyz", "current")
rates = np.linalg.solve(E, [0.2, -0.1, 0.3])
print("5.3.4  rates =", rates.round(4), " det E =", np.linalg.det(E).round(4))
assert np.allclose(rates, [0.2289, -0.1866, 0.1423], atol=1e-4) and np.isclose(np.linalg.det(E), -np.sin(deg(60)))

# Example 5.3.5
E = jc.euler_rate_matrix([0.3, np.pi / 2, 0.5], "xyz", "fixed")
print("5.3.5  rank E =", np.linalg.matrix_rank(E), "  columns:\n", E.round(4))
assert np.linalg.matrix_rank(E) == 2 and np.allclose(E[:, 1], [-np.sin(0.5), np.cos(0.5), 0])
print("all worked examples verified")
'''),

md(r"""
## 9. Summary

| Operation | Rule |
|:--|:--|
| Change of frame | ${}^{B}J=\mathrm{diag}({}^{B}_{A}R,{}^{B}_{A}R)\,{}^{A}J$; rank, determinant and singular configurations unchanged |
| Tool-frame Jacobian | ${}^{T}J=\mathrm{diag}({}^{0}_{T}R^{\mathsf T},{}^{0}_{T}R^{\mathsf T})\,{}^{0}J$; obtained directly by propagation |
| Change of reference point | $J_{P'}=\begin{bmatrix}I&-[\mathbf r]_\times\\0&I\end{bmatrix}J_P$ (angular rows unchanged) |
| Velocity between frames on one body | $X({}^{A}_{B}T)=\begin{bmatrix}R&[\mathbf p]_\times R\\0&R\end{bmatrix}$ |
| Angle-rate matrix | $\boldsymbol\omega=E(\boldsymbol\phi)\dot{\boldsymbol\phi}$; $\det E_{xyz}=\cos\beta$, $\det E_{zyz}=-\sin\beta$ |
| Analytical Jacobian | $J_a=\mathrm{diag}(I,E^{-1})J$; extra singularities where $\det E=0$ |

### Common mistakes

* Rotating only the linear rows (or only the angular rows) when changing the frame.
* Shifting the reference point with $\mathbf r$ written in the wrong frame: $\mathbf r$ must be in the frame of the Jacobian.
* Mixing the sign of the shift: $\mathbf v'=\mathbf v+\boldsymbol\omega\times\mathbf r=\mathbf v-[\mathbf r]_\times\boldsymbol\omega$ for $P'=P+\mathbf r$.
* Treating $(\dot\gamma,\dot\beta,\dot\alpha)$ as angular velocity components, or using a $J_a$ near gimbal lock without noticing the blow-up.
* Believing that a singularity of $J_a$ implies a singularity of the robot.

## Exercises

**5.3.1** Show that $\det{}^{B}J=\det{}^{A}J$ for a square Jacobian and any two frames $\{A\},\{B\}$, and give the corresponding statement for the singular values.

<details><summary>Answer</summary>

${}^{B}J=\mathrm{diag}(R,R)\,{}^{A}J$ and $\det\mathrm{diag}(R,R)=(\det R)^2=1$, hence the determinants are equal. Since $\mathrm{diag}(R,R)$ is orthogonal, $({}^{B}J)^{\mathsf T}\,{}^{B}J=({}^{A}J)^{\mathsf T}\,{}^{A}J$, so
the singular values are equal too (only the left singular vectors rotate).
</details>

**5.3.2** For the three-link planar arm $L=(1,0.8,0.5)$ at $\mathbf q=(30^\circ,-20^\circ,50^\circ)$ find the Jacobian of the tip expressed in the tip frame and verify that its determinant equals that of the base-frame Jacobian.

<details><summary>Answer</summary>

Rotate the planar rows by $R_Z(\phi)^{\mathsf T}$ with $\phi=60^\circ$ and append the unchanged row $(1,1,1)$ for $\omega_z$. The determinant is $L_1L_2\sin q_2=-0.2736$ in both frames; the matrix is printed in the cell below.
</details>

**5.3.3** A UR5e carries a probe $0.20$ m long along the tool $z$ axis. By how much do the linear rows of the Jacobian change relative to the flange Jacobian, and which rows are unchanged? State the matrix that performs the change.

<details><summary>Answer</summary>

With $\mathbf r={}^{0}_{T}R\,(0,0,0.2)^{\mathsf T}$ (expressed in the base frame) the linear rows change by $-[\mathbf r]_\times J_\omega$, i.e. by the cross product of $\mathbf r$ with the angular rows; the angular rows are unchanged. The change in the tool frame is $-[(0,0,0.2)]_\times{}^{T}J_\omega$.
The cell below checks this against the D-H model.
</details>

**5.3.4** Derive $E$ for the roll–pitch–yaw angles defined by $R=R_Z(\alpha)R_Y(\beta)R_X(\gamma)$ *in the rotating frame* (i.e. $\boldsymbol\omega_b=R^{\mathsf T}\boldsymbol\omega$) and show that it is also singular at $\beta=\pm90^\circ$.

<details><summary>Answer</summary>

$E_b=R^{\mathsf T}E$ with $R^{\mathsf T}$ orthogonal, so $\det E_b=\det E=\cos\beta$ and the singularities coincide. Explicitly $E_b=\begin{bmatrix}1&0&-s_\beta\\0&c_\gamma&s_\gamma c_\beta\\0&-s_\gamma&c_\gamma c_\beta\end{bmatrix}$ (columns for $\dot\gamma,\dot\beta,\dot\alpha$); the cell below verifies it numerically.
</details>

**5.3.5** The tool of a robot is at ZYZ angles $(\alpha,\beta,\gamma)=(10^\circ,0^\circ,25^\circ)$. Explain what goes wrong with $E_{zyz}$ and which angular velocities cannot be produced by angle rates.

<details><summary>Answer</summary>

At $\beta=0$ the first and third axes coincide (both $\hat Z$), so columns 1 and 3 of $E_{zyz}$ are equal: $E=\begin{bmatrix}0&-s_\alpha&0\\0&c_\alpha&0\\1&0&1\end{bmatrix}$, rank 2. Only $\boldsymbol\omega$ in the span of $\hat z$ and $(-s_\alpha,c_\alpha,0)$ is attainable; only $\dot\alpha+\dot\gamma$ is determined.
</details>

**5.3.6** (*Programming*) Compute, for a UR5e configuration of your choice, the analytical Jacobian for ZYZ Euler angles of the flange and verify it by finite differences, as in Section 7.

<details><summary>Answer</summary>

Use `rot_to_euler_zyz` for the angles and `jc.analytical_jacobian(J_world, angles, "zyz", "current")`; compare with `jc.jacobian_position_numeric` applied to `[p, *rot_to_euler_zyz(R)]`. The check cell prints a difference below $10^{-6}$ (choose a configuration with $\beta$ away from $0$ and $\pi$).
</details>

**5.3.7** (*MuJoCo*) Using the cell of Section 3, find the Jacobian of a point $0.3$ m *behind* the flange (negative tool $z$) and state what happens to its manipulability compared with the flange.

<details><summary>Answer</summary>

Use `shift_reference_point(J_site, R_site @ [0, 0, -0.3])` and compare `manipulability` of the $6\times6$ matrices (in the cell below). The determinant of the shift matrix is $1$, so $\det J$, and hence Yoshikawa's measure $\sqrt{\det JJ^{\mathsf T}}$ of a square $J$, is **unchanged** by the shift of reference point even though the individual singular values change.
</details>
"""),

code(r'''
# --- verification of exercise answers ---------------------------------------------------------------
# 5.3.2
Lr = [1.0, 0.8, 0.5]; q3 = np.array([deg(30), deg(-20), deg(50)])
J0 = jc.planar_jacobian(Lr, q3, orientation=True)
Rf = rotz(sum(q3))[:2, :2]
Jtip = np.vstack([Rf.T @ J0[:2], J0[2]])
print("5.3.2  tip-frame Jacobian:\n", Jtip.round(4), "\n       det:", np.linalg.det(Jtip).round(4), np.linalg.det(J0).round(4))
assert np.isclose(np.linalg.det(Jtip), np.linalg.det(J0)) and np.isclose(np.linalg.det(J0), -0.2736, atol=1e-4)

# 5.3.3  0.20 m probe along the tool z axis, UR5e
q = np.array([0.4, -1.2, 1.5, -0.9, 1.1, 0.5])
data.qpos[:] = q; mujoco.mj_forward(model, data)
Jw = jc.mujoco_jacobian(model, data, site=site); Rs = data.site_xmat[model.site(site).id].reshape(3, 3)
r_b = Rs @ np.array([0, 0, 0.2])
J_probe = jc.jacobian_in_frame(jc.jacobian_geometric(jc.UR5E_DH, q, tool=jc.UR5E_TOOL @ transl(0, 0, 0.2)), jc.UR5E_BASE_IN_WORLD[:3, :3])
change = J_probe[:3] - Jw[:3]
print("5.3.3  linear rows change by  -[r]x J_omega:", np.allclose(change, -skew(r_b) @ Jw[3:]), "  angular rows unchanged:", np.allclose(J_probe[3:], Jw[3:]))
assert np.allclose(change, -skew(r_b) @ Jw[3:]) and np.allclose(J_probe[3:], Jw[3:])

# 5.3.4  body-frame RPY rate matrix
g_, b_, a_ = 0.4, 0.7, -0.3
Eb = jc.euler_rate_matrix_body([g_, b_, a_], "xyz", "fixed")
Eb_formula = np.array([[1, 0, -np.sin(b_)], [0, np.cos(g_), np.sin(g_) * np.cos(b_)], [0, -np.sin(g_), np.cos(g_) * np.cos(b_)]])
print("5.3.4  E_b matches the closed form:", np.allclose(Eb, Eb_formula), " det E_b = cos(beta):", np.isclose(np.linalg.det(Eb), np.cos(b_)))
assert np.allclose(Eb, Eb_formula) and np.isclose(np.linalg.det(Eb), np.cos(b_))

# 5.3.5
E0 = jc.euler_rate_matrix([deg(10), 0.0, deg(25)], "zyz", "current")
print("5.3.5  rank =", np.linalg.matrix_rank(E0)); assert np.linalg.matrix_rank(E0) == 2 and np.allclose(E0[:, 0], E0[:, 2])

# 5.3.6  ZYZ analytical Jacobian of the UR5e flange
def pose_params_zyz(qq):
    p, R = pose_mj(qq); return np.r_[p, rot_to_euler_zyz(R)]
Jfd = jc.jacobian_position_numeric(pose_params_zyz, q)
data.qpos[:] = q; mujoco.mj_forward(model, data)
Jw = jc.mujoco_jacobian(model, data, site=site); p, R = pose_mj(q)
Jza = jc.analytical_jacobian(Jw, rot_to_euler_zyz(R), "zyz", "current")
print("5.3.6  max |J_a - finite differences| =", np.abs(Jza - Jfd).max()); assert np.abs(Jza - Jfd).max() < 1e-6

# 5.3.7
Jback = jc.shift_reference_point(Jw, Rs @ np.array([0, 0, -0.3]))
print("5.3.7  manipulability flange / shifted:", jc.manipulability(Jw).round(6), jc.manipulability(Jback).round(6),
      " singular values change:", not np.allclose(jc.singular_values(Jw), jc.singular_values(Jback)))
assert np.isclose(jc.manipulability(Jw), jc.manipulability(Jback))
'''),

md(r"""
## Further reading and lectures

* Craig, Ch. 5, §5.6 and §5.10; Ch. 2 §2.8 and Appendix B for the angle sets; Lynch & Park §5.1; Siciliano et al. Ch. 3 (analytical Jacobian); Corke *RVC* Ch. 8.
* Video: """ + f"{V('5.1.1 Space Jacobian')}; {V('5.1.2 Body Jacobian')}; {V('Fixed and Euler angle representation')}; {V('Euler angles — worked example')}; {V('Lecture 11 — Gimbal lock', 'Nandi')}." + r"""
"""),

nav_for(SEC),
]

if __name__ == "__main__":
    build(cells, f"{CH_DIR}/{FN}")
