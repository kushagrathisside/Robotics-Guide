"""Denavit-Hartenberg tools: frame assignment, parameter extraction, symbolic forward kinematics.

Chapter 3 of the MFoR notebooks (forward kinematics) builds on this module. It extends
:mod:`rgkit.kinematics`, which holds the two link transforms and the numerical chain product, with

* **recovering D-H parameters** from a link transform and checking the two D-H geometric rules
  (:func:`dh_params_modified`, :func:`dh_params_standard`, :func:`dh_violations`);
* **frame assignment**: Craig's recipe applied to joint axes given as lines in space
  (:func:`dh_from_axes`), which turns the geometry of a real robot (read, for example, from a
  MuJoCo model with :func:`joint_axes_from_mujoco`) into a D-H table;
* the exact relation between the **modified** and **standard** conventions
  (:func:`modified_to_standard`, :func:`standard_to_modified`);
* **symbolic forward kinematics** with SymPy (:func:`sym_dh_modified`, :func:`sym_fk`) and printing
  in Craig's ``c12, s23`` notation (:func:`latex_cs`);
* a **MuJoCo model generator** that turns a D-H table into a stick-figure robot (:func:`mjcf_from_dh`)
  so that a table can be inspected visually, and helpers to compare chains with MuJoCo
  (:func:`mujoco_poses`, :func:`pose_error`).

Conventions
-----------
A D-H table is a list of rows ``(alpha, a, d, theta, kind)`` exactly as in :mod:`rgkit.kinematics`; for
the modified (Craig) convention row ``i`` holds ``(alpha_{i-1}, a_{i-1}, d_i, theta_i)``, for the
standard (Spong) convention ``(alpha_i, a_i, d_i, theta_i)``. Angles are in radians.
Joint axes are lines ``(point, direction)`` expressed in a common ("world") frame, listed from the
base outwards, at one fixed *reference configuration* (normally all joint variables zero).
"""

from __future__ import annotations

import numpy as np

from .transforms import rotx, rotz, make_T, transl, inv_T, rot_to_quat
from .kinematics import dh_modified, dh_standard

__all__ = [
    "dh_params_modified", "dh_params_standard", "dh_violations", "table_from_frames",
    "dh_from_axes", "common_normal", "modified_to_standard", "standard_to_modified",
    "joint_axes_from_mujoco", "mujoco_poses", "pose_error", "mjcf_from_dh",
    "sym_dh_modified", "sym_dh_standard", "sym_fk", "sym_link_transforms", "cs_substitute",
    "latex_cs", "lambdify_fk", "wrap_angle",
]


def wrap_angle(x):
    """Wrap an angle (or array of angles) to ``(-pi, pi]``."""
    return np.pi - np.mod(np.pi - np.asarray(x, dtype=float), 2 * np.pi)


# ---------------------------------------------------------------------------
# Recovering D-H parameters from a link transform
# ---------------------------------------------------------------------------
def dh_params_modified(T, check: bool = True, tol: float = 1e-7):
    """Return ``(alpha, a, d, theta)`` of a *modified* link transform ``Rx(alpha) Dx(a) Rz(theta) Dz(d)``.

    The inverse of :func:`rgkit.kinematics.dh_modified`. ``ValueError`` is raised when ``T`` is not of
    that form (``check=True``), which means that the two frames do not obey the D-H rules.
    """
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    alpha = float(np.arctan2(-R[1, 2], R[2, 2]))
    theta = float(np.arctan2(-R[0, 1], R[0, 0]))
    a = float(p[0])
    d = float(-np.sin(alpha) * p[1] + np.cos(alpha) * p[2])
    if check and not np.allclose(dh_modified(alpha, a, d, theta), T, atol=tol):
        raise ValueError("transform is not of modified D-H form: the frames violate the D-H rules")
    return alpha, a, d, theta


def dh_params_standard(T, check: bool = True, tol: float = 1e-7):
    """Return ``(alpha, a, d, theta)`` of a *standard* link transform ``Rz(theta) Dz(d) Dx(a) Rx(alpha)``."""
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    alpha = float(np.arctan2(R[2, 1], R[2, 2]))
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    a = float(p[0] * np.cos(theta) + p[1] * np.sin(theta))
    d = float(p[2])
    if check and not np.allclose(dh_standard(alpha, a, d, theta), T, atol=tol):
        raise ValueError("transform is not of standard D-H form: the frames violate the D-H rules")
    return alpha, a, d, theta


def dh_violations(T_rel, convention: str = "modified"):
    """Measure how badly a relative transform ``{i-1} -> {i}`` breaks the two D-H rules.

    Returns ``(perpendicularity, intersection)``; both are zero for a valid D-H pair.

    * modified: ``X_{i-1}`` must be perpendicular to ``Z_i`` and must intersect it;
    * standard: ``X_i`` must be perpendicular to ``Z_{i-1}`` and must intersect it.

    (The second number is the distance between the two lines, measured in metres.)
    """
    T_rel = np.asarray(T_rel, dtype=float)
    R, p = T_rel[:3, :3], T_rel[:3, 3]
    if convention == "modified":
        x_dir, z_dir = np.array([1.0, 0, 0]), R[:, 2]          # X_{i-1} and Z_i in frame {i-1}
        pt_x, pt_z = np.zeros(3), p                             # a point on each line
    else:
        x_dir, z_dir = R[:, 0], np.array([0, 0, 1.0])          # X_i and Z_{i-1} in frame {i-1}
        pt_x, pt_z = p, np.zeros(3)
    perp = abs(float(np.dot(x_dir, z_dir)))
    n = np.cross(x_dir, z_dir)
    nn = np.linalg.norm(n)
    if nn < 1e-12:                                              # parallel lines: not allowed by the rules
        return perp, float(np.linalg.norm(np.cross(pt_z - pt_x, x_dir)))
    return perp, abs(float(np.dot(pt_z - pt_x, n))) / nn


def table_from_frames(frames, convention: str = "modified", kinds=None):
    """D-H table from a list of frames ``{0}, {1}, ..., {N}`` (4x4 poses in one common frame).

    Each consecutive pair must satisfy the D-H rules (see :func:`dh_violations`), otherwise ``ValueError``.
    For the modified convention row ``i`` is ``(alpha_{i-1}, a_{i-1}, d_i, theta_i)``.
    """
    n = len(frames) - 1
    kinds = ["R"] * n if kinds is None else list(kinds)
    extract = dh_params_modified if convention == "modified" else dh_params_standard
    rows = []
    for i in range(1, n + 1):
        rel = inv_T(frames[i - 1]) @ frames[i]
        alpha, a, d, theta = extract(rel)
        rows.append((alpha, a, d, theta, kinds[i - 1]))
    return rows


# ---------------------------------------------------------------------------
# Frame assignment from joint axes (Craig's recipe, automated)
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=float).ravel()
    return v / np.linalg.norm(v)


def _perp_to(z, prefer=None):
    """A unit vector perpendicular to ``z``, as close to ``prefer`` as possible."""
    z = _unit(z)
    cands = [] if prefer is None else [np.asarray(prefer, dtype=float)]
    cands += [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    for c in cands:
        v = c - np.dot(c, z) * z
        if np.linalg.norm(v) > 1e-6:
            return v / np.linalg.norm(v)
    raise RuntimeError("no perpendicular found")  # pragma: no cover


def common_normal(p1, z1, p2, z2, tol: float = 1e-9):
    """Common normal of the lines ``p1 + s z1`` and ``p2 + t z2``.

    Returns ``(foot1, foot2, kind)`` where ``foot1``/``foot2`` are the feet of the common normal on the first and the
    second line and ``kind`` is ``"skew"``, ``"intersecting"`` or ``"parallel"``. For parallel lines the common
    normal is not unique; the one through ``p1`` is returned (``foot1 = p1``).
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    z1, z2 = _unit(z1), _unit(z2)
    n = np.cross(z1, z2)
    nn = float(np.dot(n, n))
    if nn < tol:
        return p1, p2 + np.dot(p1 - p2, z2) * z2, "parallel"
    r = p2 - p1
    s = float(np.dot(np.cross(r, z2), n) / nn)
    t = float(np.dot(np.cross(r, z1), n) / nn)
    f1, f2 = p1 + s * z1, p2 + t * z2
    kind = "intersecting" if np.linalg.norm(f1 - f2) < 1e-7 else "skew"
    return f1, f2, kind


def dh_from_axes(points, directions, convention: str = "modified", kinds=None, tool_point=None,
                 tool_axis=None, x_sign: str = "craig", x_hint=(1.0, 0.0, 0.0), base_point=None,
                 tol: float = 1e-9):
    """Apply the D-H frame-assignment recipe to a chain of joint axes.

    ``points[i]`` is any point on joint axis ``i+1`` and ``directions[i]`` its direction (the axis along which a
    prismatic joint slides, or about which a revolute joint turns counter-clockwise), all expressed in
    one common "world" frame at one reference configuration. Returns ``(table, frames)``:

    * ``table`` - list of rows ``(alpha, a, d, theta, kind)`` in the requested convention. ``theta`` (revolute) or ``d``
      (prismatic) is the value of the row at the reference configuration, i.e. the *joint offset*;
    * ``frames`` - the list ``[T_0, T_1, ..., T_N]`` of 4x4 poses of the assigned frames in the world frame.

    Then ``frames[0] @ fk_dh(table, q, convention)[-1]`` is the pose of ``{N}`` in the world frame for joint values
    ``q`` measured *from the reference configuration*.

    Rules used (Craig, Ch. 3): ``Z_i`` along axis ``i``; ``X_i`` along the common normal from axis ``i`` to axis
    ``i+1`` (perpendicular to both axes if they intersect); for parallel axes the common normal through the
    foot of the previous normal, so that ``d_i = 0``; ``{0}`` coincides with ``{1}`` at the reference
    configuration; ``{N}`` has ``X_N = X_{N-1}``, with its origin at the foot of the last normal on axis ``N`` or,
    when ``tool_point`` is given, at the projection of that point onto axis ``N``. In the *standard* convention
    frame ``{i}`` has ``Z_i`` along axis ``i+1`` and ``Z_N`` along ``tool_axis`` (default: axis ``N``).

    ``base_point`` (a point on axis 1) moves the origin of the base frame ``{0}`` along axis 1, which makes ``d_1`` non-zero
    (for example, to put ``{0}`` on the mounting plate instead of at the shoulder).

    The directions of the ``X`` axes are a free choice that changes signs and joint offsets in the table (the
    robot is the same). ``x_sign="craig"`` always points ``X_i`` from axis ``i`` towards axis ``i+1``, so that every
    ``a_i >= 0``; ``x_sign="continuous"`` flips ``X_i`` when needed so that ``X_i . X_{i-1} >= 0`` (and ``X_1 . x_hint >= 0``),
    which keeps the joint offsets small and may give a negative ``a_i`` - the form in which many manufacturers
    publish their tables.
    """
    n = len(points)
    P = [np.asarray(p, dtype=float).ravel() for p in points]
    Z = [_unit(z) for z in directions]
    kinds = ["R"] * n if kinds is None else list(kinds)
    if x_sign not in ("craig", "continuous"):
        raise ValueError("x_sign must be 'craig' or 'continuous'")
    hint = None if x_sign == "craig" else np.asarray(x_hint, dtype=float)
    if convention == "modified":
        frames = _assign_modified(P, Z, tool_point, tol, hint)
    elif convention == "standard":
        frames = _assign_standard(P, Z, tool_point, tool_axis, tol, hint)
    else:
        raise ValueError("convention must be 'modified' or 'standard'")
    if base_point is not None:                               # slide {0} along axis 1 (gives d_1 != 0)
        o = np.asarray(base_point, dtype=float)
        z0 = frames[0][:3, 2]
        o0 = frames[0][:3, 3]
        frames[0] = frames[0].copy()
        frames[0][:3, 3] = o0 + np.dot(o - o0, z0) * z0
    table = table_from_frames(frames, convention, kinds)
    return table, frames


def _frame(x, z, o):
    x, z = _unit(x), _unit(z)
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), o)


def _assign_modified(P, Z, tool_point, tol, hint=None):
    """Frames {0}..{N} of the modified (proximal) convention."""
    n = len(P)
    O, X, foot = [None] * (n + 1), [None] * (n + 1), [None] * (n + 2)
    for i in range(1, n):                                    # frames {1} ... {N-1}: origin on axis i
        zi, zn, pi, pn = Z[i - 1], Z[i], P[i - 1], P[i]
        f1, f2, kind = common_normal(pi, zi, pn, zn, tol)
        if kind == "parallel":
            origin = pi if i == 1 else foot[i]               # through the foot of the previous normal: d_i = 0
            v = (pn - origin) - np.dot(pn - origin, zi) * zi
            O[i], a_len = origin, float(np.linalg.norm(v))
            X[i] = _unit(v) if a_len > 1e-9 else _perp_to(zi, X[i - 1] if i > 1 else None)
        else:
            O[i] = f1
            v = f2 - f1
            a_len = float(np.linalg.norm(v))
            X[i] = _unit(v) if a_len > 1e-9 else _unit(np.cross(zi, zn))
        foot[i + 1] = O[i] + a_len * X[i]
        if hint is not None:                                 # optional sign flip of X_i (a_i may become negative)
            ref = hint if i == 1 else X[i - 1]
            if np.dot(X[i], ref) < -1e-9:
                X[i] = -X[i]
    zN = Z[n - 1]
    if n == 1:
        X[1], foot_n = _perp_to(zN, hint), P[0]
    else:
        X[n], foot_n = X[n - 1], foot[n]
    O[n] = foot_n
    if tool_point is not None:
        O[n] = foot_n + np.dot(np.asarray(tool_point, float) - foot_n, zN) * zN
    frames = [None] + [_frame(X[i], Z[i - 1], O[i]) for i in range(1, n + 1)]
    frames[0] = _frame(X[1], Z[0], O[1] if n >= 2 else P[0])      # {0} coincides with {1} at the reference pose
    return frames


def _assign_standard(P, Z, tool_point, tool_axis, tol, hint=None):
    """Frames {0}..{N} of the standard (distal) convention; frame {i} has Z_i along axis i+1."""
    n = len(P)
    ZZ = list(Z) + [Z[n - 1] if tool_axis is None else _unit(tool_axis)]
    O, X = [None] * (n + 1), [None] * (n + 1)
    tp = None if tool_point is None else np.asarray(tool_point, float)
    for i in range(1, n + 1):
        zi, zn = ZZ[i - 1], ZZ[i]                            # axis i (through O_{i-1}) and axis i+1
        anchor = P[0] if i == 1 else O[i - 1]
        if i < n:
            pn = P[i]
        else:
            pn = tp if tp is not None else anchor
        f1, f2, kind = common_normal(anchor, zi, pn, zn, tol)
        if kind == "parallel":
            v = (pn - anchor) - np.dot(pn - anchor, zi) * zi
            if i == 1:
                O[0] = anchor
            X[i] = _unit(v) if np.linalg.norm(v) > 1e-9 else _perp_to(zi, X[i - 1] if i > 1 else None)
            O[i] = anchor + v
            if i == n and tp is not None:
                O[i] = O[i] + np.dot(tp - anchor, zi) * zi        # d_N: place {N} at the tool point
        else:
            if i == 1:
                O[0] = f1                                     # {0}: foot of the first normal on axis 1
            v = f2 - f1
            X[i] = _unit(v) if np.linalg.norm(v) > 1e-9 else _unit(np.cross(zi, zn))
            O[i] = f2
        if hint is not None and i < n:
            ref = hint if i == 1 else X[i - 1]
            if np.dot(X[i], ref) < -1e-9:
                X[i] = -X[i]
    X[0] = X[1]
    frames = [_frame(X[0], ZZ[0], O[0])]
    frames += [_frame(X[i], ZZ[i], O[i]) for i in range(1, n + 1)]
    return frames


# ---------------------------------------------------------------------------
# Modified <-> standard
# ---------------------------------------------------------------------------
def modified_to_standard(table, tool_alpha: float = 0.0, tool_a: float = 0.0):
    """Standard-convention table with the same forward kinematics as a modified-convention table.

    Because ``Rx(alpha) Dx(a) = Dx(a) Rx(alpha)``, a modified row ``(alpha_{i-1}, a_{i-1}, d_i, theta_i)`` and a
    standard row ``(alpha_i, a_i, d_i, theta_i)`` differ only in *which* ``(alpha, a)`` pair sits next to
    ``(d_i, theta_i)``: the standard table is the modified one with the ``(alpha, a)`` columns moved up one row.
    The first modified pair ``(alpha_0, a_0)`` must be zero (otherwise it is a fixed base transform and is
    returned separately in :func:`standard_to_modified`'s inverse sense); the last standard pair is
    ``(tool_alpha, tool_a)``.
    """
    rows = []
    n = len(table)
    for i, (al, a, d, th, k) in enumerate(table):
        if i == 0 and (abs(al) > 1e-12 or abs(a) > 1e-12):
            raise ValueError("alpha_0 and a_0 must be zero; absorb them in a fixed base transform")
        nxt = table[i + 1] if i + 1 < n else (tool_alpha, tool_a, 0, 0, "R")
        rows.append((nxt[0], nxt[1], d, th, k))
    return rows


def standard_to_modified(table):
    """Inverse of :func:`modified_to_standard`; returns ``(modified_table, (tool_alpha, tool_a))``."""
    rows = []
    for i, (al, a, d, th, k) in enumerate(table):
        prev = table[i - 1] if i > 0 else (0.0, 0.0, 0, 0, "R")
        rows.append((prev[0], prev[1], d, th, k))
    return rows, (table[-1][0], table[-1][1])


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------
def joint_axes_from_mujoco(model, data, joint_names=None):
    """Joint axes of a MuJoCo model at the current configuration, in the world frame.

    Returns ``(points, directions, kinds)``; ``points`` are the joint anchors ``data.xanchor`` and ``directions``
    the unit axes ``data.xaxis`` (both computed by ``mj_kinematics``). ``kinds`` is ``"R"`` for hinge and ``"P"`` for
    slide joints. Joints are listed in the order given by ``joint_names`` (default: all hinge and slide joints
    in model order).
    """
    import mujoco

    mujoco.mj_kinematics(model, data)
    if joint_names is None:
        ids = [j for j in range(model.njnt) if int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
    else:
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    pts = np.array([data.xanchor[j] for j in ids])
    dirs = np.array([data.xaxis[j] for j in ids])
    kinds = ["R" if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE) else "P" for j in ids]
    return pts, dirs, kinds


def mujoco_poses(model, data, qs, site=None, body=None, joint_names=None):
    """World poses (``(len(qs), 4, 4)``) of a site or body for every joint vector in ``qs``.

    ``qs`` has one row per configuration and one column per joint in ``joint_names`` (default: the first ``nq``
    entries of ``qpos``). Kinematics only (no dynamics) is evaluated; ``data`` is restored afterwards.
    """
    import mujoco

    saved = data.qpos.copy()
    if joint_names is None:
        adr = np.arange(np.asarray(qs).shape[1])
    else:
        adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names])
    out = []
    for q in np.atleast_2d(qs):
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        if site is not None:
            s = data.site(site)
            out.append(make_T(s.xmat.reshape(3, 3).copy(), s.xpos.copy()))
        else:
            b = data.body(body)
            out.append(make_T(b.xmat.reshape(3, 3).copy(), b.xpos.copy()))
    data.qpos[:] = saved
    mujoco.mj_kinematics(model, data)
    return np.array(out)


def pose_error(A, B):
    """Translation error (metres) and rotation error (radians) between pose arrays ``A`` and ``B``.

    Accepts single 4x4 matrices or arrays of shape ``(N, 4, 4)``; the rotation error is the angle of
    ``R_A^T R_B``.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    dt = np.linalg.norm(A[..., :3, 3] - B[..., :3, 3], axis=-1)
    Rrel = np.swapaxes(A[..., :3, :3], -1, -2) @ B[..., :3, :3]
    c = np.clip((np.trace(Rrel, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    # arccos is inaccurate near 0; use the skew part as well
    s = 0.5 * np.linalg.norm(np.stack([Rrel[..., 2, 1] - Rrel[..., 1, 2], Rrel[..., 0, 2] - Rrel[..., 2, 0],
                                       Rrel[..., 1, 0] - Rrel[..., 0, 1]], axis=-1), axis=-1)
    return dt, np.arctan2(s, c)


def mjcf_from_dh(table, convention: str = "modified", name: str = "dh_robot", radius: float = 0.012,
                 frame_length: float = 0.12, base_pos=(0, 0, 0), prismatic_length: float = 0.3,
                 site: str = "tool") -> str:
    """MJCF source of a stick-figure robot that realises a D-H table (one body per link).

    Row ``i`` of ``table`` becomes body ``link<i>``: its pose relative to the parent is the *constant* part of the link
    transform (modified convention: ``Rx(alpha) Dx(a) Rz(theta_0) Dz(d_0)``; standard convention: ``Rz(theta_0) Dz(d_0)``
    followed by a child body ``frame<i>`` carrying ``Dx(a) Rx(alpha)``) and the joint ``q<i>`` turns or slides along the
    body's own ``z`` axis. A site (default name ``tool``) marks the last frame. The result lets a reader *see* a
    table: the frame axes (red, green, blue = X, Y, Z) are drawn on every link and grey capsules join the origins. It is a
    visual check of a table, not an independent model of a physical robot.
    """
    n = len(table)
    f, r = frame_length, radius

    def quat(R):
        return " ".join(f"{v:.12g}" for v in rot_to_quat(R))

    def vec(v):
        return " ".join(f"{x:.9g}" for x in v)

    axes = (f'<geom type="capsule" fromto="0 0 0 {f} 0 0" size="{r / 2}" rgba="0.84 0.15 0.16 1" contype="0" conaffinity="0"/>'
            f'<geom type="capsule" fromto="0 0 0 0 {f} 0" size="{r / 2}" rgba="0.17 0.63 0.17 1" contype="0" conaffinity="0"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 {f}" size="{r / 2}" rgba="0.12 0.47 0.71 1" contype="0" conaffinity="0"/>')

    def rod(p0, p1, rgba="0.55 0.55 0.6 1"):
        if np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)) < 1e-9:
            return ""
        return f'<geom type="capsule" fromto="{vec(p0)} {vec(p1)}" size="{r}" rgba="{rgba}" contype="0" conaffinity="0"/>'

    opens, closes = [], []
    for i, (alpha, a, d, theta, kind) in enumerate(table, start=1):
        jtype = "hinge" if kind == "R" else "slide"
        if convention == "modified":
            static = make_T(rotx(alpha)) @ transl(a, 0, 0) @ make_T(rotz(theta)) @ transl(0, 0, d)
        else:
            static = make_T(rotz(theta)) @ transl(0, 0, d)
        # offset segment along the joint axis (revolute: the length d_i; prismatic: a guide rod)
        if kind == "R":
            geoms = rod((0, 0, -d), (0, 0, 0))
        else:
            geoms = rod((0, 0, -prismatic_length), (0, 0, 0), "0.9 0.6 0.2 1")
        if convention == "modified" and i < n:
            geoms += rod((0, 0, 0), (table[i][1], 0, 0))        # the common normal a_i leaves along X_i
        body = f'<body name="link{i}" pos="{vec(static[:3, 3])}" quat="{quat(static[:3, :3])}">'
        body += f'<joint name="q{i}" type="{jtype}" axis="0 0 1"/>' + axes + geoms
        if convention == "standard":
            child = transl(a, 0, 0) @ make_T(rotx(alpha))
            body += rod((0, 0, 0), (a, 0, 0))
            body += f'<body name="frame{i}" pos="{vec(child[:3, 3])}" quat="{quat(child[:3, :3])}">' + axes
            closes.append("</body></body>")
        else:
            closes.append("</body>")
        opens.append(body)
    inner = "".join(opens) + f'<site name="{site}" size="0.01" rgba="1 0.5 0 1"/>' + "".join(reversed(closes))
    return (f'<mujoco model="{name}"><compiler angle="radian"/><worldbody>'
            f'<body name="base" pos="{vec(base_pos)}">'
            f'<geom type="cylinder" size="0.05 0.005" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>{inner}</body>'
            f'</worldbody></mujoco>')


# ---------------------------------------------------------------------------
# Symbolic forward kinematics (SymPy)
# ---------------------------------------------------------------------------
def sym_dh_modified(alpha, a, d, theta):
    """SymPy ``Matrix`` of Craig's link transform; arguments may be symbols or numbers (``sympy.pi/2`` is exact)."""
    import sympy as sp

    ca, sa, ct, st = sp.cos(alpha), sp.sin(alpha), sp.cos(theta), sp.sin(theta)
    return sp.Matrix([[ct, -st, 0, a],
                      [st * ca, ct * ca, -sa, -sa * d],
                      [st * sa, ct * sa, ca, ca * d],
                      [0, 0, 0, 1]])


def sym_dh_standard(alpha, a, d, theta):
    """SymPy ``Matrix`` of the standard (Spong) link transform."""
    import sympy as sp

    ca, sa, ct, st = sp.cos(alpha), sp.sin(alpha), sp.cos(theta), sp.sin(theta)
    return sp.Matrix([[ct, -st * ca, st * sa, a * ct],
                      [st, ct * ca, -ct * sa, a * st],
                      [0, sa, ca, d],
                      [0, 0, 0, 1]])


def _simp(M):
    import sympy as sp

    return M.applyfunc(lambda e: sp.simplify(sp.trigsimp(sp.expand(e))))


def sym_link_transforms(table, convention: str = "modified", simplify: bool = True):
    """List of the individual symbolic link transforms ``^{i-1}_i T``; ``table`` rows are ``(alpha, a, d, theta[, kind])``."""
    f = sym_dh_modified if convention == "modified" else sym_dh_standard
    out = []
    for row in table:
        M = f(*row[:4])
        out.append(_simp(M) if simplify else M)
    return out


def sym_fk(table, convention: str = "modified", simplify: bool = True):
    """Cumulative symbolic transforms ``[^0_1 T, ^0_2 T, ..., ^0_N T]`` of a table of SymPy entries.

    Each partial product is simplified (``trigsimp``) so that sums of angles appear as ``cos(theta_1 + theta_2)``;
    see :func:`cs_substitute` for Craig's compact ``c_{12}`` notation. The table rows must already contain the joint
    variables (for example ``theta`` = ``theta_1`` or ``theta_2 + sp.pi/2``; ``d`` = ``d_2`` for a prismatic joint).
    """
    import sympy as sp

    Ts, T = [], sp.eye(4)
    for M in sym_link_transforms(table, convention, simplify):
        T = T * M
        if simplify:
            T = _simp(T)
        Ts.append(T)
    return Ts


def _angle_name(arg, sym_index):
    """Subscript for a unit-coefficient combination of angle symbols: ``12``, ``123``, ``12-4``, ``-1+2``; else ``None``."""
    import sympy as sp

    parts = []
    for t in sp.Add.make_args(sp.expand(arg)):
        c, sym = t.as_coeff_Mul()
        if sym in sym_index and c in (1, -1):
            parts.append((sym_index[sym], int(c)))
        else:
            return None
    parts.sort()
    name, prev = "", None
    for idx, sign in parts:
        if sign < 0:
            name += "-"
        elif prev is not None and prev < 0:
            name += "+"
        name += str(idx)
        prev = sign
    return name


def cs_substitute(expr, angle_symbols):
    """Replace ``sin``/``cos`` of sums of the given angle symbols by compact symbols ``s_{12}``, ``c_{1}``...

    ``angle_symbols`` maps SymPy symbols to the index shown in the subscript: ``{theta_1: 1, theta_2: 2}``
    (a list of symbols is numbered from 1). Arguments that are not a unit-coefficient combination of these symbols
    (for instance ``theta_2 + pi/2``) are left unchanged.
    """
    import sympy as sp

    if not isinstance(angle_symbols, dict):
        angle_symbols = {s: i + 1 for i, s in enumerate(angle_symbols)}

    def rep(f):
        name = _angle_name(f.args[0], angle_symbols)
        if name is None:
            return f
        return sp.Symbol(("s" if isinstance(f, sp.sin) else "c") + "_{" + name + "}")

    return expr.replace(lambda e: isinstance(e, (sp.sin, sp.cos)), rep)


def latex_cs(obj, angle_symbols=None) -> str:
    """LaTeX of a SymPy expression or matrix; sines and cosines of angle sums become ``c_{12}`` / ``s_{23}``."""
    import sympy as sp

    if angle_symbols is not None:
        if hasattr(obj, "applyfunc"):
            obj = obj.applyfunc(lambda e: cs_substitute(e, angle_symbols))
        else:
            obj = cs_substitute(obj, angle_symbols)
    return sp.latex(obj)


def lambdify_fk(T_expr, symbols):
    """Fast numerical function ``f(*values) -> 4x4 ndarray`` from a symbolic transform."""
    import sympy as sp

    f = sp.lambdify(list(symbols), T_expr, modules="numpy")
    return lambda *args: np.array(f(*args), dtype=float)
