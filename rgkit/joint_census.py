"""Reading the joints of a MuJoCo model: types, axes, chains, and mobility.

Chapter 1 asks "what joints does this robot have, and how many degrees of freedom?". These helpers answer the
question *from the simulator's own model*, so that the answer does not depend on a catalogue or on the reader's
eyes:

* :func:`joint_table` — name, type, body, axis and anchor in the world frame, range, and the position
  (``qpos``) and velocity (``qvel``) addresses of every joint;
* :func:`chain_joints` — the joints from the world to a given body, in order (the *serial chain* to that body);
* :func:`joint_symbol` — ``R``, ``P``, ``S`` (ball) or ``F`` (free) for a joint;
* :func:`axis_relation` — shortest distance and angle between two joint axes (the *link length* and *twist*
  that Chapter 3 will call ``a`` and ``alpha``);
* :func:`equality_rank` and :func:`model_mobility` — ``nv`` minus the rank of the equality-constraint Jacobian,
  which is Grübler's count carried out by the simulator.

``mujoco`` is imported on demand so that the rest of ``rgkit`` works without it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["JOINT_TYPES", "joint_table", "joint_symbol", "chain_joints", "axis_relation", "axis_kind",
           "equality_rank", "model_mobility", "kinematic_tree", "describe_dofs", "project_points"]

#: ``mjtJoint`` value -> name.
JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
_SYMBOL = {"free": "F", "ball": "S", "slide": "P", "hinge": "R"}
_NQ = {"free": 7, "ball": 4, "slide": 1, "hinge": 1}
_NV = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}


def _mj():
    import mujoco
    return mujoco


def _name(model, kind, idx):
    mj = _mj()
    n = mj.mj_id2name(model, kind, idx)
    return n if n else f"<{mj.mjtObj(kind).name[6:].lower()} {idx}>"


def joint_table(model, data):
    """List of dicts, one per joint, evaluated at the configuration stored in ``data``.

    Keys: ``id, name, type, symbol, body, parent, axis`` (world frame, unit), ``anchor`` (world), ``range``
    (``None`` if the joint is unlimited), ``qposadr, dofadr, nq, nv``. ``data`` must have had ``mj_forward``
    called.
    """
    mj = _mj()
    rows = []
    for j in range(model.njnt):
        typ = JOINT_TYPES[int(model.jnt_type[j])]
        b = int(model.jnt_bodyid[j])
        rows.append(dict(
            id=j, name=_name(model, mj.mjtObj.mjOBJ_JOINT, j), type=typ, symbol=_SYMBOL[typ],
            body=_name(model, mj.mjtObj.mjOBJ_BODY, b), parent=_name(model, mj.mjtObj.mjOBJ_BODY, int(model.body_parentid[b])),
            axis=np.array(data.xaxis[j]), anchor=np.array(data.xanchor[j]),
            range=(tuple(model.jnt_range[j]) if model.jnt_limited[j] else None),
            qposadr=int(model.jnt_qposadr[j]), dofadr=int(model.jnt_dofadr[j]), nq=_NQ[typ], nv=_NV[typ],
        ))
    return rows


def joint_symbol(row_or_type) -> str:
    """``R`` (hinge), ``P`` (slide), ``S`` (ball), ``F`` (free) for a table row or a type name."""
    typ = row_or_type["type"] if isinstance(row_or_type, dict) else row_or_type
    return _SYMBOL[typ]


def chain_joints(model, data, body):
    """Joints on the path from the world to ``body`` (a name or id), ordered from the base outwards."""
    mj = _mj()
    b = body if isinstance(body, (int, np.integer)) else mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
    path = []
    while b > 0:
        path.append(b)
        b = int(model.body_parentid[b])
    rows = joint_table(model, data)
    out = []
    for body_id in reversed(path):                       # from the base to the tip; several joints in one body
        out += [r for r, j in zip(rows, range(model.njnt)) if model.jnt_bodyid[j] == body_id]   # keep model order
    return out


def axis_relation(p1, u1, p2, u2):
    """Shortest distance and angle between two lines ``p + t u``.

    Returns ``(distance, angle)`` with the angle in ``[0, pi/2]`` between the *lines* (direction signs ignored).
    For two joint axes these are the **link length** ``a`` (distance between the axes) and the **link twist**
    ``alpha`` (angle between them) of Denavit and Hartenberg.
    """
    p1, u1, p2, u2 = (np.asarray(v, float) for v in (p1, u1, p2, u2))
    u1, u2 = u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)
    angle = float(np.arccos(min(1.0, abs(u1 @ u2))))
    cross = np.cross(u1, u2)
    if np.linalg.norm(cross) < 1e-9:                       # parallel lines
        d = (p2 - p1) - ((p2 - p1) @ u1) * u1
        return float(np.linalg.norm(d)), angle
    return float(abs((p2 - p1) @ cross) / np.linalg.norm(cross)), angle


def axis_kind(distance, angle, tol=1e-6):
    """Verbal description of two joint axes: parallel / perpendicular / intersecting / skew (offset)."""
    par = angle < 1e-6
    perp = abs(angle - np.pi / 2) < 1e-6
    meet = distance < tol
    if par:
        return "coincident" if meet else "parallel"
    if meet:
        return "perpendicular, intersecting" if perp else "intersecting at an angle"
    return "perpendicular, offset" if perp else "skew"


def kinematic_tree(model):
    """Parent map ``{body name: [children]}`` of the bodies of ``model`` (world is the root)."""
    mj = _mj()
    tree = {}
    for b in range(model.nbody):
        tree.setdefault(_name(model, mj.mjtObj.mjOBJ_BODY, b), [])
    for b in range(1, model.nbody):
        tree[_name(model, mj.mjtObj.mjOBJ_BODY, int(model.body_parentid[b]))].append(_name(model, mj.mjtObj.mjOBJ_BODY, b))
    return tree


def equality_rank(model, data, tol=1e-9):
    """Rank of the Jacobian of MuJoCo's *equality* constraints at the current state (``0`` if there are none).

    The model's Jacobian is made dense for the call. ``data`` must contain a state that satisfies the constraints.
    """
    mj = _mj()
    old = model.opt.jacobian
    model.opt.jacobian = mj.mjtJacobian.mjJAC_DENSE
    mj.mj_forward(model, data)
    rows = np.where(np.asarray(data.efc_type) == mj.mjtConstraint.mjCNSTR_EQUALITY)[0]
    rank = 0
    if rows.size:
        J = np.asarray(data.efc_J).reshape(data.nefc, model.nv)[rows]
        s = np.linalg.svd(J, compute_uv=False)
        rank = int(np.sum(s > tol * max(1.0, s[0])))
    model.opt.jacobian = old
    return rank


def model_mobility(model, data):
    """Mobility ``nv - rank(equality Jacobian)`` of a MuJoCo model, i.e. Grübler's count done by the simulator."""
    return int(model.nv - equality_rank(model, data))


def describe_dofs(model):
    """``(nq, nv, nu, neq)`` — position coordinates, velocity coordinates (DOF), actuators, equality constraints."""
    return int(model.nq), int(model.nv), int(model.nu), int(model.neq)


def project_points(model, camera, points, width=480, height=360):
    """Pixel coordinates ``(u, v)`` of 3-D world ``points`` seen by a free MuJoCo camera.

    ``camera`` is the dict accepted by :func:`rgkit.mujoco_utils.render` (``azimuth``, ``elevation``,
    ``distance``, ``lookat``). Used to put labels on rendered images. Returns an ``(n, 2)`` array; points behind the
    camera give ``nan``.
    """
    az, el = np.deg2rad(camera.get("azimuth", 90.0)), np.deg2rad(camera.get("elevation", -45.0))
    dist = camera.get("distance", 3.0)
    look = np.asarray(camera.get("lookat", (0.0, 0.0, 0.0)), float)
    f = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])      # viewing direction
    pos = look - dist * f
    right = np.cross(f, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    fovy = np.deg2rad(float(model.vis.global_.fovy))
    scale = (height / 2) / np.tan(fovy / 2)
    P = np.atleast_2d(np.asarray(points, float)) - pos
    depth = P @ f
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2 + scale * (P @ right) / depth
        v = height / 2 - scale * (P @ up) / depth
    u[depth <= 0] = v[depth <= 0] = np.nan
    return np.column_stack([u, v])
