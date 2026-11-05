"""Mobility of mechanisms: Grübler's formula, rigidity, and numerical loop closure.

This module supports Chapter 1 of MFoR (joints and degrees of freedom). It contains

* the six lower pairs and the Grübler–Kutzbach formula (:func:`grubler`),
* the number of independent loops of a kinematic graph (:func:`independent_loops`),
* a numerical measure of mobility, ``n - rank(dPhi/dq)`` (:func:`numerical_mobility`), which checks the formula
  independently of any counting argument,
* the rigidity matrix of a set of points (:func:`rigidity_dof`), the computational form of "count the points and
  subtract the distance constraints",
* a small solver for **planar linkages** built from revolute and prismatic joints (:class:`PlanarLinkage`),
  together with ready-made four-bar, slider-crank and coupling-rod linkages,
* closed-form positions of the four-bar and the slider-crank, and the Grashof criterion,
* a Stewart–Gough platform (leg lengths and the numerical mobility of the platform).

Conventions: planar poses are ``(x, y, theta)`` with ``theta`` counter-clockwise in radians; body ``0`` is the
ground and is fixed at ``(0, 0, 0)``. Nothing here depends on MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "LOWER_PAIRS", "grubler", "grubler_planar", "grubler_spatial", "independent_loops",
    "numerical_mobility", "rigidity_dof", "PlanarLinkage", "fourbar_closure", "fourbar_points",
    "fourbar_linkage", "slider_crank_x", "slider_crank_points", "slider_crank_linkage",
    "coupling_rod_linkage", "grashof", "grashof_type", "stewart_geometry", "stewart_leg_lengths",
    "stewart_mobility", "bennett_residual", "bennett_parameters",
]

# --------------------------------------------------------------------------------------------------------------
# Joints and Grübler's formula
# --------------------------------------------------------------------------------------------------------------

#: The six lower pairs: ``name -> (symbol, freedom f, relative motion allowed)``. In space each pair removes
#: ``6 - f`` of the 6 relative freedoms of two free bodies.
LOWER_PAIRS = {
    "revolute": ("R", 1, "rotation about a fixed axis"),
    "prismatic": ("P", 1, "translation along a fixed line"),
    "helical": ("H", 1, "rotation coupled to translation along the same axis (pitch h)"),
    "cylindrical": ("C", 2, "independent rotation about and translation along one axis"),
    "spherical": ("S", 3, "any rotation about a fixed point"),
    "planar": ("E", 3, "two translations in a plane and a rotation about its normal"),
}


def grubler(N: int, freedoms, m: int = 6) -> int:
    """Grübler–Kutzbach mobility ``m (N - 1 - J) + sum f_i``.

    ``N`` counts **all** links including the ground, ``freedoms`` lists the freedom ``f_i`` of every joint
    (so ``J = len(freedoms)``) and ``m`` is 3 for planar and 6 for spatial mechanisms.
    """
    freedoms = list(freedoms)
    return int(m * (N - 1 - len(freedoms)) + sum(freedoms))


def grubler_planar(N: int, freedoms) -> int:
    """Grübler's formula with ``m = 3`` (all links and joints move in one plane)."""
    return grubler(N, freedoms, 3)


def grubler_spatial(N: int, freedoms) -> int:
    """Grübler's formula with ``m = 6``."""
    return grubler(N, freedoms, 6)


def independent_loops(N: int, J: int, components: int = 1) -> int:
    """Number of independent kinematic loops ``J - N + components`` (cycle rank of the link–joint graph)."""
    return int(J - N + components)


# --------------------------------------------------------------------------------------------------------------
# Numerical mobility
# --------------------------------------------------------------------------------------------------------------

def numerical_mobility(fun, x, h: float = 1e-6, tol: float = 1e-6):
    """Instantaneous mobility of the constraint set ``fun(x) = 0`` at a point ``x`` that satisfies it.

    Returns ``(dof, singular_values)`` where ``dof = len(x) - rank(J)`` and ``J = d fun / d x`` is formed by
    central differences. A singular value counts as zero if it is below ``tol`` times the largest one.
    """
    x = np.asarray(x, float)
    cols = []
    for k in range(len(x)):
        e = np.zeros_like(x); e[k] = h
        cols.append((np.asarray(fun(x + e)) - np.asarray(fun(x - e))) / (2 * h))
    J = np.array(cols).T
    s = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(s > tol * max(s[0], 1.0))) if s.size else 0
    return len(x) - rank, s


def rigidity_dof(points, edges=None, tol: float = 1e-9) -> int:
    """Degrees of freedom of ``n`` points in the plane or in space whose pairwise distances (``edges``) are fixed.

    ``edges`` defaults to *all* pairs. The count is ``dim * n - rank(R)`` with the rigidity matrix ``R`` (one row
    ``p_i - p_j`` per edge). For ``n >= dim`` points **in general position** this equals ``dim (dim + 1) / 2``:
    3 in the plane and 6 in space. (For degenerate sets, e.g. collinear points, the first-order test reports extra
    freedoms that disappear at second order, so use it only for points in general position.)
    """
    P = np.asarray(points, float)
    n, dim = P.shape
    if edges is None:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    R = np.zeros((len(edges), n * dim))
    for r, (i, j) in enumerate(edges):
        R[r, i * dim:(i + 1) * dim] = P[i] - P[j]
        R[r, j * dim:(j + 1) * dim] = P[j] - P[i]
    s = np.linalg.svd(R, compute_uv=False) if len(edges) else np.array([])
    rank = int(np.sum(s > tol * max(1.0, s[0]))) if s.size else 0
    return n * dim - rank


# --------------------------------------------------------------------------------------------------------------
# Planar linkages (revolute and prismatic joints), solved numerically
# --------------------------------------------------------------------------------------------------------------

def _rot2(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


@dataclass
class PlanarLinkage:
    """Planar linkage of ``n`` moving rigid bodies (numbered 1..n) plus the fixed ground (body 0).

    The configuration vector is ``q = [x1, y1, th1, x2, y2, th2, ...]``. Joints are added with
    :meth:`add_revolute` (2 equations) and :meth:`add_prismatic` (2 equations); :meth:`residual` collects them.
    """

    n: int
    joints: list = field(default_factory=list)

    # -- construction ---------------------------------------------------------------------------------------
    def add_revolute(self, i, p_i, j, p_j):
        """Pin: point ``p_i`` of body ``i`` coincides with point ``p_j`` of body ``j`` (body coordinates)."""
        self.joints.append(("R", i, np.asarray(p_i, float), j, np.asarray(p_j, float), None, 0.0))
        return self

    def add_prismatic(self, i, p_i, u_i, j, p_j, dtheta=0.0):
        """Slider: ``p_j`` of body ``j`` moves on the line through ``p_i`` of body ``i`` with direction ``u_i``
        (body-``i`` coordinates), and the bodies keep the relative angle ``th_j - th_i = dtheta``."""
        u = np.asarray(u_i, float)
        self.joints.append(("P", i, np.asarray(p_i, float), j, np.asarray(p_j, float), u / np.linalg.norm(u), dtheta))
        return self

    # -- kinematics -----------------------------------------------------------------------------------------
    @property
    def n_coords(self):
        return 3 * self.n

    @property
    def N(self):
        """Number of links including the ground."""
        return self.n + 1

    @property
    def freedoms(self):
        return [1] * len(self.joints)

    def grubler(self):
        """Planar Grübler count ``3 (N - 1 - J) + sum f_i`` of this linkage."""
        return grubler_planar(self.N, self.freedoms)

    def pose(self, q, i):
        return np.zeros(3) if i == 0 else np.asarray(q, float)[3 * (i - 1):3 * i]

    def point(self, q, i, p):
        """World coordinates of body-``i`` point ``p``."""
        x, y, th = self.pose(q, i)
        return np.array([x, y]) + _rot2(th) @ np.asarray(p, float)

    def residual(self, q):
        out = []
        for kind, i, pi, j, pj, u, dth in self.joints:
            if kind == "R":
                out.extend(self.point(q, i, pi) - self.point(q, j, pj))
            else:
                ti, tj = self.pose(q, i)[2], self.pose(q, j)[2]
                nrm = _rot2(ti) @ np.array([-u[1], u[0]])
                out.append(nrm @ (self.point(q, j, pj) - self.point(q, i, pi)))
                out.append(tj - ti - dth)
        return np.array(out)

    def jacobian(self, q, h=1e-7):
        q = np.asarray(q, float)
        cols = []
        for k in range(len(q)):
            e = np.zeros_like(q); e[k] = h
            cols.append((self.residual(q + e) - self.residual(q - e)) / (2 * h))
        return np.array(cols).T

    def mobility(self, q, tol=1e-6):
        """``(dof, singular_values)``: ``n_coords - rank(dPhi/dq)`` at the configuration ``q``."""
        return numerical_mobility(self.residual, q, tol=tol)

    def solve(self, q0, fixed=None, tol=1e-12, maxit=100):
        """Newton (Gauss–Newton) solution of ``residual(q) = 0`` from ``q0``.

        ``fixed`` maps coordinate indices to prescribed values (the *drivers*, for example the crank angle).
        Raises ``RuntimeError`` if the iteration does not converge.
        """
        q = np.array(q0, float)
        fixed = dict(fixed or {})
        for k, v in fixed.items():
            q[k] = v
        for _ in range(maxit):
            r = self.residual(q)
            if np.linalg.norm(r) < tol:
                return q
            J = self.jacobian(q)
            free = [k for k in range(len(q)) if k not in fixed]
            dq = np.linalg.lstsq(J[:, free], -r, rcond=None)[0]
            q[free] += dq
        if np.linalg.norm(self.residual(q)) > 1e-8:
            raise RuntimeError("loop closure did not converge")
        return q


# -- four-bar ---------------------------------------------------------------------------------------------------

def fourbar_closure(a, b, c, d, th2, branch=+1):
    """Closed-form loop closure of the planar four-bar.

    Ground pivots ``O2 = (0, 0)`` and ``O4 = (d, 0)``; crank ``a`` (angle ``th2``), coupler ``b``, rocker ``c``.
    Returns ``(th3, th4)`` — the absolute angles of coupler and rocker — or ``None`` if the linkage cannot be
    assembled for this crank angle. ``branch = +1/-1`` selects one of the two assembly modes.
    """
    A = a * np.array([np.cos(th2), np.sin(th2)])
    O4 = np.array([d, 0.0])
    D = np.linalg.norm(O4 - A)
    if D > b + c + 1e-12 or D < abs(b - c) - 1e-12 or D == 0:
        return None
    s = (b * b - c * c + D * D) / (2 * D)
    h = np.sqrt(max(b * b - s * s, 0.0))
    u = (O4 - A) / D
    B = A + s * u + branch * h * np.array([-u[1], u[0]])
    return float(np.arctan2(*(B - A)[::-1])), float(np.arctan2(*(B - O4)[::-1]))


def fourbar_points(a, b, c, d, th2, branch=+1):
    """Joint positions ``[O2, A, B, O4]`` (4 x 2 array) of the four-bar, or ``None`` if not assemblable."""
    sol = fourbar_closure(a, b, c, d, th2, branch)
    if sol is None:
        return None
    th3, _ = sol
    A = a * np.array([np.cos(th2), np.sin(th2)])
    B = A + b * np.array([np.cos(th3), np.sin(th3)])
    return np.array([[0.0, 0.0], A, B, [d, 0.0]])


def fourbar_linkage(a, b, c, d, th2, branch=+1):
    """Four-bar as a :class:`PlanarLinkage` (crank = body 1, coupler = 2, rocker = 3) with its assembled ``q``."""
    th3, th4 = fourbar_closure(a, b, c, d, th2, branch)
    A = a * np.array([np.cos(th2), np.sin(th2)])
    L = (PlanarLinkage(3)
         .add_revolute(0, (0, 0), 1, (0, 0))
         .add_revolute(1, (a, 0), 2, (0, 0))
         .add_revolute(2, (b, 0), 3, (c, 0))
         .add_revolute(3, (0, 0), 0, (d, 0)))
    q = np.array([0, 0, th2, A[0], A[1], th3, d, 0, th4], float)
    return L, q


def grashof(a, b, c, d):
    """Grashof's condition ``s + l <= p + q`` (shortest + longest <= sum of the other two) for link lengths."""
    s = sorted([a, b, c, d])
    return s[0] + s[3] <= s[1] + s[2] + 1e-12


def grashof_type(a, b, c, d):
    """Classify the four-bar (ground ``d``, crank ``a``, coupler ``b``, rocker ``c``) by Grashof's rules."""
    if not grashof(a, b, c, d):
        return "non-Grashof (double-rocker: no link can rotate fully)"
    lengths = {"crank": a, "coupler": b, "rocker": c, "ground": d}
    shortest = min(lengths, key=lengths.get)
    return {"crank": "Grashof, crank-rocker (crank is the shortest link)",
            "rocker": "Grashof, rocker-crank (rocker is the shortest link)",
            "ground": "Grashof, double-crank (ground is the shortest link)",
            "coupler": "Grashof, double-rocker (coupler is the shortest link)"}[shortest]


# -- slider-crank -----------------------------------------------------------------------------------------------

def slider_crank_x(r, l, th, e=0.0, branch=+1):
    """Piston position ``x = r cos th + sqrt(l^2 - (r sin th - e)^2)`` of a slider-crank (offset ``e``)."""
    return r * np.cos(th) + branch * np.sqrt(l * l - (r * np.sin(th) - e) ** 2)


def slider_crank_points(r, l, th, e=0.0, branch=+1):
    """Positions ``[O, crank pin, piston pin]`` of the slider-crank."""
    pin = r * np.array([np.cos(th), np.sin(th)])
    return np.array([[0.0, 0.0], pin, [slider_crank_x(r, l, th, e, branch), e]])


def slider_crank_linkage(r, l, th, e=0.0):
    """Slider-crank as a :class:`PlanarLinkage` (crank 1, connecting rod 2, piston 3) with its assembled ``q``."""
    x = slider_crank_x(r, l, th, e)
    pin = r * np.array([np.cos(th), np.sin(th)])
    th2 = np.arctan2(e - pin[1], x - pin[0])
    L = (PlanarLinkage(3)
         .add_revolute(0, (0, 0), 1, (0, 0))
         .add_revolute(1, (r, 0), 2, (0, 0))
         .add_revolute(2, (l, 0), 3, (0, 0))
         .add_prismatic(0, (0, e), (1, 0), 3, (0, 0), 0.0))
    q = np.array([0, 0, th, pin[0], pin[1], th2, x, e, 0.0], float)
    return L, q


def coupling_rod_linkage(r=0.5, L=1.0, phi=0.6):
    """Three equal cranks joined by one coupling rod (a locomotive's coupled wheels), as a :class:`PlanarLinkage`.

    Ground pivots ``(kL, 0)``, k = 0, 1, 2; all cranks have length ``r`` and the rod carries three pins spaced
    ``L`` apart, so every loop is a parallelogram. Grübler gives 0 but the mechanism has one degree of freedom.
    Bodies: cranks 1-3 (frames at the pivots), coupling rod 4.
    """
    mech = PlanarLinkage(4)
    q = []
    for k in range(3):
        mech.add_revolute(0, (k * L, 0), k + 1, (0, 0))
        q += [k * L, 0.0, phi]
    for k in range(3):
        mech.add_revolute(k + 1, (r, 0), 4, (k * L, 0))
    q += [r * np.cos(phi), r * np.sin(phi), 0.0]
    return mech, np.array(q, float)


# --------------------------------------------------------------------------------------------------------------
# Stewart–Gough platform
# --------------------------------------------------------------------------------------------------------------

def stewart_geometry(base_radius=1.0, platform_radius=0.6, half_gap=np.deg2rad(10.0)):
    """Anchor points of a 6-6 Stewart-Gough platform: ``(base_points, platform_points)``, each ``6 x 3``.

    The base anchors form three close pairs centred at 0, 120 and 240 degrees; the platform anchors form three
    pairs centred at 60, 180 and 300 degrees, so that leg ``(k, s)`` joins the base anchor at ``120k + s g`` to the
    platform anchor at ``120k + s (60 deg - g)`` (``g`` = ``half_gap``). Platform anchors are given in the
    platform's own frame (z = 0), base anchors in the world frame.
    """
    base, plat = [], []
    for k in range(3):
        for s in (-1, 1):
            a_b = 2 * np.pi * k / 3 + s * half_gap
            a_p = 2 * np.pi * k / 3 + s * (np.pi / 3 - half_gap)
            base.append([base_radius * np.cos(a_b), base_radius * np.sin(a_b), 0.0])
            plat.append([platform_radius * np.cos(a_p), platform_radius * np.sin(a_p), 0.0])
    return np.array(base), np.array(plat)


def _rotvec_to_matrix(w):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(w, float)).as_matrix()


def stewart_leg_lengths(base_pts, plat_pts, position, rotvec):
    """Six leg lengths for the platform pose (``position`` and rotation vector ``rotvec``)."""
    R = _rotvec_to_matrix(rotvec)
    tips = plat_pts @ R.T + np.asarray(position, float)
    return np.linalg.norm(tips - base_pts, axis=1)


def stewart_mobility(base_pts, plat_pts, position, rotvec, locked_legs=()):
    """Numerical mobility of the platform when the legs in ``locked_legs`` have fixed length.

    Variables: the platform pose (6) and the six leg lengths; constraints: ``|tip_i - base_i|^2 = l_i^2`` for
    all six legs, plus ``l_i = l_i(0)`` for locked legs. Returns the integer ``dof``.
    """
    l0 = stewart_leg_lengths(base_pts, plat_pts, position, rotvec)
    x0 = np.concatenate([position, rotvec, l0])

    def fun(x):
        pos, rv, ell = x[:3], x[3:6], x[6:]
        R = _rotvec_to_matrix(rv)
        tips = plat_pts @ R.T + pos
        res = list(np.sum((tips - base_pts) ** 2, axis=1) - ell ** 2)
        res += [ell[i] - l0[i] for i in locked_legs]
        return np.array(res)

    return numerical_mobility(fun, x0)[0]


# --------------------------------------------------------------------------------------------------------------
# Bennett linkage (a paradoxical spatial four-bar)
# --------------------------------------------------------------------------------------------------------------

def bennett_parameters(a=1.0, alpha=np.deg2rad(40.0), beta=np.deg2rad(70.0)):
    """Link parameters ``(a, b, alpha, beta)`` of a Bennett linkage.

    Opposite links are equal: lengths ``a, b, a, b`` and twists ``alpha, beta, alpha, beta``; the offsets vanish,
    and Bennett's condition ``a / sin(alpha) = b / sin(beta)`` fixes ``b``. Needs ``alpha != beta``.
    """
    return a, a * np.sin(beta) / np.sin(alpha), alpha, beta


def bennett_residual(thetas, params):
    """12 entries of ``T_1 T_2 T_3 T_4 - I`` for four revolute joints with zero offsets (modified D-H, Craig).

    ``thetas`` are the four joint angles; ``params = (a, b, alpha, beta)``. Bennett's linkage is the special
    geometry for which this closure equation has a one-parameter family of solutions.
    """
    from .kinematics import dh_modified

    a, b, alpha, beta = params
    rows = [(beta, b, thetas[0]), (alpha, a, thetas[1]), (beta, b, thetas[2]), (alpha, a, thetas[3])]
    T = np.eye(4)
    for al, ln, th in rows:
        T = T @ dh_modified(al, ln, 0.0, th)
    return (T - np.eye(4))[:3, :].ravel()
