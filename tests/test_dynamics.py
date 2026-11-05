"""Tests for rgkit.dynamics (MFoR Chapter 6)."""
import numpy as np
import pytest

from rgkit import dynamics as dyn
from rgkit import transforms as tf

rng = np.random.default_rng(7)


# --------------------------------------------------------------------------- mass distribution
def _grid_inertia(inside, lo, hi, m, n=70):
    """Inertia tensor of a uniform solid by midpoint-rule integration on an n^3 grid (about the grid origin)."""
    ax = [lo[i] + (np.arange(n) + 0.5) * (hi[i] - lo[i]) / n for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    mask = inside(X, Y, Z)
    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
    return dyn.inertia_point_masses(np.full(len(pts), m / len(pts)), pts), pts.mean(axis=0)


def test_primitives_against_numerical_integration():
    m = 2.0
    cases = {
        "box": (dyn.inertia_box(m, 0.6, 0.4, 0.2), lambda X, Y, Z: (abs(X) <= 0.3) & (abs(Y) <= 0.2) & (abs(Z) <= 0.1), [-.3, -.2, -.1], [.3, .2, .1]),
        "cylinder": (dyn.inertia_cylinder(m, 0.2, 0.5, axis=1), lambda X, Y, Z: X ** 2 + Z ** 2 <= 0.04, [-.2, -.25, -.2], [.2, .25, .2]),
        "sphere": (dyn.inertia_sphere(m, 0.3), lambda X, Y, Z: X ** 2 + Y ** 2 + Z ** 2 <= 0.09, [-.3] * 3, [.3] * 3),
        "ellipsoid": (dyn.inertia_ellipsoid(m, 0.4, 0.3, 0.2), lambda X, Y, Z: (X / .4) ** 2 + (Y / .3) ** 2 + (Z / .2) ** 2 <= 1, [-.4, -.3, -.2], [.4, .3, .2]),
    }
    for name, (I, inside, lo, hi) in cases.items():
        I_num, com = _grid_inertia(inside, lo, hi, m)
        assert np.allclose(com, 0, atol=2e-3), name
        assert np.allclose(I, I_num, rtol=0.02, atol=2e-4), name


def test_capsule_equals_cylinder_plus_hemispheres():
    m, r, h = 1.3, 0.05, 0.4
    I = dyn.inertia_capsule(m, r, h, axis=0)
    assert I[0, 0] < I[1, 1] and np.isclose(I[1, 1], I[2, 2])
    assert np.allclose(dyn.inertia_capsule(m, 1e-9, h, axis=2)[0, 0], m * h * h / 12, rtol=1e-6)    # r -> 0: thin rod
    assert np.allclose(dyn.inertia_capsule(m, r, 1e-12, axis=2), dyn.inertia_sphere(m, r), rtol=1e-6)   # h -> 0: sphere


def test_parallel_axis_and_inverse():
    pts = rng.normal(size=(30, 3))
    ms = rng.uniform(0.5, 2, 30)
    com = (ms[:, None] * pts).sum(0) / ms.sum()
    I_c = dyn.inertia_point_masses(ms, pts, about=com)
    d = np.array([0.3, -0.2, 0.5])
    assert np.allclose(dyn.parallel_axis(I_c, ms.sum(), d), dyn.inertia_point_masses(ms, pts, about=com - d))
    assert np.allclose(dyn.parallel_axis_inverse(dyn.parallel_axis(I_c, 3.0, d), 3.0, d), I_c)


def test_rotation_principal_axes_and_validity():
    I = dyn.inertia_box(1.0, 0.5, 0.3, 0.1)
    R = tf.rot_axis_angle([1, 2, 3], 0.7)
    I2 = dyn.rotate_inertia(I, R)
    w, V = dyn.principal_axes(I2)
    assert np.allclose(w, np.sort(np.diag(I)))
    assert np.allclose(V.T @ I2 @ V, np.diag(w), atol=1e-12) and np.isclose(np.linalg.det(V), 1.0)
    assert dyn.is_physical_inertia(I2) and not dyn.is_physical_inertia(np.diag([1.0, 1.0, 3.0]))
    assert np.allclose(dyn.box_from_inertia(1.0, np.diag(I)), [0.5, 0.3, 0.1])


def test_combine_bodies_matches_point_mass_sum():
    pts = rng.normal(size=(20, 3))
    ms = rng.uniform(0.5, 2, 20)
    parts = [(ms[:10].sum(), (ms[:10, None] * pts[:10]).sum(0) / ms[:10].sum(), None),
             (ms[10:].sum(), (ms[10:, None] * pts[10:]).sum(0) / ms[10:].sum(), None)]
    parts[0] = (parts[0][0], parts[0][1], dyn.inertia_point_masses(ms[:10], pts[:10], about=parts[0][1]))
    parts[1] = (parts[1][0], parts[1][1], dyn.inertia_point_masses(ms[10:], pts[10:], about=parts[1][1]))
    m, com, I = dyn.combine_bodies(parts)
    assert np.isclose(m, ms.sum()) and np.allclose(I, dyn.inertia_point_masses(ms, pts, about=com))


# --------------------------------------------------------------------------- one rigid body
def test_free_body_conserves_energy_and_angular_momentum():
    I = np.diag([1.0, 2.0, 3.0])
    w0 = np.array([0.1, 3.0, 0.1])
    t, q, w, R = dyn.free_body_rotation(I, w0, 5.0, 1e-3)
    E = 0.5 * np.einsum("ti,ij,tj->t", w, I, w)
    L_world = np.einsum("tij,tj->ti", R, w @ I)
    assert np.ptp(E) < 1e-9 and np.ptp(L_world, axis=0).max() < 1e-9
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)


def test_newton_euler_body_and_euler_equation():
    I = np.diag([1.0, 2.0, 3.0])
    w, a = np.array([1.0, -2.0, 0.5]), np.array([0.2, 0.1, -0.3])
    F, N = dyn.newton_euler_body(2.0, I, [1, 0, 0], w, a)
    assert np.allclose(F, [2, 0, 0]) and np.allclose(dyn.angular_acceleration(I, w, N), a)


# --------------------------------------------------------------------------- planar arms
def two_r_point_masses(m1=2.0, m2=1.0, L1=1.0, L2=0.8):
    """Craig-style 2R with point masses at the distal ends of the links."""
    return dyn.planar_chain([L1, L2], [m1, m2], com=[L1, L2], Izz=[0.0, 0.0]), (m1, m2, L1, L2)


def closed_form_2r(q, qd, qdd, m1, m2, L1, L2, g):
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    q1, q2 = q
    d1, d2 = qd
    dd1, dd2 = qdd
    t1 = (m2 * L2 ** 2 + 2 * m2 * L1 * L2 * c2 + (m1 + m2) * L1 ** 2) * dd1 + (m2 * L2 ** 2 + m2 * L1 * L2 * c2) * dd2 \
        - m2 * L1 * L2 * s2 * d2 * d1 - m2 * L1 * L2 * s2 * (d1 + d2) * d2 \
        + m2 * L2 * g * np.cos(q1 + q2) + (m1 + m2) * L1 * g * np.cos(q1)
    t2 = (m2 * L1 * L2 * c2 + m2 * L2 ** 2) * dd1 + m2 * L2 ** 2 * dd2 + m2 * L1 * L2 * s2 * d1 ** 2 + m2 * L2 * g * np.cos(q1 + q2)
    return np.array([t1, t2])


def test_rnea_matches_closed_form_2r():
    ch, (m1, m2, L1, L2) = two_r_point_masses()
    for _ in range(5):
        q, qd, qdd = rng.uniform(-1.5, 1.5, (3, 2))
        tau = dyn.rnea(ch, q, qd, qdd, g=[0, -9.81, 0])
        assert np.allclose(tau, closed_form_2r(q, qd, qdd, m1, m2, L1, L2, 9.81))


def test_origin_form_equals_com_form():
    """Craig's centre-of-mass Newton-Euler equals the origin form used internally (1-link check by hand)."""
    ch = dyn.SerialChain([dyn.Link(1.7, [0.2, -0.1, 0.3], dyn.inertia_box(1.7, 0.3, 0.2, 0.1), axis=[0.3, 0.5, 0.8])])
    q, qd, qdd = 0.4, 1.3, -0.7
    tau, d = dyn.rnea(ch, [q], [qd], [qdd], g=[0, 0, -9.81], details=True)
    L = ch.links[0]
    w, wd, vd = d["omega"][0], d["omega_dot"][0], d["v_dot"][0]
    vc = vd + np.cross(wd, L.com) + np.cross(w, np.cross(w, L.com))
    F, N = L.m * vc, L.I_com @ wd + np.cross(w, L.I_com @ w)
    n_origin = N + np.cross(L.com, F)
    assert np.allclose(d["f"][0], F) and np.allclose(d["n"][0], n_origin)
    assert np.isclose(tau[0], n_origin @ L.axis)


def random_chain(n=4, prismatic=False, seed=3):
    r = np.random.default_rng(seed)
    links = []
    for i in range(n):
        A = r.normal(size=(3, 3))
        I = A @ A.T * 0.05 + np.eye(3) * 0.05
        Rfix = tf.rot_axis_angle(r.normal(size=3), r.uniform(0, 3))
        kind = "P" if (prismatic and i % 2 == 1) else "R"
        links.append(dyn.Link(r.uniform(0.5, 3), r.normal(size=3) * 0.2, I, Rfix, r.normal(size=3) * 0.3, r.normal(size=3), kind, f"l{i}"))
    return dyn.SerialChain(links)


@pytest.mark.parametrize("prismatic", [False, True])
def test_newton_euler_and_lagrange_agree(prismatic):
    ch = random_chain(5, prismatic)
    for _ in range(3):
        q = rng.uniform(-1, 1, ch.n)
        M_ne, M_lag = dyn.mass_matrix(ch, q), dyn.mass_matrix_lagrange(ch, q)
        assert np.allclose(M_ne, M_lag, atol=1e-12)
        assert np.allclose(M_ne, M_ne.T) and np.linalg.eigvalsh(M_ne).min() > 0


@pytest.mark.parametrize("prismatic", [False, True])
def test_christoffel_coriolis_properties(prismatic):
    ch = random_chain(4, prismatic)
    q, qd = rng.uniform(-1, 1, (2, ch.n))
    C = dyn.coriolis_matrix(ch, q, qd)
    zero_g = np.zeros(3)
    assert np.allclose(C @ qd, dyn.rnea(ch, q, qd, np.zeros(ch.n), zero_g), atol=1e-11)          # C qd = velocity-product torque
    dM = dyn.mass_matrix_derivative(ch, q)
    Mdot = np.einsum("kij,k->ij", dM, qd)
    S = Mdot - 2 * C
    assert np.allclose(S, -S.T, atol=1e-11) and np.allclose(Mdot, C + C.T, atol=1e-11)
    h = 1e-6                                                                                       # complex step vs finite differences
    e = np.zeros(ch.n)
    e[1] = h
    assert np.allclose(dM[1], (dyn.mass_matrix(ch, q + e) - dyn.mass_matrix(ch, q - e)) / (2 * h), atol=1e-7)


def test_energy_balance_and_gravity():
    ch = random_chain(3)
    q, qd = rng.uniform(-1, 1, (2, 3))
    g = np.array([0.0, 0.0, -9.81])
    # d/dq of potential energy is the gravity torque
    h = 1e-6
    grad = np.array([(dyn.potential_energy(ch, q + h * e, g) - dyn.potential_energy(ch, q - h * e, g)) / (2 * h) for e in np.eye(3)])
    assert np.allclose(grad, dyn.gravity_torque(ch, q, g), atol=1e-7)
    # power balance qd . tau = d/dt (K + V) for an arbitrary acceleration
    qdd = rng.uniform(-1, 1, 3)
    tau = dyn.rnea(ch, q, qd, qdd, g)
    eps = 1e-6
    E = lambda s: dyn.kinetic_energy(ch, q + s * qd + 0.5 * s * s * qdd, qd + s * qdd) + dyn.potential_energy(ch, q + s * qd + 0.5 * s * s * qdd, g)
    assert np.isclose(qd @ tau, (E(eps) - E(-eps)) / (2 * eps), rtol=1e-6, atol=1e-6)


def test_regressor_is_exact_and_rank_of_vertical_2r():
    ch = random_chain(3, prismatic=True)
    q, qd, qdd = rng.uniform(-1, 1, (3, 3))
    assert np.allclose(dyn.regressor(ch, q, qd, qdd) @ dyn.chain_pi(ch), dyn.rnea(ch, q, qd, qdd))
    ch2 = dyn.planar_chain([1.0, 0.8], [2.0, 1.0])
    Y = np.vstack([dyn.regressor(ch2, *rng.uniform(-2, 2, (3, 2)), g=[0, -9.81, 0]) for _ in range(40)])
    assert dyn.numerical_rank(Y) == 6                       # I11, (h1x + m2 L1), h1y, I22, h2x, h2y
    assert dyn.numerical_rank(Y[:, [i for i in range(20) if i % 10 != 2]]) == 4      # centres of mass on the link axes: h_y = 0
    ch3 = dyn.chain_with_pi(ch2, dyn.chain_pi(ch2))
    assert np.allclose(dyn.mass_matrix(ch3, [0.3, 0.9]), dyn.mass_matrix(ch2, [0.3, 0.9]))


def test_dh_chain_matches_planar_chain():
    L = [1.0, 0.8]
    ch = dyn.planar_chain(L, [2.0, 1.0])
    table = [(0, 0, 0, 0, "R"), (0, L[0], 0, 0, "R")]
    dh = dyn.dh_chain(table, [2.0, 1.0], [[0.5, 0, 0], [0.4, 0, 0]], [l.I_com for l in ch.links])
    q = [0.4, -1.1]
    assert np.allclose(dyn.mass_matrix(dh, q), dyn.mass_matrix(ch, q))
    assert np.allclose(dyn.bias_torque(dh, q, [1, 2], [0, -9.8, 0]), dyn.bias_torque(ch, q, [1, 2], [0, -9.8, 0]))


# --------------------------------------------------------------------------- simulation
def test_integrator_orders_and_energy():
    ch = dyn.planar_chain([1.0, 0.8], [2.0, 1.0])
    g = [0, -9.81, 0]
    q0, qd0 = np.array([1.0, -0.5]), np.zeros(2)
    ref = dyn.simulate(ch, q0, qd0, 1.0, 2.5e-3, g=g, method="rk4", joint_effects=False)[1][-1]
    errs = {}
    for method in ("euler", "midpoint", "rk4"):
        errs[method] = [np.abs(dyn.simulate(ch, q0, qd0, 1.0, dt, g=g, method=method, joint_effects=False)[1][-1] - ref).max() for dt in (0.02, 0.01)]
    assert errs["euler"][0] / errs["euler"][1] == pytest.approx(2, rel=0.3)
    assert errs["midpoint"][0] / errs["midpoint"][1] > 3
    assert errs["rk4"][1] < 1e-5
    t, q, qd = dyn.simulate(ch, q0, qd0, 5.0, 1e-3, g=g, joint_effects=False)
    E = [dyn.kinetic_energy(ch, a, b) + dyn.potential_energy(ch, a, g) for a, b in zip(q[::500], qd[::500])]
    assert np.ptp(E) < 1e-7


def test_friction_dissipates_energy():
    ch = dyn.planar_chain([1.0], [1.0])
    ch.damping[:] = 0.2
    g = [0, -9.81, 0]
    t, q, qd = dyn.simulate(ch, [1.0], [0.0], 3.0, 1e-3, g=g)
    E = np.array([dyn.kinetic_energy(ch, a, b) + dyn.potential_energy(ch, a, g) for a, b in zip(q[::100], qd[::100])])
    assert np.all(np.diff(E) < 1e-9)


# --------------------------------------------------------------------------- symbolic
def test_symbolic_lagrangian_equals_newton_euler():
    import sympy as sp

    q = sp.symbols("q1:4")
    qd = sp.symbols("qd1:4")
    m, L, g = sp.symbols("m L g", positive=True)
    zero = sp.zeros(3, 1)

    def rod(Rfix, p, axis):
        return dict(R=Rfix, p=p, axis=axis, m=m, com=sp.Matrix([0, 0, L / 2]), I=sp.diag(m * L ** 2 / 12, m * L ** 2 / 12, 0))

    ey = sp.Matrix([0, 1, 0])
    links = [rod(sp.eye(3), zero, [0, 0, 1]), rod(sp.eye(3), sp.Matrix([0, 0, L]), [0, 1, 0]), rod(sp.eye(3), sp.Matrix([0, 0, L]), [0, 1, 0])]
    K, V = dyn.sym_energies(links, q, qd, [0, 0, -g])
    M, C, gv = dyn.euler_lagrange(K, V, q, qd, simplify=sp.trigsimp)
    nl = [dyn.Link(1.5, [0, 0, 0.4], np.diag([1.5 * 0.64 / 12, 1.5 * 0.64 / 12, 0]), p=[0, 0, 0 if i == 0 else 0.8], axis=[0, 0, 1] if i == 0 else [0, 1, 0]) for i in range(3)]
    ch = dyn.SerialChain(nl)
    sub = {m: 1.5, L: 0.8, g: 9.81}
    qv, qdv = rng.uniform(-1, 1, (2, 3))
    sub_all = {**sub, **dict(zip(q, qv)), **dict(zip(qd, qdv))}
    assert np.allclose(np.array(M.subs(sub_all), float), dyn.mass_matrix(ch, qv), atol=1e-10)
    assert np.allclose(np.array(gv.subs(sub_all), float).ravel(), dyn.gravity_torque(ch, qv), atol=1e-10)
    assert np.allclose(np.array((C * sp.Matrix(qd)).subs(sub_all), float).ravel(), dyn.coriolis_matrix(ch, qv, qdv) @ qdv, atol=1e-10)


# --------------------------------------------------------------------------- MuJoCo cross-checks
def _robot(name):
    mujoco = pytest.importorskip("mujoco")
    from rgkit import robots
    try:
        model, data = robots.load_robot(name, scene="bare")
    except Exception as exc:                                       # no network and no cache
        pytest.skip(f"robot model unavailable: {exc}")
    return mujoco, model


@pytest.mark.parametrize("name,tip", [("2r", "second_segment"), ("ur5e", "wrist_3_link"), ("panda", "hand")])
def test_matches_mujoco_mass_bias_and_inverse_dynamics(name, tip):
    mujoco, model0 = _robot(name)
    model = dyn.mujoco_rigid_copy(model0, passive=True)
    ch = dyn.chain_from_mujoco(model, tip)
    ch.frictionloss[:] = 0.0                      # all constraints (including frictionloss) are disabled in this copy
    n = ch.n
    data = mujoco.MjData(model)
    r = np.random.default_rng(11)
    lo, hi = model.jnt_range[:n].T
    for _ in range(4):
        q, qd, qdd = r.uniform(lo, hi), r.uniform(-1, 1, n), r.uniform(-1, 1, n)
        data.qpos[:] = model.qpos0
        data.qpos[:n], data.qvel[:] = q, 0.0
        data.qvel[:n] = qd
        mujoco.mj_forward(model, data)
        M = dyn.mujoco_mass_matrix(model, data)
        assert np.allclose(M[:n, :n], dyn.mass_matrix(ch, q, joint_effects=True), atol=1e-9)
        assert np.allclose(data.qfrc_bias[:n], dyn.bias_torque(ch, q, qd, model.opt.gravity), atol=1e-9)
        data.qacc[:] = 0.0
        data.qacc[:n] = qdd
        mujoco.mj_inverse(model, data)
        assert np.allclose(data.qfrc_inverse[:n], dyn.inverse_dynamics(ch, q, qd, qdd, model.opt.gravity), atol=1e-9)


def test_simulation_matches_mujoco_step():
    mujoco, model0 = _robot("2r")
    model = dyn.mujoco_rigid_copy(model0, passive=True)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model.opt.timestep = 1e-3
    ch = dyn.chain_from_mujoco(model, "second_segment")
    ch.frictionloss[:] = 0.0
    data = mujoco.MjData(model)
    q0 = np.array([0.8, -0.4])
    data.qpos[:] = q0
    for _ in range(1000):
        mujoco.mj_step(model, data)
    t, q, qd = dyn.simulate(ch, q0, np.zeros(2), 1.0, 1e-3, g=model.opt.gravity)
    assert np.allclose(q[-1], data.qpos, atol=1e-8) and np.allclose(qd[-1], data.qvel, atol=1e-8)
