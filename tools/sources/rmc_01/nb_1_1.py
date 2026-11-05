from _common import *

K = 1
FN = NB[K][1]
cells = [
header(COURSE, CHAPTER, "1.1", "Modelling a Robot Joint: Transfer Functions, Block Diagrams and State Space",
       f"{CH_DIR}/{FN}", [
    ("Primary reading", "Ogata, *Modern Control Engineering*, Chs. 2–4 (mathematical modelling; chapter numbers of the 5th edition) · Nise, *Control Systems Engineering* (the chapters on modelling in the frequency and time domains) · Craig, *Introduction to Robotics*, Ch. 9 (linear control of manipulators)"),
    ("Parallel reading", "Spong, Hutchinson & Vidyasagar, Ch. 8 (independent-joint control: actuator and gear model) · Siciliano et al., Ch. 8 (motion control) · Lynch & Park, Ch. 11 (robot control)"),
    ("Lectures", f"{NANDI2} · {BRUNTON}"),
    ("Prerequisites", f"{M5} · {M6}"),
    ("Study time", "about 3 hours"),
], """\
Every joint of a robot arm is, to a good approximation, a motor driving an inertia through a gearbox, and that simple
fact is why the classical language of control — transfer functions, poles, feedback loops — remains the working language
of motion control. This notebook builds the model that the rest of the course will control: a **DC-motor-driven joint**
with a gear ratio $N$, reflected inertia and friction, and a gravity-loaded link. We derive its transfer function and its
state-space form, learn to reduce block diagrams, linearise the one nonlinear term (gravity), and check every step against
MuJoCo's `dcmotor` actuator and against the real Dynamixel 2R and UR5e models."""),

md(r"""
## Learning objectives

1. Write the equations of a robot joint (motor, gearbox, link) and explain how a gear ratio reflects inertia and friction to the joint side.
2. Derive the transfer functions from armature voltage to joint speed and angle, and interpret the electrical and mechanical time constants.
3. Reduce block diagrams (series, parallel, feedback, a disturbance entering the loop) to one transfer function and verify the result numerically.
4. Convert between a differential equation, a transfer function and a state-space model, and read the poles as eigenvalues of $A$.
5. Linearise the nonlinear (gravity-loaded) joint about an equilibrium, and state how far the linear model can be trusted.
6. Read inertia, servo gains and gravity stiffness out of a MuJoCo robot model and confirm hand-derived predictions against the simulator.

## Contents

1. [A joint as a control problem](#1.-A-joint-as-a-control-problem)
2. [Transfer functions in code](#2.-Transfer-functions-in-code)
3. [Block diagrams and their algebra](#3.-Block-diagrams-and-their-algebra)
4. [The DC motor, the gearbox and the link](#4.-The-DC-motor,-the-gearbox-and-the-link)
5. [Cross-check: MuJoCo's DC-motor actuator](#5.-Cross-check:-MuJoCo's-DC-motor-actuator)
6. [State-space models](#6.-State-space-models)
7. [Linearisation](#7.-Linearisation)
8. [Real robots: reading a joint out of a MuJoCo model](#8.-Real-robots:-reading-a-joint-out-of-a-MuJoCo-model)
9. [Worked examples](#9.-Worked-examples)
10. [Summary, common mistakes and exercises](#10.-Summary)
"""),

code(SETUP + r'''
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.special import ellipk
from rgkit import linear_control as lc, control_plots as cp, joint_sim as js, mujoco_utils as mju, robots
from rgkit.linear_control import TF
import mujoco

p = js.dc_joint_params()          # the reference joint used throughout this chapter (see Section 1)
'''),

md(r"""
## 1. A joint as a control problem

A robot arm moves because each joint is driven by an actuator. In the great majority of arms the actuator is an electric
motor, the motor is followed by a reduction gearbox, and a sensor on the motor or the joint shaft reports the angle. The
**control problem** of a single joint is to make the measured angle $\theta$ follow a desired angle $\theta_d$ supplied by a
trajectory generator, despite friction, gravity and the motion of the other joints. The standard decomposition of the
loop is shown below: the *controller* turns the error into a command (a voltage or a current), the *amplifier and motor*
convert it into a motor torque, the *gearbox* multiplies the torque by the gear ratio $N$, and the *link* (with whatever it
carries) responds by accelerating. Everything between the command and the angle — amplifier, motor, gearbox, link — is called
the **plant**; the controller is what we design.

Two features make this problem friendly. First, with a high gear ratio the motor's own inertia, reflected through the gearbox,
usually dominates the link's inertia, so the joint behaves like a **constant-inertia** system even though the arm's
configuration changes (Section 4 quantifies this). Second, apart from gravity the dynamics are **linear**, so the entire toolbox
of linear control — poles, Bode plots, root locus, state feedback — applies. This is the viewpoint of *independent-joint
control* in Craig Ch. 9 and Spong Ch. 8; Chapters 3–4 of this course show what it costs and how to do better.
"""),

code(r'''
fig, ax = plt.subplots(figsize=(11.5, 1.9))
cp.draw_chain(ax, ["Controller", "Amplifier\nand motor", "Gearbox  $N{:}1$", "Link and\npayload"],
              [r"$\theta_d - \theta$", r"$V$", r"$\tau_m$", r"$\tau$", r"$\theta$"])
ax.set_title("One joint servo: the plant is everything to the right of the controller (the encoder feeds $\\theta$ back)", fontsize=10.5)
plt.show()
'''),

md(r"""
**The reference joint.** Throughout Chapter 1 we use one concrete joint, a model of a small collaborative-arm joint: a brushed DC
motor behind a $50{:}1$ gearbox driving a 0.4 m, 0.8 kg link. The values are *teaching values of realistic magnitude* (a motor
of roughly 100 W), not a manufacturer's datasheet. MuJoCo can simulate exactly this joint with its `dcmotor` actuator, which
lets us test every formula against a simulator written by someone else.
"""),

code(r'''
rows = [("gear ratio", "N", f"{p['N']:.0f}", "–"),
        ("torque constant", "K_t", f"{p['Kt']:.3f}", "N m/A"),
        ("back-EMF constant", "K_e", f"{p['Ke']:.3f}", "V s/rad"),
        ("armature resistance", "R", f"{p['R']:.2f}", "Ω"),
        ("armature inductance", "L", f"{p['L'] * 1e3:.2f}", "mH"),
        ("rotor inertia (motor side)", "J_m", f"{p['Jm']:.1e}", "kg m²"),
        ("viscous friction (joint side)", "b", f"{p['b']:.3f}", "N m s/rad"),
        ("link mass / length", "m, ℓ", f"{p['m']} / {p['ell']}", "kg / m"),
        ("supply voltage limit", "V_max", f"{p['vmax']:.0f}", "V"),
        ("**reflected rotor inertia**", "N² J_m", f"{p['J_rotor']:.4f}", "kg m²"),
        ("**link inertia about the pivot**", "J_L = mℓ²/3", f"{p['J_link']:.4f}", "kg m²"),
        ("**total inertia at the joint**", "J", f"{p['J']:.4f}", "kg m²"),
        ("**peak gravity torque**", "m g ℓ/2", f"{p['mgd']:.3f}", "N m")]
display(Markdown("| quantity | symbol | value | unit |\n|:--|:--|--:|:--|\n" + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)))
'''),

md(r"""
## 2. Transfer functions in code

The Laplace transform turns a linear differential equation with constant coefficients into algebra; its definition,
table and the idea of a transfer function are developed in the primer chapter [M6](../../math-primer/M6-laplace-and-linear-systems.md)
and are not repeated here. We need only the working definition: for zero initial conditions the **transfer function**
$G(s)=Y(s)/U(s)$ of a linear time-invariant system is the ratio of the Laplace transforms of output and input, and for a
differential equation it can be written down by replacing $d/dt$ with $s$. Its **poles** (roots of the denominator) set the
natural modes $e^{pt}$ of the response and its **zeros** (roots of the numerator) shape how those modes are combined.

The chapter library `rgkit.linear_control` provides a small class `TF` that stores numerator and denominator coefficients
(highest power first) and supports the algebra we need: `*` (series), `+` (parallel), `.feedback()`, `.poles()`, `.zeros()`,
`.dcgain()` and `.step()`. The package `control` is *not* required; everything is plain NumPy and SciPy.

As a first example take the **bare joint**: an inertia $J$ with viscous friction $b$ driven by a torque $\tau$,

$$
J\,\ddot\theta + b\,\dot\theta = \tau
\qquad\Longrightarrow\qquad
G(s) = \frac{\Theta(s)}{T(s)} = \frac{1}{s\,(Js+b)}.
$$

It has a pole at the origin (the joint does not return to any angle when released: it is a pure integrator in position) and a pole at
$-b/J$. For a constant torque $\tau_0$ the speed approaches $\tau_0/b$ with time constant $J/b$, and the angle is
$\theta(t) = \dfrac{\tau_0}{b}\left[t - \dfrac{J}{b}\left(1-e^{-bt/J}\right)\right]$.
"""),

code(r'''
J, b = p["J"], p["b"]
G_bare = TF([1.0], [J, b, 0.0])
print(G_bare, "\npoles:", G_bare.poles(), "  -b/J =", -b / J)
t = np.linspace(0, 20, 2001)
tau0 = 0.5
theta_tf = tau0 * G_bare.step(t)
theta_exact = tau0 / b * (t - J / b * (1 - np.exp(-b * t / J)))
print("TF step response equals the closed form:", np.allclose(theta_tf, theta_exact, atol=1e-6))
print(f"final speed tau0/b = {tau0 / b:.1f} rad/s; time constant J/b = {J / b:.2f} s")
'''),

md(r"""
## 3. Block diagrams and their algebra

A **block diagram** shows how signals flow through transfer functions. Its value is that the *topology* of a loop, not
the individual formulae, determines the closed-loop behaviour. Four rules reduce almost every diagram:

| Configuration | Equivalent block |
|:--|:--|
| series (cascade) | $G_1(s)\,G_2(s)$ |
| parallel (summing junction) | $G_1(s) \pm G_2(s)$ |
| negative feedback loop, forward $G$, feedback $H$ | $\dfrac{G}{1+GH}$ (positive feedback: $\dfrac{G}{1-GH}$) |
| moving a pick-off point or a summing junction across a block | multiply or divide the moved branch by that block |

For a system with more than one input — a reference $R$ and a disturbance $D$ — **superposition** applies: set all
inputs but one to zero, reduce, and add the results. The diagram below is the standard feedback loop of this course: a
controller $C(s)$, a plant $G(s)$, a sensor $H(s)$ and a disturbance torque $D(s)$ (gravity, friction, the reaction of another joint)
entering at the plant input. Let us have a computer algebra system solve the loop equations rather than doing it by hand.
"""),

code(r'''
fig, ax = plt.subplots(figsize=(8.5, 3.0))
cp.draw_loop(ax, forward=("C(s)", "G(s)"), feedback="H(s)", r="$R(s)$", y="$Y(s)$", d="$D(s)$", d_at=1)
ax.set_title("The standard feedback loop with a plant-input disturbance", fontsize=10.5)
plt.show()

import sympy as sp
C_, G_, H_, R_, D_, Y_, E_ = sp.symbols("C G H R D Y E")
sol = sp.solve([sp.Eq(E_, R_ - H_ * Y_), sp.Eq(Y_, G_ * (C_ * E_ + D_))], [E_, Y_], dict=True)[0]
Y_expr = sp.simplify(sol[Y_])
display(Math(r"Y = " + sp.latex(sp.collect(sp.expand(sp.simplify(Y_expr * (1 + C_ * G_ * H_))), [R_, D_])) + r"\,/\,(1+CGH)"))
'''),

md(r"""
The result is

$$
Y(s) \;=\; \underbrace{\frac{C\,G}{1+C\,G\,H}}_{T_{r}(s)}\,R(s) \;+\; \underbrace{\frac{G}{1+C\,G\,H}}_{T_{d}(s)}\,D(s),
$$

where $L=CGH$ is the **loop transfer function** (the product of everything met once round the loop) and $1+L$ is the
quantity whose zeros are the closed-loop poles. Three lessons are already visible. The reference and the disturbance see the *same
denominator*, hence the same poles. The disturbance transfer function $G/(1+CGH)$ is *small where $|CGH|$ is large*, so high loop gain suppresses
disturbances — the reason feedback is used at all. And the numerators differ, so a design that tracks well need not
reject disturbances well. The next cell confirms the formulae numerically with the `TF` class at one frequency.
"""),

code(r'''
C = TF([6.0, 2.0], [1.0, 0.0]) * 0 + TF([30.0, 10.0], [1.0, 0.0])      # a PI-type controller (30 s + 10)/s
Gp = TF([1.5], np.polymul([1, 0], [0.1177, 0.02]) )                        # a plant
H = TF([1.0], [0.002, 1.0])                                               # a first-order sensor filter
Tr = (C * Gp).feedback(H)                       # R -> Y
Td = Gp.feedback(C * H)                         # D -> Y   (plant with the controller-sensor path as feedback)
s0 = 3.0 + 4.0j
L0 = C(s0) * Gp(s0) * H(s0)
print("T_r matches  C G/(1+CGH):", np.isclose(Tr(s0), C(s0) * Gp(s0) / (1 + L0)))
print("T_d matches  G/(1+CGH)  :", np.isclose(Td(s0), Gp(s0) / (1 + L0)))
print("same closed-loop poles  :", np.allclose(np.sort_complex(Tr.poles()), np.sort_complex(Td.poles())))
'''),

md(r"""
## 4. The DC motor, the gearbox and the link

**The motor.** A permanent-magnet DC motor with armature resistance $R$, inductance $L$, torque constant $K_t$ and back-EMF
constant $K_e$ obeys two coupled laws. *Electrically*, Kirchhoff's voltage law gives
$V = R\,i + L\,\dfrac{di}{dt} + e_b$, where the **back-EMF** $e_b = K_e\,\omega_m$ is proportional to the motor shaft speed
$\omega_m$. *Mechanically*, the torque is proportional to the current, $\tau_m = K_t\, i$. In SI units
$K_t = K_e$: the electrical power absorbed by the back-EMF, $e_b i = K_e\omega_m i$, must equal the mechanical power delivered,
$\tau_m\omega_m = K_t i\,\omega_m$.

**The gearbox.** An ideal gearbox with ratio $N{:}1$ turns the motor $N$ times for each turn of the joint, so
$\theta_m = N\theta$, and (power conserved) the joint torque is $\tau_j = N\tau_m$ minus losses. Writing the motor's
Newton equation $J_m\dot\omega_m = \tau_m - \tau_j/N - b_m\omega_m$ and eliminating the motor variables in favour of joint
variables gives the **joint-side** equation

$$
J\,\ddot\theta + b\,\dot\theta + \tau_g(\theta) = N K_t\, i,
\qquad J = N^2 J_m + J_L,\qquad b = N^2 b_m + b_L,
$$

with $J_L$ the link inertia about the joint axis, $\tau_g$ the gravity torque, and $J_m$, $b_m$ the rotor's inertia and
friction. This is the central result of the section: **inertia and friction on the motor side are multiplied by $N^2$ when seen from
the joint.** (In MuJoCo the reflected rotor inertia $N^2J_m$ is entered as the joint's `armature`.)

**The transfer function.** Take the Laplace transform of the electrical equation, $V = (Ls+R)I + K_e N\,s\Theta$ (joint
speed $s\Theta$, motor speed $N s\Theta$), and of the mechanical one with $\tau_g=0$, $(Js^2+bs)\Theta = N K_t I$. Eliminating $I$:

$$
\frac{\Omega(s)}{V(s)} = \frac{N K_t}{(Ls+R)(Js+b) + N^2K_tK_e},
\qquad
\frac{\Theta(s)}{V(s)} = \frac{1}{s}\,\frac{\Omega(s)}{V(s)}.
$$

The denominator is a quadratic whose two roots are real for realistic motors. They are separated by a factor of the order of
100 or more: one **fast electrical pole** near $-R/L$ and one **slow mechanical pole**. When the inductance is negligible
($L\to0$) the current follows the voltage instantly, $i=(V-NK_e\omega)/R$, and substitution into the mechanical equation gives a first-order
model,

$$
J\dot\omega + \Big(b + \frac{N^2K_tK_e}{R}\Big)\omega = \frac{NK_t}{R}\,V
\quad\Longrightarrow\quad
\frac{\Omega}{V} \approx \frac{K_\omega}{\tau_m s + 1},\quad
K_\omega = \frac{N K_t}{Rb+N^2K_tK_e},\quad
\tau_m = \frac{R J}{Rb+N^2K_tK_e}.
$$

The term $N^2K_tK_e/R$ is an *electrical damping*: the motor, acting as a generator into its own resistance, brakes itself in
proportion to speed. For this reason a motor with its terminals shorted resists being turned, and a servo with a large
$K_tK_e/R$ is heavily damped without any added friction.
"""),

code(r'''
N, Kt, Ke, R, L = p["N"], p["Kt"], p["Ke"], p["R"], p["L"]
G_w = js.dc_joint_tf(p, "speed")            # voltage -> joint speed
G_th = js.dc_joint_tf(p, "angle")           # voltage -> joint angle
K_w = N * Kt / (R * b + N**2 * Kt * Ke)
tau_m = R * J / (R * b + N**2 * Kt * Ke)
print("voltage->speed :", G_w)
print("poles of the speed model:", G_w.poles().round(2), "  (electrical ~ -R/L = %.0f, mechanical ~ -1/tau_m = %.2f)" % (-R / L, -1 / tau_m))
print(f"K_w = {K_w:.4f} (rad/s)/V,  tau_m = {tau_m * 1e3:.1f} ms,  tau_e = L/R = {L / R * 1e3:.1f} ms")
print(f"no-load speed at {p['vmax']:.0f} V: {p['vmax'] * K_w:.2f} rad/s;  stall current {p['vmax'] / R:.1f} A, stall torque {N * Kt * p['vmax'] / R:.1f} N m")

t = np.linspace(0, 0.4, 4001)
y_full = 5.0 * G_w.step(t)
y_red = 5.0 * lc.first_order(K_w, tau_m).step(t)
err = np.abs(y_full - y_red).max() / (5.0 * K_w)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax[0].plot(t, y_full, color=cp.BLUE, lw=2, label="full model (3rd-order $\\Theta/V$ differentiated)")
ax[0].plot(t, y_red, "--", color=cp.ORANGE, lw=2, label="reduced model $K_\\omega/(\\tau_m s+1)$")
ax[0].set(xlabel="time (s)", ylabel="joint speed (rad/s)", title="5 V step: speed")
ax[0].legend(fontsize=8, loc="lower right")
tt = np.linspace(0, 0.03, 600)
i_full = 5.0 * (G_w * TF([1.0, 0.0]) * 0 + TF([J, b], [L * J, L * b + R * J, R * b + N**2 * Kt * Ke])).step(tt)
ax[1].plot(tt * 1e3, i_full, color=cp.BLUE, lw=2)
ax[1].set(xlabel="time (ms)", ylabel="armature current (A)", title="current: the 5 V step demands $V/R$ at once")
plt.tight_layout(); plt.show()
print(f"largest difference between the two speed models: {100 * err:.2f} % of the final speed")
assert err < 0.02
'''),

md(r"""
The first panel shows why the reduced model is so useful: after the first millisecond the two curves are indistinguishable (the
largest difference is about 1.5 % of the final speed, and it fades with the mechanical time constant). The second panel shows the
current drawn when a voltage step is applied to a motor at rest: it jumps to $V/R$ — here $5/1.2 \approx 4.2$ A, and **20 A** at the full 24 V —
and then falls as the back-EMF builds up. Real amplifiers limit this current; Chapter 2 treats torque and current limits.

**Why geared joints decouple.** The inertia that a joint presents depends on how far the arm is extended and on the payload,
because $J_L$ varies. But the joint inertia is $J = N^2J_m + J_L$: the larger $N$, the smaller the *relative* variation of $J$.
Craig (Ch. 9) and Spong (Ch. 8) use this to justify controlling each joint as an independent constant-inertia servo and treating
the rest as a disturbance. The figure quantifies it for our joint when the link inertia triples (the arm reaches out, or picks up
a load).
"""),

code(r'''
Ns = np.array([1, 5, 10, 20, 50, 100, 200])
ratio = (Ns**2 * p["Jm"] + 3 * p["J_link"]) / (Ns**2 * p["Jm"] + p["J_link"])
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.semilogx(Ns, ratio, "o-", color=cp.BLUE, lw=2)
ax.axhline(3, color=cp.GREY, ls="--"); ax.text(1.1, 3.03, "direct drive: $J$ triples", fontsize=9, color=cp.GREY)
ax.axhline(1, color=cp.GREY, ls="--")
ax.set(xlabel="gear ratio $N$", ylabel="$J(3J_L)\\,/\\,J(J_L)$", ylim=(0.9, 3.2),
       title="Variation of the joint inertia when the link inertia triples")
plt.tight_layout(); plt.show()
for n_, r_ in zip(Ns, ratio):
    print(f"N = {n_:4d}:  J varies by a factor {r_:.2f}")
'''),

md(r"""
**Inertia matching.** Given a motor of rotor inertia $J_m$ and a load $J_L$, which gear ratio gives the load the greatest
acceleration for a given motor torque $\tau_m$? The load acceleration is $\alpha = N\tau_m/(N^2J_m+J_L)$. Setting
$d\alpha/dN=0$ gives $N^2J_m = J_L$, i.e. **the reflected rotor inertia equals the load inertia** ($N^* = \sqrt{J_L/J_m}$).
The maximum is flat — a ratio 30 % away from $N^*$ costs less than 2 % in acceleration — so one chooses $N$ for other reasons
(speed range, torque, cost, backlash, stiffness) and rarely exactly at $N^*$. High-ratio robot gearboxes therefore usually
operate with $N^2J_m \gg J_L$, which is what makes the independent-joint model work.
"""),

md(r"""
## 5. Cross-check: MuJoCo's DC-motor actuator

We now build the reference joint in MuJoCo and compare it with the transfer function. MuJoCo (version 3.5 and later) provides a
`dcmotor` actuator with exactly the electrical dynamics of Section 4: its control input is the **terminal voltage**, it carries the
armature current as an internal *actuator state*, and with `gear` $=N$ it applies the joint torque $N K_t i$ and the back-EMF
$N K_e\dot\theta$. The reflected rotor inertia becomes the joint's `armature`, and viscous friction its `damping`. The helper
`joint_sim.dc_joint_xml` writes the model; read it once to see how each symbol of the table in Section 1 appears.
"""),

code(r'''
xml = js.dc_joint_xml(p, gravity=False)          # gravity off: the model is then exactly linear
model, data = mju.load(xml)
print(xml[xml.index("<actuator>"):xml.index("</actuator>") + 11])
print(xml[xml.index("<joint"):xml.index("<geom type=\"capsule\"")].strip())
'''),

code(r'''
M = js.mass_matrix(model, data)
print(f"MuJoCo effective inertia M[0,0] = {M[0, 0]:.5f} kg m^2   hand calculation N^2 Jm + m l^2/3 = {p['J']:.5f}")
assert np.isclose(M[0, 0], p["J"])

V = 5.0
out = js.simulate(model, data, lambda t, y: V, T=0.4, Ts=2e-4)           # a 5 V step
t = out["t"]
# predictions from the transfer function and from the state-space model (state: angle, speed, current)
A, B, C, D = lc.dc_motor_ss(Kt, Ke, R, L, J, b, N)
_, y_ss, x_ss = signal.lsim((A, B, np.eye(3), np.zeros((3, 1))), V * np.ones_like(t), t)
th_tf, om_tf = V * G_th.step(t), V * G_w.step(t)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, sim, pred, lab in zip(ax, (out["q"], out["qd"], out["i"]), (th_tf, om_tf, x_ss[:, 2]),
                             ("joint angle (rad)", "joint speed (rad/s)", "armature current (A)")):
    a.plot(t, sim, color=cp.BLUE, lw=3, alpha=0.6, label="MuJoCo `dcmotor`")
    a.plot(t, pred, "--", color=cp.ORANGE, lw=1.8, label="model")
    a.set(xlabel="time (s)", ylabel=lab)
ax[0].legend(fontsize=8)
plt.suptitle("Reference joint, 5 V step, gravity off: MuJoCo against the hand-derived model", y=1.02, fontsize=11)
plt.tight_layout(); plt.show()

e_th = np.abs(out["q"] - th_tf).max() / th_tf[-1]
e_om = np.abs(out["qd"] - om_tf).max() / om_tf[-1]
e_ss = np.abs(x_ss[:, 0] - th_tf).max() / th_tf[-1]
print(f"max error, angle: {100 * e_th:.3f} % of final;   speed: {100 * e_om:.3f} %;   (TF vs state space: {100 * e_ss:.1e} %)")
assert e_th < 3e-3 and e_om < 3e-3 and e_ss < 1e-6
'''),

md(r"""
The simulator and the model agree to a fraction of a percent in all three signals. The tiny residual comes from MuJoCo's
fixed time step (0.2 ms here) and its implicit integrator; making the step smaller shrinks it. We have thus verified, independently of
our algebra, the claim that the plant from voltage to angle is the third-order transfer function of Section 4. From now on the
joint model can be trusted as a design model, and the same MuJoCo model will act as the "real" plant for every controller in
this chapter.

A rendering of the model follows. Positive $\theta$ swings the rod towards $-x$; the rod hangs straight down at $\theta=0$, and
$\theta=\pi$ is the inverted position that Notebook 1.6 will stabilise.
"""),

code(r'''
model_g, data_g = mju.load(js.dc_joint_xml(p, gravity=True))
frames = []
for ang in (0.0, np.radians(60), np.radians(120)):
    data_g.qpos[0] = ang
    mujoco.mj_forward(model_g, data_g)
    frames.append(mju.render(model_g, data_g, camera=dict(azimuth=90, elevation=0, distance=1.9, lookat=(0, 0, -0.25)), width=360, height=300))
filmstrip(frames, [r"$\theta=0$ (hanging)", r"$\theta=60^\circ$", r"$\theta=120^\circ$"], width=10)
'''),

md(r"""
## 6. State-space models

A transfer function describes only the input–output relation. A **state-space model** describes the *internal* evolution as a
set of first-order equations,

$$
\dot{\mathbf x} = A\mathbf x + B u,\qquad y = C\mathbf x + D u,
$$

where the **state** $\mathbf x$ collects the quantities that, together with the future input, determine the future
(here: angle, speed, current). It is the natural form for simulation, for multi-input and multi-output systems, and for the
design methods of Notebook 1.6. For the reference joint, choose $\mathbf x=[\theta,\ \omega,\ i]^{\mathsf T}$ and the voltage
$V$ as input:

$$
\begin{bmatrix}\dot\theta\\ \dot\omega\\ \dot i\end{bmatrix} =
\begin{bmatrix}0&1&0\\ 0 & -b/J & NK_t/J\\ 0 & -NK_e/L & -R/L\end{bmatrix}
\begin{bmatrix}\theta\\ \omega\\ i\end{bmatrix} +
\begin{bmatrix}0\\0\\ 1/L\end{bmatrix}V,\qquad y=\theta .
$$

The transfer function follows from $G(s) = C(sI-A)^{-1}B + D$, and the **poles of the transfer function are the eigenvalues of
$A$** (when no pole-zero cancellation occurs). State-space models are not unique: any invertible change of coordinates
$\mathbf x = T\mathbf z$ gives $(T^{-1}AT,\ T^{-1}B,\ CT,\ D)$ with the same transfer function and the same eigenvalues.
Conversely, a transfer function has many realisations; one standard choice, the *controllable canonical form*, is used in
Notebook 1.6.
"""),

code(r'''
A, B, C, D = lc.dc_motor_ss(Kt, Ke, R, L, J, b, N)
display(Math(matrix_to_latex(A, 3, "A")))
print("eigenvalues of A      :", np.sort(np.linalg.eigvals(A).real).round(3))
print("poles of the TF Theta/V:", np.sort(G_th.poles().real).round(3))
G_from_ss = lc.ss_to_tf(A, B, C, D)
w = np.array([0.5, 5.0, 50.0, 500.0])
print("C(sI-A)^-1 B equals the transfer function:", np.allclose(G_from_ss(1j * w), G_th(1j * w)))

# a change of coordinates leaves the eigenvalues and the transfer function unchanged
rng = np.random.default_rng(0)
T = rng.normal(size=(3, 3)); Ti = np.linalg.inv(T)
A2, B2, C2 = Ti @ A @ T, Ti @ B, C @ T
print("similarity transform keeps eigenvalues:", np.allclose(np.sort(np.linalg.eigvals(A2).real), np.sort(np.linalg.eigvals(A).real)))
print("... and the transfer function         :", np.allclose(lc.ss_to_tf(A2, B2, C2, D)(1j * w), G_th(1j * w)))
'''),

md(r"""
**Exploring the design space.** The explorer below changes the gear ratio and the link mass of the reference joint (the viscous
friction scales as $b=N^2 b_m$ with $b_m=8\times10^{-6}$ N m s/rad) and redraws the speed step response and the pole positions of
$\Theta/V$. With a larger $N$ the reflected inertia grows as $N^2$ while the torque per volt grows only as $N$, so the joint becomes
**slower** ($\tau_m$ grows) but stronger — a trade-off every gearbox choice makes. Notice how weakly the response depends on the link mass when $N$ is large.
"""),

code(r'''
def explore(N_=50, m_=0.8):
    q = js.dc_joint_params(N=N_, m=m_, b=N_**2 * 8e-6)
    Gw_ = js.dc_joint_tf(q, "speed")
    Kw_ = q["N"] * q["Kt"] / (q["R"] * q["b"] + q["N"]**2 * q["Kt"] * q["Ke"])
    tm_ = q["R"] * q["J"] / (q["R"] * q["b"] + q["N"]**2 * q["Kt"] * q["Ke"])
    tt = np.linspace(0, 0.6, 600)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax[0].plot(tt, 24 * Gw_.step(tt), color=cp.BLUE, lw=2)
    ax[0].axhline(24 * Kw_, color=cp.GREY, ls="--")
    ax[0].set(xlabel="time (s)", ylabel="speed (rad/s) for a 24 V step", ylim=(0, 60),
              title=f"$N$={N_}, $m$={m_:.1f} kg: $J$={q['J']:.3f} kg m², $\\tau_m$={tm_ * 1e3:.0f} ms")
    cp.pz_map(ax[1], np.roots(np.polymul([q['L'], q['R']], [q['J'], q['b']]) + [0, 0, q['N']**2 * q['Kt'] * q['Ke']]))
    ax[1].set(xscale="symlog", xlim=(-3000, 5), title="poles of the speed model (log axis)")
    plt.tight_layout(); plt.show()

interact(explore, N_=widgets.IntSlider(value=50, min=10, max=150, step=5, description="N"),
         m_=widgets.FloatSlider(value=0.8, min=0.2, max=3.0, step=0.1, description="m (kg)"))
'''),

md(r"""
## 7. Linearisation

The only nonlinear term in our joint is gravity. For a uniform link of mass $m$ and length $\ell$ hanging from the pivot,
with $\theta=0$ pointing straight down,

$$
J\ddot\theta + b\dot\theta + m g\tfrac{\ell}{2}\sin\theta = \tau .
$$

Most of control theory is linear; the bridge is **linearisation**. For a system $\dot{\mathbf x} = f(\mathbf x,\mathbf u)$ and an
**equilibrium** $(\mathbf x_e,\mathbf u_e)$ with $f(\mathbf x_e,\mathbf u_e)=0$, let $\delta\mathbf x=\mathbf x-\mathbf x_e$ and
$\delta\mathbf u=\mathbf u-\mathbf u_e$. A first-order Taylor expansion gives

$$
\delta\dot{\mathbf x} \approx A\,\delta\mathbf x + B\,\delta\mathbf u,\qquad
A = \left.\frac{\partial f}{\partial\mathbf x}\right|_{(\mathbf x_e,\mathbf u_e)},\quad
B = \left.\frac{\partial f}{\partial\mathbf u}\right|_{(\mathbf x_e,\mathbf u_e)} .
$$

For the joint with state $(\theta,\omega)$ and torque input, the equilibria with $\tau=0$ are $\theta_e=0$ (hanging) and
$\theta_e=\pi$ (inverted), and

$$
\delta\ddot\theta = -\frac{b}{J}\,\delta\dot\theta - \underbrace{\frac{m g(\ell/2)\cos\theta_e}{J}}_{\omega_0^2\ \text{or}\ -\omega_0^2}\,\delta\theta + \frac{1}{J}\delta\tau,
\qquad \omega_0^2 = \frac{m g\,\ell/2}{J}.
$$

About the hanging equilibrium the linear model is a *stable* oscillator of natural frequency $\omega_0$ (gravity acts as a spring); about the
inverted equilibrium the sign of the stiffness flips and the model has a real **unstable** pole at $+\sqrt{\omega_0^2+b^2/4J^2}-b/2J \approx +\omega_0$ (gravity is a negative spring). A linear model can only be
trusted while the state stays near the equilibrium. How near? Compare the exact and the linearised free swing from several
amplitudes; for a pendulum without friction the exact period is $T = 4K(\sin\tfrac{\theta_0}{2})/\omega_0$ with $K$ the complete elliptic
integral, so that $T/T_0 = 2K/\pi$ increases with amplitude.
"""),

code(r'''
w0 = np.sqrt(p["mgd"] / J)

def swing(theta0, b_=0.0, T=6.0):
    f = lambda t, x: [x[1], -p["mgd"] / J * np.sin(x[0]) - b_ / J * x[1]]
    tt = np.linspace(0, T, 3001)
    return tt, solve_ivp(f, [0, T], [theta0, 0.0], t_eval=tt, rtol=1e-10, atol=1e-12).y[0]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a_deg, col in zip((10, 60, 120), (cp.BLUE, cp.ORANGE, cp.GREEN)):
    tt, th = swing(np.radians(a_deg))
    ax[0].plot(tt, np.degrees(th), color=col, lw=2, label=f"nonlinear, {a_deg}°")
    ax[0].plot(tt, a_deg * np.cos(w0 * tt), ":", color=col, lw=1.6)
ax[0].plot([], [], ":", color="k", label="linearised")
ax[0].set(xlabel="time (s)", ylabel="angle (deg)", title="free swing from rest, no friction"); ax[0].legend(fontsize=8, ncol=2)
amp = np.linspace(1, 170, 120)
ax[1].plot(amp, 2 * ellipk(np.sin(np.radians(amp) / 2) ** 2) / np.pi, color=cp.BLUE, lw=2)
ax[1].axhline(1, color=cp.GREY, ls="--")
ax[1].set(xlabel="amplitude (deg)", ylabel="$T / T_0$", title="period of the nonlinear swing relative to the linear model", ylim=(0.95, 2.6))
plt.tight_layout(); plt.show()

# the exact period from a simulation agrees with the elliptic-integral formula
tt, th = swing(np.radians(60), T=8.0)
cross = tt[:-1][np.sign(th[:-1]) * np.sign(th[1:]) < 0]
T_sim = 2 * (cross[1] - cross[0]) * 2 / 2 * 1.0       # two zero crossings of theta are half a period apart
T_sim = 2 * np.mean(np.diff(cross))
print(f"60 deg: simulated period {T_sim:.4f} s,  T0 = {2 * np.pi / w0:.4f} s,  T/T0 = {T_sim / (2 * np.pi / w0):.4f},  elliptic formula {2 * ellipk(np.sin(np.radians(60) / 2) ** 2) / np.pi:.4f}")
assert np.isclose(T_sim / (2 * np.pi / w0), 2 * ellipk(np.sin(np.radians(60) / 2) ** 2) / np.pi, rtol=2e-3)
'''),

md(r"""
For a swing of $10^\circ$ the linear model is excellent (the period is wrong by 0.2 %), at $60^\circ$ the period is 7 % too short, and at
$120^\circ$ it is 37 % too short and the waveform is visibly non-sinusoidal. A useful rule of thumb: sinusoidal approximations of $\sin\theta$ are good
to about 1 % up to $\theta\approx14^\circ$ and to 5 % up to $\approx 31^\circ$. The animation shows the same thing for a swing from $80^\circ$ with the
joint's real friction: the nonlinear link (orange) and the linearised model (blue) drift apart within the first swing.
"""),

code(r'''
th0 = np.radians(80)
f_nl = lambda t, x: [x[1], -p["mgd"] / J * np.sin(x[0]) - b / J * x[1]]
f_lin = lambda t, x: [x[1], -p["mgd"] / J * x[0] - b / J * x[1]]
tt = np.linspace(0, 4.0, 41)
nl = solve_ivp(f_nl, [0, 4], [th0, 0], t_eval=tt, rtol=1e-9).y[0]
li = solve_ivp(f_lin, [0, 4], [th0, 0], t_eval=tt, rtol=1e-9).y[0]
ell = p["ell"]
fig, ax = plt.subplots(figsize=(5.2, 4.2))

def update(k):
    ax.cla()
    for th, col, lab in ((nl[k], cp.ORANGE, "nonlinear"), (li[k], cp.BLUE, "linearised")):
        ax.plot([0, ell * np.sin(th)], [0, -ell * np.cos(th)], "-", color=col, lw=6, solid_capstyle="round", alpha=0.85, label=lab)
    ax.plot(0, 0, "ko", ms=8)
    ax.plot(np.degrees(np.linspace(-1, 1, 3)) * 0, [0] * 3, alpha=0)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.15); ax.set_aspect("equal")
    ax.set_title(f"t = {tt[k]:.1f} s:  nonlinear {np.degrees(nl[k]):6.1f}°,  linear {np.degrees(li[k]):6.1f}°", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
animate(fig, update, range(len(tt)), interval=90, gif="assets/1.1-linearisation-swing.gif", n_keyframes=4)
'''),

md(r"""
**Linearising the whole voltage-driven joint, and asking MuJoCo to do it.** The same procedure applies to the full three-state
model: only the term $-\dfrac{mg\ell/2}{J}\cos\theta_e\,\theta$ enters the second row of $A$ (we call $k_g = m g\ell\cos\theta_e/2$ the **gravity stiffness**; `dc_motor_ss(..., kg=k_g)`
implements it). MuJoCo can also linearise *itself*: `mjd_transitionFD` differentiates one simulation step by finite differences,
and `joint_sim.fd_linearize` converts the resulting discrete map into continuous-time $(A, B)$. The eigenvalues must match the
analytic ones — at $\theta_e=0$ all three are in the left half-plane, at $\theta_e=\pi$ exactly one is not.
"""),

code(r'''
model_g, data_g = mju.load(js.dc_joint_xml(p, gravity=True))
for th_e, sign in ((0.0, +1), (np.pi, -1)):
    data_g.qpos[0], data_g.qvel[0], data_g.act[:], data_g.ctrl[:] = th_e, 0.0, 0.0, 0.0     # equilibrium: zero current, zero voltage
    A_fd, B_fd, _, _ = js.fd_linearize(model_g, data_g)
    A_an, B_an, _, _ = lc.dc_motor_ss(Kt, Ke, R, L, J, b, N, kg=sign * p["mgd"])
    ev_fd, ev_an = np.sort(np.linalg.eigvals(A_fd).real), np.sort(np.linalg.eigvals(A_an).real)
    print(f"theta_e = {np.degrees(th_e):5.0f} deg   eigenvalues  MuJoCo FD: {ev_fd.round(3)}   analytic: {ev_an.round(3)}")
    assert np.allclose(ev_fd, ev_an, rtol=5e-3, atol=2e-2)
    print("                 unstable eigenvalues:", int(np.sum(ev_fd > 0)))
'''),

md(r"""
## 8. Real robots: reading a joint out of a MuJoCo model

Robot-model files hide the quantities a control engineer needs inside an XML file and in compiled arrays. Here we extract them
for two real robots and check them by hand and by simulation.

**The Dynamixel 2R** is a two-link planar arm built from Dynamixel servos, hanging in a vertical plane (its model comes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie); see `robots.describe`). Both joints have a `position`
actuator with $k_p=50$ N m/rad and `dampratio=1`, a rotor `armature` of 0.005 kg m² and dry friction (`frictionloss` 0.1 N m).
A MuJoCo position actuator produces the torque

$$
\tau = k_p\,(u - \theta) - k_v\,\dot\theta,
$$

a *proportional–derivative servo*: `gainprm[0]` stores $k_p$ and `biasprm` stores $[0,\,-k_p,\,-k_v]$. The XML never states
$k_v$ — `dampratio` asks the compiler to compute it so that the servo is critically damped for the **inertia that the joint has when all
other joints are free**, i.e. $1/(M^{-1})_{jj}$. When the neighbouring joint is not free, but locked, the true inertia is the larger
$M_{jj}$ and the servo is *under*damped. The next cell reads everything out and rebuilds the inertia by hand from the XML numbers.
"""),

code(r'''
m2, d2 = robots.load_robot("2r")                       # Dynamixel 2R, light scene
print(robots.describe("2r"))
g = js.servo_gains(m2, "R2")                           # the distal joint R2
dof = g["dof"]
M2 = js.mass_matrix(m2, d2)
J_locked, J_free = M2[dof, dof], 1 / np.linalg.inv(M2)[dof, dof]
print(f"R2 servo:  kp = {g['kp']:.1f} N m/rad,  kv = {g['kv']:.4f} N m s/rad   (read from gainprm / biasprm)")
print(f"inertia with R1 locked  M[2,2]       = {J_locked:.5f} kg m^2")
print(f"inertia with R1 free    1/(M^-1)[2,2] = {J_free:.5f} kg m^2")
print(f"dampratio = 1 means kv = 2 sqrt(kp J_free) = {2 * np.sqrt(g['kp'] * J_free):.4f}   (stored value {g['kv']:.4f})")
assert np.isclose(g["kv"], 2 * np.sqrt(g["kp"] * J_free), rtol=1e-4)

# the locked-joint inertia from the XML numbers: armature + inertia about the axis through the COM + m d^2
body = m2.body("second_segment")
Ri = np.zeros(9); mujoco.mju_quat2Mat(Ri, body.iquat); Ri = Ri.reshape(3, 3)
I_com = Ri @ np.diag(body.inertia) @ Ri.T              # inertia tensor about the COM, in the body frame
axis = m2.jnt_axis[dof]; r = body.ipos                 # joint axis and COM position (joint at the body origin)
J_hand = m2.dof_armature[dof] + axis @ I_com @ axis + body.mass[0] * (r @ r - (axis @ r) ** 2)
print(f"by hand: armature {m2.dof_armature[dof]:.4f} + I_axis {axis @ I_com @ axis:.5f} + m d^2 {body.mass[0] * (r @ r - (axis @ r) ** 2):.5f} = {J_hand:.5f}")
assert np.isclose(J_hand, J_locked, rtol=1e-6)

kg = js.gravity_stiffness(m2, d2, dof)
kg_hand = body.mass[0] * 9.81 * np.sqrt(r @ r - (axis @ r) ** 2)
print(f"gravity stiffness at the hanging pose: finite differences {kg:.4f},  m g d = {kg_hand:.4f} N m/rad")
assert np.isclose(kg, kg_hand, rtol=1e-4)
wn = np.sqrt((g["kp"] + kg) / J_locked); zeta = g["kv"] / (2 * np.sqrt((g["kp"] + kg) * J_locked))
print(f"closed-loop servo with R1 locked: omega_n = {wn:.2f} rad/s, zeta = {zeta:.3f}   (the subject of Notebook 1.2)")
'''),

md(r"""
The hand calculation reproduces MuJoCo's inertia to six digits, and the gravity stiffness is simply $mgd$, the weight times the distance of
the centre of mass from the axis. We also learn something about the model: the Dynamixel servo is *critically damped only
when its neighbour is free*; with the neighbour held fixed it has $\zeta\approx0.44$ and will overshoot by about 20 % — a model
fact that a designer must discover by reading the model, not by trusting its XML attribute. Notebook 1.2 confirms this in simulation.

As a check of the gravity-stiffness idea, release the link by switching the actuators off, lock R1, and let R2 swing freely
with a small amplitude: it should oscillate at $\omega_0=\sqrt{k_g/J}$ — the same formula as for our reference pendulum.
"""),

code(r'''
m2.dof_frictionloss[:] = 0.0                                  # remove dry friction for a clean oscillation
m2.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION  # servos off: a passive pendulum
m2.opt.timestep = 5e-4
with js.locked(m2, d2, [0]):
    d2.qpos[:] = [0.0, 0.05]; d2.qvel[:] = 0.0; d2.time = 0.0
    mujoco.mj_forward(m2, d2)
    ts, qs = [], []
    for _ in range(int(4.0 / m2.opt.timestep)):
        mujoco.mj_step(m2, d2); ts.append(d2.time); qs.append(d2.qpos[dof])
ts, qs = np.array(ts), np.array(qs)
zc = ts[:-1][np.sign(qs[:-1]) * np.sign(qs[1:]) < 0]
T_meas = 2 * np.mean(np.diff(zc))
T_pred = 2 * np.pi / np.sqrt(kg / J_locked)
print(f"small-oscillation period: MuJoCo {T_meas:.4f} s,  2 pi sqrt(J/kg) = {T_pred:.4f} s  ({100 * (T_meas / T_pred - 1):+.2f} %)")
assert abs(T_meas / T_pred - 1) < 0.01
m2.opt.disableflags = 0                                         # servos back on
'''),

md(r"""
**The UR5e** ([Universal Robots](https://www.universal-robots.com/); Menagerie model) uses the same pattern with different numbers:
its six actuators are `general` position servos, three large ones ($k_p=2000$, $k_v=400$) and three wrist ones ($k_p=500$,
$k_v=100$, torque limit 28 N m), every joint with a rotor `armature` of 0.1 kg m². With the other joints locked at the manufacturer's
home pose, each joint's inertia is $M_{jj}$ and its servo damping ratio is $\zeta = k_v/2\sqrt{k_pJ}$.
"""),

code(r'''
m3, d3 = robots.load_robot("ur5e")
print(robots.describe("ur5e"))
M3 = js.mass_matrix(m3, d3)
rows = []
for i in range(m3.nu):
    gi = js.servo_gains(m3, i)
    Ji = M3[gi["dof"], gi["dof"]]
    rows.append((m3.actuator(i).name, gi["kp"], gi["kv"], m3.dof_armature[gi["dof"]], Ji,
                 gi["kv"] / (2 * np.sqrt(gi["kp"] * Ji)), -1 / np.roots([Ji, gi["kv"], gi["kp"]]).real.max() ** -1 if False else np.roots([Ji, gi["kv"], gi["kp"]])))
tab = "| joint | k_p (N m/rad) | k_v (N m s/rad) | armature (kg m²) | J at home (kg m²) | ζ | poles of the servo (rad/s) |\n|:--|--:|--:|--:|--:|--:|:--|\n"
for name, kp_, kv_, arm, Ji, zt, pl in rows:
    tab += f"| {name} | {kp_:.0f} | {kv_:.0f} | {arm:.2f} | {Ji:.3f} | {zt:.2f} | {np.round(np.sort(pl.real)[::-1], 1).tolist()} |\n"
display(Markdown(tab))
img = mju.render(m3, d3, camera=dict(azimuth=140, elevation=-20, distance=1.9, lookat=(0, 0, 0.35)), width=420, height=320)
img2 = mju.render(m2, d2, camera=dict(azimuth=90, elevation=0, distance=1.3, lookat=(0, 0, 0.3)), width=420, height=320)
filmstrip([img, img2], ["UR5e at the manufacturer's home pose", "Dynamixel 2R hanging at $q=0$"], width=8)
'''),

md(r"""
Every UR5e joint is **heavily overdamped** ($\zeta>1$, between 2.8 and 7). The reason is the large armature: a rotor inertia of 0.1 kg m² dominates, and the gains
were chosen for robust, smooth motion rather than speed. An overdamped second-order servo behaves like a first-order lag whose
time constant is approximately $k_v/k_p$ — 0.2 s for the wrist joints. This is a typical modelling situation: the *simulator's*
servo is a design decision of the model's author, not a law of nature. In Notebook 1.2 we shall identify these joints from step
responses, as one would a real robot.
"""),

md(r"""
## 9. Worked examples

**Example 1.1.1 — reflecting motor quantities to the joint.** The motor of the reference joint has a rotor inertia
$J_m = 3\times10^{-5}$ kg m² and a viscous friction $b_m=8\times10^{-6}$ N m s/rad at its shaft, behind $N=50$. The link has
$J_L = m\ell^2/3$. Find the joint inertia and friction and the fraction of the inertia that belongs to the rotor.

*Solution.* $J = N^2J_m + J_L = 2500\cdot3\times10^{-5} + 0.8\cdot0.4^2/3 = 0.075 + 0.0427 = 0.1177$ kg m²;
$b = N^2b_m = 0.020$ N m s/rad. The rotor supplies $0.075/0.1177 = 63.7\,\%$ of the inertia, which is why the joint is only mildly
sensitive to the link.

**Example 1.1.2 — block-diagram reduction of a cascade loop.** A common robot-servo structure has an inner **velocity loop** and an outer **position loop**. The joint's
speed model is $K_\omega/(\tau_ms+1)$ with $K_\omega = 0.660$ (rad/s)/V and $\tau_m=62.1$ ms. The inner loop uses a proportional gain
$K_v = 5$ V/(rad/s) on the measured speed; the outer loop sends the velocity command $K_p(\theta_d-\theta)$ with $K_p = 30$ s$^{-1}$ to the inner loop.
Find the closed-loop transfer function $\Theta/\Theta_d$.

*Solution.* Inner loop: $T_v = \dfrac{K_vK_\omega}{\tau_ms+1+K_vK_\omega}$, which has DC gain $K_vK_\omega/(1+K_vK_\omega)=0.767$ and a pole
at $-(1+K_vK_\omega)/\tau_m = -69.2$ rad/s. The position loop is $L = K_pT_v/s$, so
$\Theta/\Theta_d = L/(1+L) = \dfrac{K_pK_vK_\omega}{\tau_ms^2+(1+K_vK_\omega)s+K_pK_vK_\omega}$, a second-order system with unit DC gain,
$\omega_n=39.9$ rad/s and $\zeta=0.867$ (poles $-34.6\pm19.9j$). Reduction by block-diagram rules thus turned a two-loop structure into a standard second-order
system we can reason about; Notebook 1.2 predicts its overshoot (about 0.6 %).

**Example 1.1.3 — inertia matching.** For $J_m=3\times10^{-5}$ and $J_L=0.0427$ find the gear ratio that maximises the load acceleration for a given motor
torque and compare with $N=50$.

*Solution.* $N^*=\sqrt{J_L/J_m}=37.7$. The acceleration per unit motor torque is $N/(N^2J_m+J_L)$: $441.9$ rad/s² per N m at $N^*$ against $424.9$ at $N=50$ —
the choice $N=50$ gives up only 3.8 % of the achievable acceleration (the optimum is shallow).

**Example 1.1.4 — how good is the first-order motor model?** Compare $\Omega/V$ with $K_\omega/(\tau_ms+1)$ for the reference joint.

*Solution.* $\tau_e=L/R=1.00$ ms and $\tau_m=62.1$ ms, a ratio of 62. The largest difference between the two speed step responses is 1.46 % of the final value (reached at about 3.6 ms), and it decays with $\tau_m$
afterwards. For control bandwidths well below $1/\tau_e = 1000$ rad/s the first-order model is therefore adequate; the full model is needed when the loop crosses over near
the electrical pole, or when the current loop matters.

**Example 1.1.5 — from state space to transfer function.** Using the $(A,B,C,D)$ of Section 6, find $\Theta/V$ and its poles.

*Solution.* $C(sI-A)^{-1}B$ gives $\Theta/V = \dfrac{1.062\times10^{4}}{s^3+1000\,s^2+1.61\times10^{4}\,s}$, poles $0$, $-16.4$, $-984$ rad/s — the eigenvalues of $A$
(the zero eigenvalue is the free integration from speed to angle).

**Example 1.1.6 — period of the hanging link.** Predict the small-oscillation period of the reference link with the motor unpowered and friction ignored, and
the period of a $60^\circ$ swing.

*Solution.* $\omega_0=\sqrt{mg\ell/2J}=\sqrt{1.5696/0.1177}=3.652$ rad/s, $T_0=2\pi/\omega_0=1.720$ s. For $60^\circ$, $T/T_0 = 2K(k^2=\sin^2 30^\circ)/\pi=1.0732$, i.e. $T=1.846$ s,
which the simulation of Section 7 reproduced.
"""),

code(r'''
# Example 1.1.1
Jm, bm, JL = p["Jm"], 8e-6, p["J_link"]
J1, b1 = N**2 * Jm + JL, N**2 * bm
print("Ex 1.1.1  J =", round(J1, 4), " b =", round(b1, 4), " rotor fraction =", round(N**2 * Jm / J1, 3))
assert np.isclose(J1, 0.1177, atol=5e-5) and np.isclose(b1, 0.020) and np.isclose(N**2 * Jm / J1, 0.637, atol=1e-3)

# Example 1.1.2
Kv_, Kp_ = 5.0, 30.0
Gw1 = lc.first_order(K_w, tau_m)
Tv = (Gw1 * Kv_).feedback()                                  # inner loop
T_cl = (Tv * Kp_ * TF([1.0], [1.0, 0.0])).feedback()          # outer loop
wn_, z_ = np.sqrt(T_cl.den[2] / T_cl.den[0]), T_cl.den[1] / T_cl.den[0] / (2 * np.sqrt(T_cl.den[2] / T_cl.den[0]))
print("Ex 1.1.2  T_v pole", Tv.poles().round(1), " DC gain", round(Tv.dcgain(), 3), "| closed loop poles", T_cl.poles().round(1), " wn, zeta =", round(wn_, 1), round(z_, 3), " dc", T_cl.dcgain())
assert np.isclose(Tv.poles()[0], -69.2, atol=0.1) and np.isclose(Tv.dcgain(), 0.767, atol=1e-3)
assert np.isclose(wn_, 39.9, atol=0.1) and np.isclose(z_, 0.867, atol=1e-3) and np.isclose(T_cl.dcgain(), 1.0)
pp = np.sort_complex(T_cl.poles()); assert np.isclose(pp[0].real, -34.6, atol=0.1) and np.isclose(abs(pp[0].imag), 19.9, atol=0.1)

# Example 1.1.3
Nstar = np.sqrt(JL / Jm); acc = lambda n: n / (n**2 * Jm + JL)
print("Ex 1.1.3  N* =", round(Nstar, 1), " acc(N*) =", round(acc(Nstar), 1), " acc(50) =", round(acc(50), 1), " loss =", round(100 * (1 - acc(50) / acc(Nstar)), 1), "%")
assert np.isclose(Nstar, 37.7, atol=0.05) and np.isclose(acc(Nstar), 441.9, atol=0.1) and np.isclose(acc(50), 424.9, atol=0.1)
assert acc(Nstar) == max(acc(n) for n in np.linspace(5, 200, 4000)) or np.isclose(acc(Nstar), max(acc(n) for n in np.linspace(5, 200, 40000)), rtol=1e-6)

# Example 1.1.4
t = np.linspace(0, 0.5, 50001)
e = np.abs(G_w.step(t) - Gw1.step(t)) / K_w
print("Ex 1.1.4  tau_e =", L / R * 1e3, "ms, tau_m =", round(tau_m * 1e3, 1), "ms, max rel. difference =", round(100 * e.max(), 2), "% at t =", round(t[e.argmax()] * 1e3, 1), "ms")
assert np.isclose(e.max(), 0.0146, atol=2e-4) and np.isclose(t[e.argmax()] * 1e3, 3.6, atol=0.2) and np.isclose(tau_m * 1e3, 62.1, atol=0.05)

# Example 1.1.5
print("Ex 1.1.5  ", G_from_ss, " poles", np.sort(G_from_ss.poles().real).round(1))
assert np.isclose(G_from_ss.num[-1], 1.062e4, rtol=1e-3) and np.allclose(G_from_ss.den[:3], [1, 1000, 1.61e4], rtol=2e-3)

# Example 1.1.6
T0 = 2 * np.pi / w0; ratio60 = 2 * ellipk(np.sin(np.radians(30)) ** 2) / np.pi
print("Ex 1.1.6  w0 =", round(w0, 3), " T0 =", round(T0, 3), " T/T0 (60 deg) =", round(ratio60, 4), " T =", round(T0 * ratio60, 3))
assert np.isclose(w0, 3.652, atol=5e-4) and np.isclose(T0, 1.720, atol=5e-4) and np.isclose(ratio60, 1.0732, atol=5e-5) and np.isclose(T0 * ratio60, 1.846, atol=1e-3)
'''),

md(r"""
## 10. Summary

| Result | Formula |
|:--|:--|
| Reflection through a gearbox | $J = N^2J_m + J_L$, $\ b = N^2b_m + b_L$ |
| DC motor | $V = Ri + L\,di/dt + K_e\omega_m$, $\ \tau_m = K_ti$, $\ K_t=K_e$ (SI) |
| Voltage-to-speed | $\dfrac{\Omega}{V}=\dfrac{NK_t}{(Ls+R)(Js+b)+N^2K_tK_e}\approx\dfrac{K_\omega}{\tau_ms+1}$ |
| Time constants | $\tau_m = \dfrac{RJ}{Rb+N^2K_tK_e}$, $\ \tau_e = L/R$ |
| Block-diagram loop | $\dfrac{Y}{R}=\dfrac{CG}{1+CGH}$, $\ \dfrac{Y}{D}=\dfrac{G}{1+CGH}$ |
| State space | $\dot{\mathbf x}=A\mathbf x+Bu$, $\ y=C\mathbf x+Du$; $\ G=C(sI-A)^{-1}B+D$; poles $=\operatorname{eig}A$ |
| Linearisation | $A=\partial f/\partial\mathbf x$, $B=\partial f/\partial\mathbf u$ at an equilibrium; gravity stiffness $\pm mg\ell/2$ |
| Inertia matching | $N^*=\sqrt{J_L/J_m}$ maximises load acceleration |
| MuJoCo position servo | $\tau=k_p(u-q)-k_v\dot q$ with $k_p=$ `gainprm[0]`, $k_v=-$`biasprm[2]` |

### Common mistakes

* Forgetting the factor $N^2$: rotor inertia and friction are multiplied by $N^2$, but the torque only by $N$.
* Using $K_t$ in N m/A and $K_e$ in V/(rpm) (or $K_e$ in V/(krpm)) without converting: in consistent SI units they are equal.
* Treating a disturbance entering at the plant input as if it entered at the output; the transfer functions $G/(1+CGH)$ and $1/(1+CGH)$ differ.
* Linearising about a point that is not an equilibrium, or trusting a linear model for large swings (the period of a $120^\circ$ swing is off by 37 %).
* Assuming `dampratio=1` in a MuJoCo model gives critical damping in every configuration: it fixes $k_v$ once, for the compile-time inertia.
* Reading the sign of gravity stiffness wrongly: it is *positive* for a link hanging below its pivot and *negative* above it.
"""),

md(r"""
## Exercises

**1.1.1** A flexible-joint approximation models a joint with a torsional spring of stiffness $k=50$ N m/rad (in parallel with
the viscous friction) as $J\ddot\theta+b\dot\theta+k\theta=\tau$, with $J=0.1177$ kg m² and $b=0.02$ N m s/rad. Write the transfer function
$\Theta/T$ and find $\omega_n$, $\zeta$ and the DC gain.

<details><summary>Answer</summary>

$G(s)=1/(Js^2+bs+k)$; $\omega_n=\sqrt{k/J}=20.6$ rad/s, $\zeta=b/(2\sqrt{kJ})=0.0041$ (almost undamped), DC gain $1/k=0.02$ rad/(N m).
</details>

**1.1.2** In the loop of Section 3 take $C=K=20$, $G=1/(s(0.1177s+0.02))$ and $H=1$. Find $\Theta/D$, the transfer function from a torque disturbance at the plant input to the angle, its DC gain, and the
steady-state angle error under a constant disturbance of 1 N m.

<details><summary>Answer</summary>

$\Theta/D = G/(1+CG) = 1/(0.1177s^2+0.02s+20)$. DC gain $1/K=0.05$ rad/(N m), so a constant 1 N m torque leaves a steady-state angle of $0.05$ rad ($2.9^\circ$): proportional control of this joint cannot reject a constant torque exactly.
</details>

**1.1.3** For the reference motor ($J_m=3\times10^{-5}$ kg m²) and link, compute the factor by which $J$ changes when the link inertia triples, for $N=1$, $20$ and $100$.

<details><summary>Answer</summary>

$J(3J_L)/J(J_L)=2.999,\ 2.561,\ 1.249$: with $N=1$ the joint inertia triples; with $N=100$ it grows by only 25 %.
</details>

**1.1.4** Write a state-space model for $\dddot y+6\ddot y+11\dot y+6y=6u$ with $\mathbf x=[y,\dot y,\ddot y]^{\mathsf T}$, and find its poles.

<details><summary>Answer</summary>

$A=\begin{bmatrix}0&1&0\\0&0&1\\-6&-11&-6\end{bmatrix}$, $B=[0,0,6]^{\mathsf T}$, $C=[1,0,0]$. Poles $-1,-2,-3$ (the roots of $s^3+6s^2+11s+6$); DC gain 1.
</details>

**1.1.5** *(linearisation)* The joint with quadratic air drag obeys $J\ddot\theta+c\,|\dot\theta|\dot\theta+mg\tfrac{\ell}{2}\sin\theta=\tau$, $c=0.01$. Linearise it about the *horizontal* equilibrium $\theta_e=\pi/2$ held by $\tau_e=mg\ell/2$. What is special about the result?

<details><summary>Answer</summary>

$A=\begin{bmatrix}0&1\\-\tfrac{mg\ell}{2J}\cos\theta_e & -\tfrac{2c|\omega_e|}{J}\end{bmatrix}=\begin{bmatrix}0&1\\0&0\end{bmatrix}$, $B=[0,\,1/J]^{\mathsf T}$: a **double integrator** (both poles at the origin). At the horizontal pose gravity is neither a spring nor an anti-spring, and quadratic drag has zero slope at zero speed. The linear model is only marginally stable — the nonlinear terms decide the true behaviour.
</details>

**1.1.6** *(MuJoCo)* In the UR5e model, lock all joints except `wrist_3` at the home pose. Predict the servo poles from $k_p$, $k_v$ and $J=M_{66}$, then check them against MuJoCo's own linearisation (`fd_linearize`; use a time step of $10^{-4}$ s).

<details><summary>Answer</summary>

Predicted roots of $Js^2+k_vs+k_p$ with $J=0.1001$: $-5.03$ and $-993.7$ rad/s. MuJoCo's finite-difference linearisation reproduces both to within 1 %.
</details>

**1.1.7** *(robot-joint design)* A joint must carry a link of inertia $J_L=0.5$ kg m² and give it an acceleration of at least $10$ rad/s² up to a speed of $3$ rad/s, using a motor with $J_m=5\times10^{-5}$ kg m², $K_t=K_e=0.05$ N m/A, $R=0.8\ \Omega$ and a $24$ V supply (ignore friction and gravity). (a) Find the matched gear ratio $N^*$. (b) At $N^*$, what motor torque is needed for the acceleration, and what current? (c) Can the motor reach the speed target from $24$ V?

<details><summary>Answer</summary>

(a) $N^*=\sqrt{J_L/J_m}=100$. (b) The total joint inertia is $2J_L=1.0$ kg m², so $\tau=J\alpha = 10$ N m at the joint, i.e. a motor torque of $0.10$ N m and a current of $2.0$ A. (c) The no-load speed is $V/(NK_e)=24/(100\cdot0.05)=4.8$ rad/s $>3$ rad/s; with the speed at 3 rad/s the back-EMF is $15$ V, leaving $9$ V to drive $R i$: $i_{\max}=11.25$ A $\gg2$ A, so the targets are met with margin.
</details>
"""),

code(r'''
# --- verification of the exercise answers ---------------------------------------------------------------
# 1.1.1
Jx, k = p["J"], 50.0
print("1.1.1", round(np.sqrt(k / Jx), 1), round(p["b"] / (2 * np.sqrt(k * Jx)), 4), 1 / k)
assert np.isclose(np.sqrt(k / Jx), 20.6, atol=0.05) and np.isclose(p["b"] / (2 * np.sqrt(k * Jx)), 0.0041, atol=5e-5)

# 1.1.2
Cx, Gx = TF([20.0]), TF([1.0], [0.1177, 0.02, 0.0])
Td_ = Gx.feedback(Cx)
print("1.1.2", Td_, " DC gain", Td_.dcgain())
assert np.allclose(Td_.den, [0.1177, 0.02, 20.0]) and np.isclose(Td_.dcgain(), 0.05)

# 1.1.3
for n_ in (1, 20, 100):
    print("1.1.3  N =", n_, round((n_**2 * Jm + 3 * JL) / (n_**2 * Jm + JL), 3))
assert np.allclose([(n_**2 * Jm + 3 * JL) / (n_**2 * Jm + JL) for n_ in (1, 20, 100)], [2.999, 2.561, 1.249], atol=1e-3)

# 1.1.4
Aq, Bq, Cq, Dq = lc.controllable_canonical([6.0], [1, 6, 11, 6])
print("1.1.4  A =", Aq.tolist(), " B =", Bq.ravel(), " poles", np.sort(np.linalg.eigvals(Aq).real))
assert np.allclose(Aq, [[0, 1, 0], [0, 0, 1], [-6, -11, -6]]) and np.allclose(Bq.ravel(), [0, 0, 1]) and np.allclose(Cq, [[6, 0, 0]])
assert np.allclose(np.sort(np.linalg.eigvals(Aq).real), [-3, -2, -1])

# 1.1.5
c_ = 0.01
fx = lambda x, u: np.array([x[1], (u[0] - c_ * abs(x[1]) * x[1] - p["mgd"] * np.sin(x[0])) / J])
Ax, Bx = lc.linearize(fx, [np.pi / 2, 0.0], [p["mgd"]])
print("1.1.5  A =", Ax.round(6).tolist(), " B =", Bx.ravel().round(3))
assert np.allclose(Ax, [[0, 1], [0, 0]], atol=1e-5) and np.allclose(Bx.ravel(), [0, 1 / J])

# 1.1.6
m3b, d3b = robots.load_robot("ur5e", scene="bare")
m3b.opt.timestep = 1e-4
w3 = m3b.actuator("wrist_3").trnid[0]; dof3 = int(m3b.jnt_dofadr[w3])
g3 = js.servo_gains(m3b, "wrist_3")
J3 = js.effective_inertia(m3b, d3b, dof3)
pred = np.sort(np.roots([J3, g3["kv"], g3["kp"]]).real)
others = [i for i in range(m3b.nv) if i != dof3]
with js.locked(m3b, d3b, others):
    for i in range(m3b.nu):
        d3b.ctrl[i] = d3b.qpos[int(m3b.jnt_qposadr[m3b.actuator_trnid[i, 0]])]
    A_fd, B_fd, _, _ = js.fd_linearize(m3b, d3b)
ev = np.linalg.eigvals(A_fd).real
ev = np.sort(ev[np.abs(ev) > 1.0])                     # discard the locked joints' (near-zero) modes
print("1.1.6  predicted", pred.round(2), " MuJoCo FD", ev.round(2))
assert np.allclose(ev, pred, rtol=0.01) and np.allclose(pred, [-993.7, -5.03], atol=0.1)

# 1.1.7
Jm7, JL7, Kt7, R7, V7 = 5e-5, 0.5, 0.05, 0.8, 24.0
N7 = np.sqrt(JL7 / Jm7); J7 = N7**2 * Jm7 + JL7
tau_j = J7 * 10; tau_m7 = tau_j / N7; i7 = tau_m7 / Kt7
w_nl = V7 / (N7 * Kt7); i_max = (V7 - N7 * Kt7 * 3.0) / R7
print("1.1.7  N* =", N7, " J =", J7, " tau_motor =", tau_m7, " i =", i7, " no-load speed =", w_nl, " i_max at 3 rad/s =", i_max)
assert np.isclose(N7, 100) and np.isclose(J7, 1.0) and np.isclose(tau_m7, 0.10) and np.isclose(i7, 2.0) and np.isclose(w_nl, 4.8) and np.isclose(i_max, 11.25)
'''),

md(r"""
## Further reading and lectures

* Ogata, Chs. 2–4 (modelling of mechanical and electrical systems; the DC motor); Nise, the chapters on modelling in the frequency and time domains, including the DC-motor and gear-train examples.
* Craig, Ch. 9: the single-joint model, effective inertia and the decoupling argument; Spong, Hutchinson & Vidyasagar, Ch. 8: the DC-motor model and the effect of gear ratio; Siciliano et al., Ch. 8.
* MuJoCo documentation: [actuators](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator) (`position`, `dcmotor`) and [`mjd_transitionFD`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).
* Lectures: """ + f"{NANDI2}; {BRUNTON}; {TECHTALKS}; {NPTEL_RMC}." + r"""
* The Python Control Systems Library ([python-control.org](https://python-control.org)) offers the same operations with a full-featured API; this chapter implements the essentials in `rgkit.linear_control` so that every line is visible.
"""),

nav_for(K),
]

if __name__ == "__main__":
    build(cells, f"{CH_DIR}/{FN}")
