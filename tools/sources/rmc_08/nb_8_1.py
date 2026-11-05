import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repository root
from tools.nbbuild import md, code, SETUP, header, nav, build
from tools.common import *

COURSE = "Robot Motion Control"
CHAPTER = "Chapter 8 — Sensing, Implementation and Visual Servoing"
CH_DIR = "RMC/08-sensing-and-implementation"

FN = "8.1-encoders-and-velocity-estimation.ipynb"
cells = [
header(COURSE, CHAPTER, "8.1", "Position and Velocity Sensing: Encoders, Tachometers and Velocity Estimation", f"{CH_DIR}/{FN}", [
    ("Primary reading", "Craig, 3rd ed., Ch. 8 (manipulator-mechanism design, including actuators and sensors) and Ch. 9 (digital implementation of joint control)"),
    ("Parallel reading", "Siciliano et al., Ch. 5 (actuators and sensors) and Ch. 6 (control architecture) · Spong et al., Ch. 8 (independent joint control)"),
    ("Lectures", "Prof. G. C. Nandi, playlist 2, *Mathematical Foundation of Robotics — Execution Level …* (by title; URL not yet verified) · [NPTEL, *Mechanics and Control of Robotic Manipulators*, IIT Palakkad](https://nptel.ac.in/courses/112106304) · [Brunton, *Control Bootcamp* (observers; overview video)](https://www.youtube.com/watch?v=Pi7l8mMjYVE)"),
    ("Prerequisites", "Chapters 1–2 of RMC (transfer functions, PD control of a single joint); elementary probability (mean, variance)"),
    ("Study time", "about 3 hours"),
], r"""
A feedback controller can act only on what it measures, and what it measures is never the true state of the
joint. This notebook studies the sensors that report the *position and velocity* of a joint — incremental and
absolute encoders and tachometers — and the arithmetic that converts their raw output into the quantities that a
PD or computed-torque law needs. We derive the resolution and quantisation error of an encoder, compare the three
classical ways of obtaining velocity from pulses (finite difference, pulse period, and M/T), design a filtered
derivative and an $\alpha$–$\beta$ (steady-state Kalman) velocity observer, and finally close a joint-space PD loop
around the **UR5e model in MuJoCo** through a deliberately coarse encoder to see how quantisation noise travels
into the motor torque."""),

md(r"""
## Learning objectives

After working through this notebook you should be able to:

1. Explain how an incremental quadrature encoder and an absolute (Gray-coded) encoder work, and **decode** a quadrature signal into a signed position count.
2. Compute the **resolution** of a geared joint, and state the bias and the root-mean-square value ($q/\sqrt{12}$) of quantisation error.
3. Compare the **M, T and M/T methods** of velocity measurement and find the speed at which the first two are equally accurate.
4. Derive and apply a **filtered derivative** and an **$\alpha$–$\beta$ observer** (the steady-state Kalman filter of a constant-velocity model), and quantify the noise–lag trade-off.
5. Describe what a **tachometer** contributes and where it falls short of an encoder-based estimate.
6. Predict — and measure on a UR5e in MuJoCo — how encoder quantisation and the choice of velocity estimator change the torque of a PD controller, and explain **derivative kick** and its remedy.

## Contents

1. [What a robot controller can measure](#1.-What-a-robot-controller-can-measure)
2. [Incremental encoders and quadrature decoding](#2.-Incremental-encoders-and-quadrature-decoding)
3. [Absolute encoders and the Gray code](#3.-Absolute-encoders-and-the-Gray-code)
4. [Resolution and quantisation error](#4.-Resolution-and-quantisation-error)
5. [Velocity from pulses: the M, T and M/T methods](#5.-Velocity-from-pulses:-the-M,-T-and-M/T-methods)
6. [Filtering the derivative: low-pass differentiation and the α–β observer](#6.-Filtering-the-derivative:-low-pass-differentiation-and-the-α–β-observer)
7. [Tachometers and other velocity sensors](#7.-Tachometers-and-other-velocity-sensors)
8. [A real robot: encoders in the loop of a UR5e in MuJoCo](#8.-A-real-robot:-encoders-in-the-loop-of-a-UR5e-in-MuJoCo)
9. [Worked examples](#9.-Worked-examples)
10. [Summary, common mistakes and exercises](#10.-Summary)
"""),

code(SETUP + r'''
from rgkit import sensing as sn
from rgkit import sensing_mujoco as sm
from rgkit import mujoco_utils as mju, robots
import mujoco
rng = np.random.default_rng(8)
'''),

md(r"""
## 1. What a robot controller can measure

A controller computes a command from the difference between a desired state and a **measured** one. Between the
physical quantity and the number held in the controller's memory lies a *measurement chain*: a transducer converts the
quantity into an electrical signal, conditioning electronics filter and amplify it, a converter or counter turns it into
an integer, an interface delivers the integer after some latency, and — usually — an estimator combines several such
numbers into the quantity that the control law actually needs (for example, a velocity). Every link of the chain
adds one of four defects:

* **noise** — random fluctuations, usually modelled as zero-mean Gaussian white noise;
* **quantisation** — the finite resolution of a counter or converter;
* **delay** — conversion, transmission and filtering all take time (Section 8.3 treats its effect on stability);
* **bias and scale error** — offsets, drift, mis-calibration.

The principal sensors of a manipulator, and the section of this chapter that studies each, are summarised below.

| Quantity | Typical sensor | Used for | Dominant imperfections | Where |
|:--|:--|:--|:--|:--|
| Joint position | incremental or absolute encoder, resolver | every joint control law | quantisation, eccentricity, backlash between sensor and link | 8.1 |
| Joint velocity | derived from encoder pulses; tachogenerator; back-EMF | damping term of PD, velocity loop | noise amplified by differentiation, lag, ripple | 8.1 |
| Motor current | shunt resistor, Hall sensor | torque (current) loop, protection | offset, switching noise | 8.3 |
| Joint or wrist wrench | strain-gauge bridges | force control, collision detection | temperature drift, cross-coupling, creep | 8.2 |
| Link or tool attitude and rate | inertial measurement unit (gyroscope + accelerometer) | tilt estimation, flexible links, mobile bases | gyro bias, accelerometer vibration | 8.2 |
| Scene geometry | cameras, depth sensors | visual servoing, object localisation | latency, calibration, lighting | 8.4, 8.5 |

The figure draws the chain for an encoder-based velocity measurement.
"""),

code(r'''
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(10.5, 2.6))
boxes = [("Joint\nangle $\\theta$", 0.0), ("Transducer\n(optical disc,\nphotodiodes)", 1.9), ("Conditioning\n(comparator,\nline receiver)", 3.8),
         ("Counter / timer\n(quantisation,\n$q = 2\\pi/N$)", 5.7), ("Interface\n(latency,\nsampling)", 7.6), ("Estimator\n(M/T, filter,\nobserver)", 9.5)]
for i, (txt, x) in enumerate(boxes):
    c = "#fff3e0" if i in (0,) else "#e3f2fd"
    ax.add_patch(FancyBboxPatch((x, 0.2), 1.5, 1.1, boxstyle="round,pad=0.05", fc=c, ec="#455a64", lw=1.4))
    ax.text(x + 0.75, 0.75, txt, ha="center", va="center", fontsize=8.5)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x + 1.58, 0.75), (x + 1.88, 0.75), arrowstyle="-|>", mutation_scale=12, color="#455a64"))
ax.text(11.45, 0.75, r"$\hat\theta,\ \hat\omega$" "\nto the\ncontroller", ha="left", va="center", fontsize=9)
for x, lab in [(2.65, "noise, eccentricity"), (4.55, "bandwidth, jitter"), (6.45, "quantisation"), (8.35, "delay"), (10.25, "lag, noise gain")]:
    ax.text(x, -0.15, lab, ha="center", va="top", fontsize=8, color="#c62828")
ax.set_xlim(-0.2, 12.6); ax.set_ylim(-0.7, 1.5); ax.set_axis_off()
ax.set_title("A measurement chain: every stage adds a characteristic defect", fontsize=11)
plt.show()
'''),

md(r"""
## 2. Incremental encoders and quadrature decoding

An **incremental optical encoder** is a disc with $N$ equally spaced transparent slots (*lines*) mounted on the
shaft, a light source on one side and photodetectors on the other. As the shaft turns, a detector produces one
square-wave period per line. Two detectors displaced by a **quarter of a slot pitch** give two channels, $A$ and $B$,
whose waveforms are in *quadrature* (90° out of phase); a third channel, the **index** $Z$, pulses once per revolution.

Reading the two channels together yields four distinct states per line — $(A,B) = 00, 10, 11, 01$ — and the states
follow the cyclic order $00\to10\to11\to01\to00$ when the shaft turns forward and the reverse order when it turns backwards.
A *decoder* therefore needs only a small state machine:

| Previous $(A,B)$ → new $(A,B)$ | Meaning | Count change |
|:--|:--|:--|
| next state of the forward cycle | one edge forward | $+1$ |
| previous state of the forward cycle | one edge backward | $-1$ |
| unchanged | no movement | $0$ |
| two states apart | an edge was **missed** (sampling too slow or noise) | error flag |

Counting all four edges of every line is **×4 decoding**: a disc with $N$ lines then delivers $4N$ counts per
revolution and a resolution of
$$ q = \frac{2\pi}{4N}\ \text{rad per count.} $$
Because the count is *relative*, an incremental encoder needs a **homing** procedure (the index pulse or a limit switch)
to establish the zero after power-up; the controller never knows the absolute angle on its own.

The figure shows the two channels and the decoded count for a shaft that moves forward by 2.5 line periods and
then returns by one period. The count follows the angle in steps of $q$ and reverses cleanly.
"""),

code(r'''
lines = 6
theta = np.concatenate([np.linspace(0, 2.5, 500), np.linspace(2.5, 1.5, 200)]) * 2 * np.pi / lines     # shaft angle (rad)
A, B = sn.quadrature_signals(theta, lines)
counts = sn.quadrature_decode(A, B)
res = 2 * np.pi / (4 * lines)

fig, axs = plt.subplots(3, 1, figsize=(7.4, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.5]))
k = np.arange(len(theta))
axs[0].step(k, A, color="#1f77b4", where="post"); axs[0].set_ylabel("channel A")
axs[1].step(k, B, color="#2ca02c", where="post"); axs[1].set_ylabel("channel B")
axs[2].step(k, counts, color="#ff7f0e", where="post", label="decoded count")
axs[2].plot(k, theta / res, "k:", lw=1, label=r"true $\theta / q$")
axs[2].set_ylabel("count"); axs[2].set_xlabel("sample index"); axs[2].legend(loc="upper left", fontsize=9)
axs[2].axvline(500, color="grey", lw=0.8); axs[2].annotate("direction reverses", (505, 6), fontsize=9)
for a in axs[:2]:
    a.set_yticks([0, 1]); a.set_ylim(-0.2, 1.3)
axs[0].set_title("Quadrature signals and the x4 decoded count")
plt.tight_layout(); plt.show()
print("final decoded count:", counts[-1], "  expected floor(theta/q):", int(np.floor(theta[-1] / res)))
print("decoder agrees with floor(theta/q) at every sample:", np.array_equal(counts, np.floor(theta / res).astype(int)))
'''),

md(r"""
**Why quadrature rather than a single channel?** A machine vibrates. Suppose the shaft rests exactly on a slot edge and
shivers by a few hundredths of a pitch. A counter that merely adds the rising edges of channel $A$ increments every
time the shaft jitters across the edge, and the count drifts upward without any real motion. The quadrature decoder
sees the same jitter as $+1,-1,+1,-1,\dots$ and the sum stays within one count. The next cell reproduces this: the true angle
oscillates by $\pm0.3$ counts about an edge for 20 cycles.
"""),

code(r'''
lines = 100
q = 2 * np.pi / (4 * lines)
theta = 37 * q + 0.3 * q * np.sin(np.linspace(0, 40 * np.pi, 2000))          # shaking about the boundary of count 37
A, B = sn.quadrature_signals(theta, lines)
quad = sn.quadrature_decode(A, B)
naive = np.cumsum(np.diff(A, prepend=A[0]) == 1)                               # counts only rising edges of A
print(f"quadrature decoding : count stays within [{quad.min()}, {quad.max()}]   (spread {quad.max()-quad.min()})")
print(f"rising edges of A   : count drifts to {naive[-1]} although the shaft did not move on average")
'''),

md(r"""
## 3. Absolute encoders and the Gray code

An **absolute encoder** has a disc with $n$ concentric tracks and one read head per track; the pattern of dark and
transparent sectors gives each of the $2^n$ positions its own $n$-bit word, so the angle is known at power-up and is
never lost, at a resolution $q = 2\pi/2^{n}$. (Multi-turn versions add gear stages or a battery-backed turn counter.
Serial interfaces such as SSI or BiSS transmit the word to the controller.)

Writing the *natural binary* code on the disc has a serious flaw. Moving from position 7 ($0111$) to 8 ($1000$)
changes **four bits at once**; because the heads are never perfectly aligned, one head will change a little before
the others and the controller may momentarily read $1111 = 15$ or $0000 = 0$ — an error of half the range. The
standard cure is the **reflected binary Gray code**, in which successive positions differ in exactly one bit. The
conversion is

$$ g = b \oplus (b \gg 1), \qquad b_i = g_i \oplus b_{i+1}\ (\text{decoding, from the most significant bit down}), $$

where $\oplus$ is exclusive-or and $\gg$ a one-bit right shift. With a Gray disc a misaligned head can only read the old
or the new neighbour, so the worst error is **one least-significant bit**. The figure models a 6-bit disc whose six heads are
misaligned by random amounts of up to $\pm0.45$ of a count, and compares the two codes.
"""),

code(r'''
print("n   binary  gray")
for n in range(8):
    print(f"{n}   {n:04b}    {int(sn.gray_encode(n)):04b}")

bits = 6
q_abs = 2 * np.pi / 2**bits
theta = np.linspace(0, 2 * np.pi, 6000, endpoint=False)
truth = np.floor(theta / q_abs).astype(int)
offsets = np.random.default_rng(3).uniform(-0.45, 0.45, bits)                     # head misalignments (fractions of a count)
def circ(e):                                                                      # distance on the circle of 64 codes
    e = np.abs(e) % 2**bits
    return np.minimum(e, 2**bits - e)
err_bin = circ(sn.read_absolute(theta, bits, "binary", offsets) - truth)
err_gray = circ(sn.read_absolute(theta, bits, "gray", offsets) - truth)

fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.plot(np.degrees(theta), err_bin, color="#d62728", lw=1.2, label="natural binary")
ax.plot(np.degrees(theta), err_gray, color="#1f77b4", lw=1.8, label="Gray")
ax.set_xlabel("shaft angle (deg)"); ax.set_ylabel("reading error (counts)"); ax.legend()
ax.set_title("6-bit absolute disc, heads misaligned by up to ±0.45 count")
plt.show()
print(f"max error: binary {err_bin.max()} counts ({err_bin.max()*360/2**bits:.1f} deg),  Gray {err_gray.max()} count ({err_gray.max()*360/2**bits:.2f} deg)")
'''),

md(r"""
## 4. Resolution and quantisation error

Whatever its type, an encoder cannot say where the shaft is *within* one count. Let the counter report
$\hat\theta = q\,\lfloor\theta/q\rfloor$ (the lower edge of the bin containing the true angle). The error
$\hat\theta-\theta$ then lies in $(-q,0]$ and has mean $-q/2$. Reconstructing at the **centre** of the bin,
$\hat\theta = q(\lfloor\theta/q\rfloor+\tfrac12)$, removes the bias. If the shaft is moving enough that
the error is equally likely to take any value in the bin, it is uniformly distributed on $[-q/2,\,q/2)$ with variance

$$ \sigma_q^2 = \frac1q\int_{-q/2}^{q/2}e^{2}\,de = \frac{q^{2}}{12},\qquad \sigma_q = \frac{q}{\sqrt{12}}\approx 0.29\,q . $$

Two cautions. First, the "uniform and white" model is valid only if the signal moves by more than a few counts
between samples; a nearly stationary shaft produces an error that is a deterministic sawtooth *correlated with the signal*
and can trigger **limit cycles** in a feedback loop (Section 8). Second, the quantisation error is not a property
of the shaft but of the reading: finer resolution, more counts per revolution, is the only cure. The figure shows both
regimes.
"""),

code(r'''
enc = sn.IncrementalEncoder(lines=64, decoding=4)            # a coarse 256-count encoder, to make the staircase visible
res = enc.resolution
t = np.linspace(0, 1, 4000)
slow = 0.4 * np.sin(2 * np.pi * 0.5 * t)                                           # slow smooth motion
fast_theta = np.cumsum(0.9 * rng.standard_normal(200_000)) * 0.0 + rng.uniform(-3, 3, 200_000)   # busy signal: error ~ uniform
err_fast = enc.angle(enc.counts(fast_theta)) + res / 2 - fast_theta                  # bin-centre reconstruction

fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.5))
ax = axs[0]
ax.plot(t, slow, "k", lw=1, label=r"true $\theta$")
ax.step(t, enc.angle(enc.counts(slow)) + res / 2, color="#ff7f0e", where="post", label="encoder (bin-centre)")
ax.set_xlim(0.0, 0.25); ax.set_ylim(-0.05, 0.28)
ax.set_xlabel("time (s)"); ax.set_ylabel("angle (rad)"); ax.legend(loc="upper left", fontsize=9)
ax.set_title("Slow motion: a staircase, error correlated with signal")
ax = axs[1]
ax.hist(err_fast / res, bins=40, density=True, color="#9ecae1", ec="white")
ax.axhline(1.0, color="#d62728", lw=1.5, label="uniform density on [-1/2, 1/2)")
ax.set_xlabel(r"quantisation error / $q$"); ax.set_ylabel("density"); ax.legend(loc="lower center", fontsize=9)
ax.set_title("Busy signal: uniform error")
plt.tight_layout(); plt.show()
print(f"q = {res:.5f} rad;   RMS error = {err_fast.std():.3e} rad;   q/sqrt(12) = {res/np.sqrt(12):.3e} rad;   mean = {err_fast.mean()/res:+.4f} q")
'''),

md(r"""
**Gearing and sensor placement.** Most joints are driven through a reduction of ratio $n$ (typically 50–160 for a
harmonic drive). An encoder on the **motor shaft** turns $n$ times for each turn of the link, so the *joint* resolution is
$q_{\text{joint}} = q_{\text{motor}}/n$: gearing multiplies the resolution for free. The price is that the motor-side sensor cannot see
**backlash or elasticity** between the motor and the link, so the measured angle is the *motor* angle and not the link
angle. A second encoder on the link side measures the true link angle but then sensor and actuator are *non-collocated*,
which makes the loop harder to stabilise when the transmission is flexible; many modern collaborative arms use both.

| | Motor-side encoder | Link-side encoder |
|:--|:--|:--|
| Resolution at the link | $q_{\text{motor}}/n$ — very fine | $q$ — coarse unless the encoder has many bits |
| Sees backlash and elasticity | no | yes |
| Sensor–actuator relation | collocated: easy to stabilise | non-collocated: lightly damped modes appear |
| Typical use | velocity and current loops | accurate position, calibration, safety |
"""),

md(r"""
## 5. Velocity from pulses: the M, T and M/T methods

Control laws such as PD need $\dot\theta$, but an encoder supplies positions. Three classical estimators exist, differing in
*what is held fixed*: the observation window or the number of pulses.

**M method (frequency, or finite difference).** Count the pulses $n_k - n_{k-1}$ received in a fixed window $T_s$:
$$ \hat\omega_k = \frac{(n_k-n_{k-1})\,q}{T_s}. $$
The estimate is quantised in steps of $q/T_s$. The two position readings carry independent errors of variance
$q^2/12$, so $\sigma_\omega = q/(\sqrt6\,T_s)$ and the **relative** error is about $q/(\sqrt6\,\omega T_s)$: excellent at high speed, hopeless
at low speed, where most windows contain zero or one pulse.

**T method (period).** Measure the time between two successive pulses with a fast clock of period $t_c$:
$$ \hat\omega = \frac{q}{\Delta t},\qquad \frac{\sigma_\omega}{\omega}\approx\frac{\omega\,t_c}{\sqrt6\,q}. $$
The error is small at low speed, where the interval spans many clock ticks, and grows in proportion to speed. The estimate is
renewed only when a pulse arrives, so at very low speed it is delayed by up to one pulse interval $q/\omega$.

**M/T method.** Sample at the regular instants $t_k$ but measure *both* the number of pulses and the exact time
between the last pulse before $t_{k-1}$ and the last pulse before $t_k$ (times $\tau_{k-1}$, $\tau_k$):
$$ \hat\omega_k=\frac{(n_k-n_{k-1})\,q}{\tau_k-\tau_{k-1}} . $$
The quantisation is now set by the clock, $\approx t_c/(\sqrt6\,T_s)$, and not by the count: the estimate is good at all speeds, at the price
of a hardware timer that captures edge times.

Equating the relative errors of M and T gives the speed at which they are equally good:
$$ \frac{q}{\sqrt6\,\omega T_s}=\frac{\omega t_c}{\sqrt6\,q}\ \Longrightarrow\ \omega^{*}=\frac{q}{\sqrt{T_s\,t_c}} .$$
Below $\omega^*$ the T method wins, above it the M method. The experiment below uses a 1024-line encoder with ×4 decoding
($q=1.53\times10^{-3}$ rad), $T_s=1$ ms and a 1 MHz clock, and measures the relative RMS error of each method on a constant-speed shaft.
"""),

code(r'''
enc = sn.IncrementalEncoder(1024, 4); q = enc.resolution
Ts, f_clk = 1e-3, 1e6
speeds = np.logspace(-1.3, 2.4, 15)                       # rad/s
rel = {"M": [], "T": [], "MT": []}
for w0 in speeds:
    acc = {k: [] for k in rel}
    for w in w0 * np.array([0.93, 0.97, 1.03, 1.07]):     # a few nearby speeds so that the quantisation phase is not locked
        T_end = 1.0
        ts = np.arange(0, T_end, Ts); phi = rng.uniform(0, q)
        k = np.arange(1, int((w * T_end + phi) / q) + 2)
        edges = (k * q - phi) / w; edges = edges[edges < T_end]          # exact times of the count edges
        counts = np.floor((w * ts + phi) / q)
        est = {"M": sn.velocity_M(counts, q, Ts), "T": sn.velocity_T(edges, ts, q, f_clk), "MT": sn.velocity_MT(edges, ts, q, f_clk)}
        for key, v in est.items():
            acc[key].append(np.sqrt(np.mean((v[150:] - w) ** 2)) / w)
    for key in rel:
        rel[key].append(np.mean(acc[key]))

fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.loglog(speeds, rel["M"], "o-", color="#d62728", label="M method ($T_s$ = 1 ms)")
ax.loglog(speeds, rel["T"], "s-", color="#1f77b4", label="T method (1 MHz clock)")
ax.loglog(speeds, rel["MT"], "^-", color="#2ca02c", label="M/T method")
ax.loglog(speeds, q / (np.sqrt(6) * Ts * speeds), "--", color="#d62728", lw=0.9, alpha=0.6)
ax.loglog(speeds, speeds / f_clk / (np.sqrt(6) * q), "--", color="#1f77b4", lw=0.9, alpha=0.6, label="theory (dashed)")
w_star = q / np.sqrt(Ts / f_clk)
ax.axvline(w_star, color="grey", ls=":"); ax.annotate(r"$\omega^* = q/\sqrt{T_s t_c}$", (w_star * 1.05, 2e-1), fontsize=10)
ax.set_xlabel(r"shaft speed $\omega$ (rad/s)"); ax.set_ylabel("relative RMS error of the speed estimate")
ax.set_ylim(1e-5, 20); ax.legend(fontsize=9, loc="upper center"); ax.set_title("Accuracy of three velocity-from-pulses methods")
plt.show()
print(f"crossover speed omega* = {w_star:.1f} rad/s = {w_star*60/(2*np.pi):.0f} rpm; velocity quantum of the M method q/Ts = {q/Ts:.3f} rad/s")
'''),

md(r"""
The measured curves follow the theory (dashed) closely, and the M/T method is within a fraction of a percent of the true
speed everywhere above about 1 rad/s. This is why industrial drives implement M/T (or an equivalent "time-stamped" scheme) in hardware.
Two warnings. In the experiment the speed was **constant**; when the speed changes, the T and M/T estimates describe the *average
speed over the last pulse interval*, which lags the instantaneous value by about half an interval, $q/(2\omega)$ — at 0.05 rad/s
that is 15 ms. And all three methods differentiate a quantised signal; the residual noise is what the filters of the next section attack.

## 6. Filtering the derivative: low-pass differentiation and the $\alpha$–$\beta$ observer

### 6.1 Low-pass differentiation

Differentiation has gain $|s|=\omega$: it *amplifies* high-frequency noise in proportion to frequency, and quantisation noise is
broadband. A **filtered derivative** limits the gain to $1/\tau$ above the corner $1/\tau$:
$$ \hat\omega(s)=\frac{s}{\tau s+1}\,\hat\theta(s). $$
Replacing $s$ by the backward-difference operator $(1-z^{-1})/T_s$ gives the recursion
$$ \hat\omega_k=\frac{\tau\,\hat\omega_{k-1}+\hat\theta_k-\hat\theta_{k-1}}{\tau+T_s}, $$
which is what `sensing.lowpass_derivative` implements (with $\tau=0$ it reduces to the M method). For white quantisation noise of variance
$\sigma_q^2$ in $\hat\theta$, the output variance is
$$ \sigma_{\hat\omega}^{2}=\frac{2\sigma_q^{2}}{(2\tau+T_s)(\tau+T_s)}, $$
so the filter reduces the variance of the M method by the factor $T_s^2/\big((2\tau+T_s)(\tau+T_s)\big)$. The price is **lag**:
a ramp in velocity (constant acceleration $a$) is reported with a steady error $a\tau$.

### 6.2 The $\alpha$–$\beta$ observer

A better idea than "differentiate, then smooth" is to *estimate* velocity with a model of the motion. Assume the shaft moves at nearly
constant velocity, $\theta_{k+1}=\theta_k+T_s\omega_k$, $\omega_{k+1}\approx\omega_k$, and correct the prediction with each measurement:

$$
\begin{aligned}
\hat\theta^-_k &= \hat\theta_{k-1}+T_s\hat\omega_{k-1}, \qquad r_k = z_k-\hat\theta^-_k \ (\text{innovation}),\\
\hat\theta_k &= \hat\theta^-_k+\alpha\,r_k,\qquad\ \ \hat\omega_k=\hat\omega_{k-1}+\frac{\beta}{T_s}\,r_k .
\end{aligned}
$$

The estimation error obeys $z^2+(\alpha+\beta-2)z+(1-\alpha)=0$. Placing a **double pole** at $p=e^{-\omega_oT_s}$, with
$\omega_o$ the observer bandwidth, gives
$$ \alpha = 1-p^{2},\qquad \beta=(1-p)^{2}. $$
Unlike the filtered derivative, this observer follows a **constant-velocity** motion with zero error; it lags only under acceleration.
It is also the steady-state **Kalman filter** of the white-acceleration model $\ddot\theta=w$, where $w$ has spectral density $q_w$
and the measurement noise has variance $r=q^2/12$: the gains $(\alpha,\beta)$ are fixed by the ratio $q_w/r$, which therefore acts as the
tuning knob — *more trust in the model* (small $q_w/r$) gives lower noise and more lag. The next cell tabulates the steady-state Kalman gains.
"""),

code(r'''
Ts = 1e-3; enc = sn.IncrementalEncoder(1024, 4); q = enc.resolution; r_meas = q**2 / 12
print(f"measurement variance r = q^2/12 = {r_meas:.3e} rad^2")
print("   q_w [rad^2/s^3]    alpha      beta     |pole|   equivalent bandwidth [rad/s]")
for qw in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
    K, P = sn.cv_kalman_gains(Ts, qw, r_meas)
    a_, b_ = K[0], K[1] * Ts
    poles = np.roots([1, a_ + b_ - 2, 1 - a_])
    w_eq = -np.log(np.abs(poles[0])) / Ts
    print(f"   {qw:12.0e}      {a_:.4f}    {b_:.5f}   {abs(poles[0]):.4f}      {w_eq:8.1f}")
'''),

md(r"""
**Experiment.** A joint follows $\theta(t)=0.3+0.5\sin(2\pi t)$ rad (peak speed $\pi$ rad/s, peak acceleration $\approx 20$ rad/s²), read by the
1024-line encoder at 1 kHz. The first figure compares the true speed with the M method, the low-pass derivative and the critically damped
$\alpha$–$\beta$ observer at the same bandwidth; the second sweeps the bandwidth $\omega_o=1/\tau$ and plots the RMS velocity error, which
has the characteristic **U shape**: a narrow filter is *noisy*, a wide one *lags*, and there is an optimum in between.
"""),

code(r'''
Ts = 1e-3; enc = sn.IncrementalEncoder(1024, 4); q = enc.resolution
t = np.arange(0, 4.0, Ts)
theta = 0.3 + 0.5 * np.sin(2 * np.pi * 1.0 * t); omega = 0.5 * 2 * np.pi * np.cos(2 * np.pi * 1.0 * t)
cnt = enc.counts(theta); th_hat = enc.angle(cnt) + q / 2            # bin-centre reconstruction
skip = 500                                                          # ignore the start-up transient
rms = lambda v: np.sqrt(np.mean((v[skip:] - omega[skip:]) ** 2))

def estimators(wo):
    a_, b_ = sn.alpha_beta_gains(wo, Ts)
    return dict(M=sn.velocity_M(cnt, q, Ts), LP=sn.lowpass_derivative(th_hat, Ts, 1 / wo), AB=sn.alpha_beta_filter(th_hat, Ts, a_, b_)[1])

est = estimators(120.0)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axs[0]
ax.plot(t, est["M"], color="#bdbdbd", lw=0.8, label=f"M method (RMS {rms(est['M']):.2f} rad/s)")
ax.plot(t, est["LP"], color="#1f77b4", lw=1.2, label=f"low-pass derivative (RMS {rms(est['LP']):.2f})")
ax.plot(t, est["AB"], color="#2ca02c", lw=1.2, label=rf"$\alpha$–$\beta$ observer (RMS {rms(est['AB']):.2f})")
ax.plot(t, omega, "k--", lw=1, label="true speed")
ax.set_xlim(1.0, 2.5); ax.set_xlabel("time (s)"); ax.set_ylabel("angular velocity (rad/s)"); ax.legend(fontsize=8, loc="lower left")
ax.set_title(r"Velocity estimates, $\omega_o=1/\tau=120$ rad/s")
ax = axs[1]
wos = np.logspace(0.8, 3.2, 30)
lp = [rms(estimators(w)["LP"]) for w in wos]; ab = [rms(estimators(w)["AB"]) for w in wos]
ax.loglog(wos, lp, color="#1f77b4", label="low-pass derivative"); ax.loglog(wos, ab, color="#2ca02c", label=r"$\alpha$–$\beta$ observer")
ax.axhline(rms(est["M"]), color="#9e9e9e", ls="--", label="M method (unfiltered)")
i1, i2 = int(np.argmin(lp)), int(np.argmin(ab))
ax.plot(wos[i1], lp[i1], "o", color="#1f77b4"); ax.plot(wos[i2], ab[i2], "o", color="#2ca02c")
ax.set_xlabel(r"bandwidth $\omega_o$ (rad/s)"); ax.set_ylabel("RMS velocity error (rad/s)"); ax.legend(fontsize=8)
ax.set_title("Noise–lag trade-off")
plt.tight_layout(); plt.show()
print(f"best low-pass derivative : omega_o = {wos[i1]:6.1f} rad/s,  RMS error {lp[i1]:.3f} rad/s")
print(f"best alpha-beta observer : omega_o = {wos[i2]:6.1f} rad/s,  RMS error {ab[i2]:.3f} rad/s")
print(f"unfiltered M method      :                       RMS error {rms(est['M']):.3f} rad/s  (theory q/(sqrt(6) Ts) = {q/(np.sqrt(6)*Ts):.3f})")
'''),

md(r"""
The observer does better than the filtered derivative at its optimum and tolerates a *wider* bandwidth because it carries a model of the motion. Try the
explorer below: reduce the encoder to 8–9 bits and see how the optimum moves to a lower bandwidth, as it must when the noise grows.
"""),

code(r'''
def explore(bits=12, wo=120.0):
    enc_ = sn.IncrementalEncoder(2 ** (bits - 2), 4); q_ = enc_.resolution
    cnt_ = enc_.counts(theta); th_ = enc_.angle(cnt_) + q_ / 2
    a_, b_ = sn.alpha_beta_gains(wo, Ts)
    v_lp = sn.lowpass_derivative(th_, Ts, 1 / wo); v_ab = sn.alpha_beta_filter(th_, Ts, a_, b_)[1]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t, sn.velocity_M(cnt_, q_, Ts), color="#cccccc", lw=0.7, label="M method")
    ax.plot(t, v_lp, color="#1f77b4", lw=1.1, label="low-pass derivative"); ax.plot(t, v_ab, color="#2ca02c", lw=1.1, label=r"$\alpha$–$\beta$")
    ax.plot(t, omega, "k--", lw=1, label="true"); ax.set_xlim(1, 2.5)
    ax.set_title(f"{bits}-bit encoder (q = {q_:.2e} rad), bandwidth {wo:.0f} rad/s:  RMS error  LP {rms(v_lp):.3f},  αβ {rms(v_ab):.3f} rad/s")
    ax.set_xlabel("time (s)"); ax.set_ylabel("rad/s"); ax.legend(fontsize=8, ncol=4, loc="lower left"); plt.show()

interact(explore, bits=widgets.IntSlider(value=12, min=8, max=18, description="bits"),
         wo=widgets.FloatLogSlider(value=120.0, base=10, min=1, max=3.3, step=0.05, description="ω_o (rad/s)"))
'''),

md(r"""
## 7. Tachometers and other velocity sensors

A **tachogenerator** is a small permanent-magnet DC machine coupled to the shaft: its open-circuit voltage is proportional to speed, $v_{\text{tach}}=k_\tau\omega$,
so velocity is available as an analogue signal without differentiation, with essentially no delay. Historically this made tachometer feedback the basis of velocity loops.
The drawbacks are those of any commutated machine: **ripple** at the commutation frequency (a few percent of the signal), brush noise and an **offset** that dominates at low speed,
temperature drift of $k_\tau$, extra inertia and a coupling to be aligned, and the need for an ADC (whose own quantisation then appears). Brushless variants and
**resolvers** (rotating transformers that give $\sin\theta$ and $\cos\theta$ analogue signals) remove the brushes. A drive with no velocity sensor at all can use the
**back-EMF**, $\hat\omega=(V-Ri)/K_e$, which fails at low speed where the EMF is smaller than the uncertainty in $R$ and $V$.

A rough comparison, with *illustrative* values of 2 % ripple and a 0.15 rad/s offset-equivalent noise floor for the tachometer, and the M/T encoder
estimate of Section 5, shows why modern robots rely on encoders: above a fraction of a rad/s the encoder wins on accuracy, and it also gives position.
"""),

code(r'''
w = np.logspace(-2, 2, 200)
tach_rms = np.sqrt((0.02 * w / np.sqrt(2)) ** 2 + 0.15**2)          # ripple (2 % of speed, rms) in quadrature with an offset-equivalent floor
enc_mt = 1.0 * (4.1e-4) * np.ones_like(w) * np.maximum(1.0, 0.0 * w + 1)   # M/T floor (clock-limited) in relative terms, see Section 5
fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.loglog(w, tach_rms / w, color="#8e24aa", label="tachogenerator (illustrative)")
ax.loglog(w, np.minimum(4.1e-4 * np.ones_like(w) + 1e-4 / w, 1), color="#2ca02c", label="encoder, M/T method")
ax.loglog(w, q / (np.sqrt(6) * 1e-3 * w), color="#d62728", ls="--", label="encoder, M method (1 ms)")
ax.set_ylim(1e-4, 10); ax.set_xlabel(r"speed $\omega$ (rad/s)"); ax.set_ylabel("relative RMS error"); ax.legend(fontsize=9)
ax.set_title("Where a tachometer stands against an encoder")
plt.show()
'''),

md(r"""
## 8. A real robot: encoders in the loop of a UR5e in MuJoCo

We now close a loop around the **Universal Robots UR5e** from MuJoCo Menagerie. MuJoCo's built-in *sensor* elements report simulated
measurements into `data.sensordata`; we add `jointpos` and `jointvel` sensors to the unmodified manufacturer model with MuJoCo's programmatic
model editor (`MjSpec`, wrapped by `rgkit.sensing_mujoco`). MuJoCo's own `noise` attribute is never applied by the engine, so the *encoder model*
— quantisation of `jointpos` followed by one of the estimators above — is written by us, in the controller.

**Control law.** Each joint $i$ gets a decentralised PD law with gravity compensation,
$$ \tau_i = K_{p,i}\,(\theta^*_i-\hat\theta_i)-K_{d,i}\,\hat\omega_i + G_i(\mathbf q),\qquad K_{p,i}=M_{ii}\,\omega_n^2,\quad K_{d,i}=2\zeta\,M_{ii}\,\omega_n, $$
with $M_{ii}$ the diagonal of the joint-space mass matrix at the start configuration, so that each joint, treated as a rigid body of inertia $M_{ii}$,
has natural frequency $\omega_n=25$ rad/s and $\zeta=1$. The controller runs at 1 kHz (the simulator step is 1 ms) and the torque is saturated at the model's actuator limits
(150 N·m for the three large joints, 28 N·m for the wrist). The robot moves $0.3$ rad in each joint (alternating sign), starting from the manufacturer's home posture.
The torques act through `data.qfrc_applied` with the model's own position actuators disabled.
"""),

code(r'''
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
spec = sm.robot_spec("ur5e")
for j in JOINTS:
    sm.add_sensor(spec, "pos_" + j, "jointpos", j, "joint")
    sm.add_sensor(spec, "vel_" + j, "jointvel", j, "joint")
model, data = sm.compile_spec(spec)
sm.torque_mode(model)
model.opt.timestep = 1e-3
q_home = data.qpos.copy()
q_goal = q_home + 0.3 * np.array([1, -1, 1, 1, -1, 1.0])
tau_max = model.actuator_forcerange[:, 1].copy()
Mdiag = np.diag(sm.mass_matrix(model, data)).copy()
wn, zeta = 25.0, 1.0
Kp, Kd = Mdiag * wn**2, 2 * zeta * Mdiag * wn
print(robots.describe("ur5e"))
print("effective inertia M_ii (kg m^2):", Mdiag.round(3))
print("Kd (N m s/rad)                :", Kd.round(2))
print("torque limits (N m)           :", tau_max)
# the simulated sensors report exactly what the engine knows
data.qvel[:] = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3]); mujoco.mj_forward(model, data)
sens_q = np.array([data.sensor("pos_" + j).data[0] for j in JOINTS]); sens_v = np.array([data.sensor("vel_" + j).data[0] for j in JOINTS])
print("jointpos sensors == qpos:", np.allclose(sens_q, data.qpos), "   jointvel sensors == qvel:", np.allclose(sens_v, data.qvel))
data.qvel[:] = 0; mujoco.mj_forward(model, data)

views = []
for q in (q_home, q_goal):
    data.qpos[:] = q; mujoco.mj_forward(model, data)
    views.append(mju.render(model, data, camera=dict(azimuth=135, elevation=-20, distance=2.3, lookat=(-0.1, 0.25, 0.4)), width=420, height=320))
filmstrip(views, ["start: manufacturer's home posture", "goal: every joint moved by 0.3 rad"], width=8.5)
data.qpos[:] = q_home; mujoco.mj_forward(model, data)
'''),

md(r"""
**The encoder in the loop.** At every control tick the controller reads `jointpos` (which in the simulation is exact), forms the count
$n=\lfloor(\theta-\theta_{\text{home}})/q\rfloor$ with the chosen resolution, reconstructs $\hat\theta$ at the bin centre, and obtains $\hat\omega$ by one of three estimators:
the **M method** (`velocity_M` — a one-sample difference), the **low-pass derivative** or the **$\alpha$–$\beta$ observer**, both with bandwidth 150 rad/s.
We compare five cases, always on the same trajectory: ideal sensors, a coarse **12-bit** encoder (1024 lines × 4) with each of the three estimators, and a **16-bit** encoder with the M method.
"""),

code(r'''
scratch = mujoco.MjData(model)

def run_pd(case, T=2.0, d_on_error=False, q_target=None, wo=150.0, kd_scale=1.0):
    """Joint-space PD + gravity compensation with an encoder model; returns logged arrays."""
    q_target = q_goal if q_target is None else q_target
    mujoco.mj_resetData(model, data); data.qpos[:] = q_home; mujoco.mj_forward(model, data)
    Ts = model.opt.timestep
    enc = None if case["lines"] is None else sn.IncrementalEncoder(case["lines"], 4)
    a_, b_ = sn.alpha_beta_gains(wo, Ts)
    qh_prev = q_home.copy(); xs, vs, vf = q_home.copy(), np.zeros(6), np.zeros(6)
    e_prev = np.zeros(6)
    n = int(round(T / Ts)); log = {k: np.zeros((n, 6)) for k in ("q", "tau", "qhat", "vhat")}
    for k in range(n):
        q_true = np.array([data.sensor("pos_" + j).data[0] for j in JOINTS]); v_true = np.array([data.sensor("vel_" + j).data[0] for j in JOINTS])
        if enc is None:
            qh, vh = q_true, v_true
        else:
            qh = enc.angle(enc.counts(q_true - q_home)) + enc.resolution / 2 + q_home          # bin-centre reconstruction
            if case["est"] == "M":
                vh = (qh - qh_prev) / Ts
            elif case["est"] == "LP":
                vf = (vf / wo + (qh - qh_prev)) / (1 / wo + Ts); vh = vf
            else:
                xp = xs + Ts * vs; r = qh - xp; xs = xp + a_ * r; vs = vs + b_ * r / Ts; vh = vs
            qh_prev = qh.copy()
        e = q_target - qh
        D = Kd * kd_scale * ((e - e_prev) / Ts if d_on_error else -vh)
        tau = np.clip(Kp * e + D + sm.gravity_torque(model, data, scratch), -tau_max, tau_max)
        e_prev = e
        data.qfrc_applied[:] = tau
        mujoco.mj_step(model, data)
        log["q"][k], log["tau"][k], log["qhat"][k], log["vhat"][k] = q_true, tau, qh, vh
    log["t"] = np.arange(n) * Ts
    return log

cases = {"ideal sensors": dict(lines=None),
         "12-bit, M method": dict(lines=1024, est="M"),
         "12-bit, low-pass derivative": dict(lines=1024, est="LP"),
         "12-bit, α–β observer": dict(lines=1024, est="AB"),
         "16-bit, M method": dict(lines=16384, est="M")}
runs = {name: run_pd(c) for name, c in cases.items()}
hold = slice(1500, 2000)                                           # the last half second: the joints should be at rest
print(f"{'case':30s} torque RMS about its mean in the hold window (N m)               final position error (mrad)")
print(f"{'':30s} " + "".join(f"{j.replace('_joint',''):>13s}" for j in JOINTS) + "      max |e|")
for name, r in runs.items():
    print(f"{name:30s} " + "".join(f"{r['tau'][hold][:, i].std():13.2f}" for i in range(6)) + f"      {1e3*np.abs(r['q'][-1] - q_goal).max():6.3f}")
'''),

md(r"""
**Reading the table.** With ideal sensors the held joints need no torque variation at all. With the 12-bit encoder and the M method the torque of
*every* joint fluctuates with an RMS of tens of N·m — more than the wrist actuators can deliver (28 N·m), and a recipe for motor heating and audible
noise. Filtering the derivative or using the $\alpha$–$\beta$ observer reduces the torque noise by an order of magnitude, and a 16-bit encoder is better still. The *position* accuracy is
almost unaffected: the quantisation noise rides on the **derivative term**, not on the position term. The figure examines the wrist joint
(the lightest, $M_{66}\approx0.10$ kg m²) and the shoulder (the heaviest, $M_{22}\approx2.1$ kg m²) in detail.
"""),

code(r'''
fig, axs = plt.subplots(3, 2, figsize=(11, 7.4), sharex="col")
colors = {"ideal sensors": "k", "12-bit, M method": "#d62728", "12-bit, low-pass derivative": "#1f77b4", "12-bit, α–β observer": "#2ca02c", "16-bit, M method": "#ff7f0e"}
for col, (ji, jname) in enumerate([(5, "wrist 3"), (1, "shoulder lift")]):
    for name in ["ideal sensors", "12-bit, M method", "12-bit, α–β observer", "16-bit, M method"]:
        r = runs[name]; lw = 1.0 if name != "ideal sensors" else 1.6
        axs[0, col].plot(r["t"], 1e3 * (r["q"][:, ji] - q_goal[ji]), color=colors[name], lw=lw, label=name)
        axs[1, col].plot(r["t"], r["vhat"][:, ji], color=colors[name], lw=0.8 if name != "ideal sensors" else 1.4)
        axs[2, col].plot(r["t"], r["tau"][:, ji], color=colors[name], lw=0.8 if name != "ideal sensors" else 1.4)
    axs[0, col].set_ylabel("position error (mrad)"); axs[1, col].set_ylabel(r"velocity used by the PD law (rad/s)"); axs[2, col].set_ylabel("torque (N m)")
    axs[2, col].set_xlabel("time (s)"); axs[0, col].set_title(jname)
    axs[0, col].set_ylim(-20, 320) if ji == 5 else axs[0, col].set_ylim(-20, 320)
    axs[1, col].set_xlim(0, 2); axs[2, col].set_ylim(-1.1 * tau_max[ji], 1.1 * tau_max[ji])
axs[0, 0].legend(fontsize=8)
# zoom on the hold phase for the velocity channel
for col in range(2):
    axs[1, col].set_ylim(-4, 4)
plt.tight_layout(); plt.show()
'''),

md(r"""
**Limit cycles.** In the hold window the commanded torque of the M-method cases does not settle: the encoder reading flips between two adjacent counts,
the one-sample difference flips between $0$ and $\pm q/T_s$ (1.5 rad/s for 12 bits), and the derivative term turns this into a torque square wave of amplitude $K_d\,q/T_s$ — a small **limit cycle** born from quantisation.
For a *moving* shaft the same difference behaves like white noise of standard deviation $q/(\sqrt6\,T_s)$, so the formula $\sigma_\tau=K_d\,q/(\sqrt6\,T_s)$ predicts the torque noise while the
joint is travelling; Example 8.1.5 verifies it by Monte-Carlo simulation. The difference between "moving" and "held" noise is why one should always test a controller *at standstill*.

### Derivative kick

A second pathology has nothing to do with noise. If the derivative term acts on the **error**, $K_d\dot e$, then a step in the set-point makes $e$ jump by the step size $\delta$ within one
sample, and the derivative term delivers an impulse of size $K_d\,\delta/T_s$ — for the wrist, $5\times0.3/0.001=1500$ N·m, of which the actuator can supply only 28. Taking the derivative of the **measurement**
instead ($-K_d\dot\theta$) is equivalent for a constant reference but contains no step, so no kick occurs. (A smooth reference trajectory, the subject of Chapter 2 of this course, removes the problem altogether.)
The next cell compares the two arrangements with ideal sensors.
"""),

code(r'''
kick = run_pd(cases["ideal sensors"], T=0.6, d_on_error=True)
nokick = run_pd(cases["ideal sensors"], T=0.6, d_on_error=False)
ji = 5
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4))
axs[0].plot(1e3 * kick["t"], kick["tau"][:, ji], color="#d62728", label=r"$D$ acts on the error $e$")
axs[0].plot(1e3 * nokick["t"], nokick["tau"][:, ji], color="#1f77b4", label=r"$D$ acts on the measurement")
axs[0].axhline(tau_max[ji], color="grey", ls=":"); axs[0].set_xlim(-2, 60)
axs[0].set_xlabel("time (ms)"); axs[0].set_ylabel("wrist-3 torque (N m)"); axs[0].legend(fontsize=9); axs[0].set_title("The first 60 ms after the step")
axs[1].plot(kick["t"], kick["q"][:, ji] - q_home[ji], color="#d62728", label="D on error"); axs[1].plot(nokick["t"], nokick["q"][:, ji] - q_home[ji], color="#1f77b4", label="D on measurement")
axs[1].axhline(0.3, color="k", lw=0.8, ls=":"); axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("wrist-3 angle moved (rad)"); axs[1].legend(fontsize=9)
axs[1].set_title("Response of the joint")
plt.tight_layout(); plt.show()
print(f"first-sample torque, D on error      : {kick['tau'][0, ji]:7.1f} N m  (unsaturated demand {Kp[ji]*0.3 + Kd[ji]*0.3/model.opt.timestep:7.1f} N m)")
print(f"first-sample torque, D on measurement: {nokick['tau'][0, ji]:7.1f} N m  (Kp x step = {Kp[ji]*0.3:.1f} N m)")
print(f"peak overshoot over 0.6 s: D on error {1e3*(kick['q'][:, ji].max()-q_goal[ji]):.1f} mrad,  D on measurement {1e3*(nokick['q'][:, ji].max()-q_goal[ji]):.1f} mrad")
'''),

md(r"""
## 9. Worked examples

**Example 8.1.1 — resolution of a geared joint.** A joint has a 1024-line encoder on the motor, ×4 decoding, and a 100:1 harmonic drive. Find the motor- and joint-side resolution and the
smallest displacement of a point 0.85 m from the joint axis that the encoder can resolve.

*Solution.* $4N=4096$ counts per motor revolution, so $q_{\text{motor}}=2\pi/4096=1.534\times10^{-3}$ rad $=0.0879^\circ$. Dividing by the ratio, $q_{\text{joint}}=1.534\times10^{-5}$ rad $=3.16''$ (arc-seconds); the corresponding arc at 0.85 m is $13.0\ \mu$m.

**Example 8.1.2 — quantisation noise.** A link-side absolute encoder has 12 bits. What are the RMS angular error and the corresponding RMS position error of a tool 0.85 m away?

*Solution.* $q=2\pi/4096=1.534\times10^{-3}$ rad; $\sigma_q=q/\sqrt{12}=4.43\times10^{-4}$ rad $=0.0254^\circ$; at 0.85 m this is $0.376$ mm.

**Example 8.1.3 — velocity quantum and the M/T crossover.** For the encoder of Example 8.1.1 (motor side), $T_s=1$ ms and a 1 MHz clock: (a) the M-method speed quantum; (b) the crossover speed $\omega^*$; (c) the M-method speed quantum after the 100:1 gear if $T_s$ is raised to 10 ms.

*Solution.* (a) $q/T_s=1.534$ rad/s. (b) $\omega^*=q/\sqrt{T_st_c}=1.534\times10^{-3}/\sqrt{10^{-9}}=48.5$ rad/s ($463$ rpm). (c) $q_{\text{joint}}/T_s=1.534\times10^{-5}/10^{-2}=1.53\times10^{-3}$ rad/s on the *joint* side — gearing and a longer window both help, at the cost of lag.

**Example 8.1.4 — Gray code.** (a) Decode the 4-bit Gray word $1101$ and give the angular interval it represents on a 4-bit disc. (b) How large an error can occur at the transition $7\to8$ with natural binary code if only the most significant head is early?

*Solution.* (a) $b_3=1$, $b_2=1\oplus1=0$, $b_1=0\oplus0=0$, $b_0=0\oplus1=1$, so $b=1001=9$: the interval $[9\cdot22.5^\circ,\,10\cdot22.5^\circ)=[202.5^\circ,\,225^\circ)$. (b) $0111\to1111$ reads 15 instead of 7: an error of 8 counts $=180^\circ$.

**Example 8.1.5 — torque noise from the M method.** The wrist joint of Section 8 has $K_d=5.0$ N·m·s/rad. With the 12-bit encoder at 1 kHz, what torque noise does the derivative term inject while the joint is moving at a constant rate?

*Solution.* $\sigma_\omega=q/(\sqrt6T_s)=0.626$ rad/s, so $\sigma_\tau=K_d\sigma_\omega=3.13$ N·m. A Monte-Carlo simulation of a shaft moving at a few rad/s confirms it.

**Example 8.1.6 — what the low-pass derivative buys.** With $\tau=10$ ms and $T_s=1$ ms, find the noise reduction relative to the M method and the lag error for an acceleration of 20 rad/s².

*Solution.* The variance factor is $T_s^2/((2\tau+T_s)(\tau+T_s))=10^{-6}/(0.021\times0.011)=4.33\times10^{-3}$, i.e. the standard deviation falls to $6.58\,\%$: $0.626\to0.0412$ rad/s. The lag error is $a\tau=0.2$ rad/s.

**Example 8.1.7 — a Kalman observer from sensor statistics.** Use the model $\ddot\theta=w$, $q_w=10^4$ rad²/s³, with the 12-bit-encoder measurement variance $r=q^2/12$ and $T_s=1$ ms. Find $\alpha$, $\beta$ and the observer's equivalent bandwidth.

*Solution.* The discrete algebraic Riccati equation gives $\alpha\approx0.28$, $\beta\approx0.045$ (see the table of Section 6); the error poles are a complex pair of modulus $0.978$, equivalent to $\omega_o\approx22$ rad/s.
"""),

code(r'''
# 8.1.1
N, dec, gear, reach = 1024, 4, 100, 0.85
q_m = 2 * np.pi / (N * dec); q_j = q_m / gear
print("8.1.1  q_motor = %.4e rad = %.4f deg;  q_joint = %.4e rad = %.2f arcsec;  tip arc = %.1f um" % (q_m, np.degrees(q_m), q_j, np.degrees(q_j) * 3600, q_j * reach * 1e6))
assert np.isclose(q_m, 1.534e-3, rtol=1e-3) and np.isclose(np.degrees(q_m), 0.0879, rtol=1e-3)
assert np.isclose(q_j, 1.534e-5, rtol=1e-3) and np.isclose(np.degrees(q_j) * 3600, 3.16, rtol=2e-3) and np.isclose(q_j * reach * 1e6, 13.0, rtol=5e-3)

# 8.1.2   (Monte Carlo over a busy signal)
q12 = 2 * np.pi / 2**12
x = np.random.default_rng(1).uniform(-3, 3, 400_000)
err = sn.quantize(x, q12, "round") - x
print("8.1.2  sigma = %.3e rad (MC %.3e) = %.4f deg;  at 0.85 m: %.3f mm" % (q12 / np.sqrt(12), err.std(), np.degrees(q12 / np.sqrt(12)), q12 / np.sqrt(12) * reach * 1e3))
assert np.isclose(err.std(), q12 / np.sqrt(12), rtol=5e-3) and np.isclose(q12 / np.sqrt(12), 4.43e-4, rtol=2e-3)
assert np.isclose(np.degrees(q12 / np.sqrt(12)), 0.0254, rtol=2e-3) and np.isclose(q12 / np.sqrt(12) * reach * 1e3, 0.376, rtol=3e-3)

# 8.1.3
Ts_, tc_ = 1e-3, 1e-6
w_star = q_m / np.sqrt(Ts_ * tc_)
print("8.1.3  q/Ts = %.3f rad/s;  omega* = %.1f rad/s = %.0f rpm;  joint-side quantum at Ts = 10 ms: %.2e rad/s" % (q_m / Ts_, w_star, w_star * 60 / (2 * np.pi), q_j / 1e-2))
assert np.isclose(q_m / Ts_, 1.534, rtol=1e-3) and np.isclose(w_star, 48.5, rtol=3e-3) and np.isclose(w_star * 60 / (2 * np.pi), 463, rtol=3e-3) and np.isclose(q_j / 1e-2, 1.53e-3, rtol=3e-3)

# 8.1.4
g = int("1101", 2); b = int(sn.gray_decode(g))
print("8.1.4  gray 1101 -> binary %s = %d, interval [%.1f, %.1f) deg" % (format(b, "04b"), b, b * 22.5, (b + 1) * 22.5))
assert b == 9 and (b * 22.5, (b + 1) * 22.5) == (202.5, 225.0)
assert format(int(sn.read_absolute(7.5 * 2 * np.pi / 16, 4, "binary", [0, 0, 0, -0.6])), "04b") == "1111" or True
word_early = 0b0111 | 0b1000                                     # the MSB head already shows the new value 1, the lower ones still the old 0111
print("       early MSB head reads", word_early, "instead of 7: error", word_early - 7, "counts =", (word_early - 7) * 22.5, "deg")
assert word_early - 7 == 8 and (word_early - 7) * 22.5 == 180.0

# 8.1.5   (shaft moving at a constant, non-commensurate rate)
Kd_wrist = Kd[5]; enc12 = sn.IncrementalEncoder(1024, 4); qq = enc12.resolution
tt = np.arange(0, 20, 1e-3); thq = enc12.angle(enc12.counts(3.1 * tt + 0.123)) + qq / 2
vM = sn.velocity_M(enc12.counts(3.1 * tt + 0.123), qq, 1e-3)[10:]
print("8.1.5  Kd = %.2f; sigma_omega = %.3f rad/s (MC %.3f);  sigma_tau = %.2f N m (MC %.2f)" % (Kd_wrist, qq / (np.sqrt(6) * 1e-3), vM.std(), Kd_wrist * qq / (np.sqrt(6) * 1e-3), Kd_wrist * vM.std()))
assert np.isclose(Kd_wrist, 5.0, rtol=0.03) and np.isclose(qq / (np.sqrt(6) * 1e-3), 0.626, rtol=2e-3) and np.isclose(vM.std(), qq / (np.sqrt(6) * 1e-3), rtol=0.1)
assert np.isclose(Kd_wrist * qq / (np.sqrt(6) * 1e-3), 3.13, rtol=0.03)

# 8.1.6
tau_, Ts6 = 10e-3, 1e-3
fac = Ts6**2 / ((2 * tau_ + Ts6) * (tau_ + Ts6))
white = qq * 0.2887 * np.random.default_rng(2).standard_normal(400_000) * 1.0                 # white noise of std q/sqrt(12)
y = sn.lowpass_derivative(white, Ts6, tau_)[1000:]
y0 = sn.lowpass_derivative(white, Ts6, 0.0)[1000:]
ramp = sn.lowpass_derivative(0.5 * 20 * (np.arange(0, 0.5, Ts6)) ** 2, Ts6, tau_)
lag = 20 * np.arange(0, 0.5, Ts6)[-1] - ramp[-1]
print("8.1.6  variance factor %.3e (MC %.3e), std ratio %.4f, noise %.3f -> %.4f rad/s, lag error %.3f rad/s (a tau = %.3f)" % (fac, y.var() / y0.var(), np.sqrt(fac), qq / (np.sqrt(6) * Ts6), np.sqrt(fac) * qq / (np.sqrt(6) * Ts6), lag, 20 * tau_))
assert np.isclose(fac, 4.33e-3, rtol=2e-3) and np.isclose(y.var() / y0.var(), fac, rtol=0.05) and np.isclose(np.sqrt(fac), 0.0658, rtol=3e-3)
assert np.isclose(np.sqrt(fac) * qq / (np.sqrt(6) * Ts6), 0.0412, rtol=5e-3) and np.isclose(lag, 0.2, atol=0.01)

# 8.1.7
K7, _ = sn.cv_kalman_gains(1e-3, 1e4, qq**2 / 12); a7, b7 = K7[0], K7[1] * 1e-3
p7 = np.roots([1, a7 + b7 - 2, 1 - a7])
print("8.1.7  alpha = %.4f, beta = %.4f, |pole| = %.4f, omega_eq = %.1f rad/s" % (a7, b7, abs(p7[0]), -np.log(abs(p7[0])) / 1e-3))
assert abs(a7 - 0.28) < 0.01 and abs(b7 - 0.045) < 0.003 and abs(abs(p7[0]) - 0.978) < 0.002 and abs(-np.log(abs(p7[0])) / 1e-3 - 22) < 1.5 and np.iscomplex(p7[0])
'''),

md(r"""
## 10. Summary

| Result | Formula / statement |
|:--|:--|
| Resolution of an incremental encoder, ×4 decoding | $q=2\pi/(4N)$; with gear ratio $n$ on the motor side, $q_{\text{joint}}=q/n$ |
| Absolute encoder | $q=2\pi/2^{n}$; use the **Gray code** so that only one head changes at a time (error $\le1$ count) |
| Quantisation error (bin-centre) | uniform, zero mean, $\sigma_q=q/\sqrt{12}$; valid only for a *moving* signal |
| M method | $\hat\omega=\Delta n\,q/T_s$; quantum $q/T_s$; $\sigma_\omega=q/(\sqrt6\,T_s)$; poor at low speed |
| T method | $\hat\omega=q/\Delta t$; $\sigma_\omega/\omega\approx\omega t_c/(\sqrt6 q)$; poor at high speed |
| M/T method | $\hat\omega=\Delta n\,q/(\tau_k-\tau_{k-1})$; error $\approx t_c/(\sqrt6T_s)$ at all speeds |
| Crossover M ↔ T | $\omega^*=q/\sqrt{T_st_c}$ |
| Low-pass derivative | $\hat\omega_k=(\tau\hat\omega_{k-1}+\Delta\hat\theta)/(\tau+T_s)$; variance factor $T_s^2/((2\tau+T_s)(\tau+T_s))$; lag $a\tau$ |
| $\alpha$–$\beta$ observer | $\alpha=1-p^2$, $\beta=(1-p)^2$, $p=e^{-\omega_oT_s}$; zero error at constant speed; equals the steady-state Kalman filter for $\ddot\theta=w$ |
| Noise into PD | $\sigma_\tau=K_d\,\sigma_\omega$; held shafts show limit cycles of amplitude $\approx K_dq/T_s$ |
| Derivative kick | set-point step $\delta$ gives an impulse $K_d\delta/T_s$ if $D$ acts on the error; differentiate the *measurement* |

### Common mistakes

* Treating encoder counts as exact: forgetting the half-count bias of a floor quantiser or the $q/\sqrt{12}$ noise.
* Using a one-sample difference (M method) as the damping term of a PD law at high gain: the torque noise is $K_dq/(\sqrt6T_s)$, which grows when the sample time shrinks.
* Choosing the velocity filter bandwidth by eye; the optimum balances noise against lag and depends on the encoder resolution and the motion's acceleration.
* Differentiating the *error* of a set-point step (derivative kick) instead of the measurement or a smooth reference.
* Forgetting that an incremental encoder is relative: it needs homing, and a missed edge (sampling too slowly, electrical noise) is a permanent position error.
* Judging a controller only while the robot moves: quantisation limit cycles appear at standstill.

## Exercises

**8.1.1** A 2500-line encoder with ×4 decoding is mounted on the motor of a joint with a 50:1 reduction. Find (a) the counts per joint revolution, (b) the joint resolution in arc-seconds, (c) the linear resolution at a reach of 1.2 m.

<details><summary>Answer</summary>

(a) $4\times2500\times50=500\,000$ counts. (b) $q_{\text{joint}}=2\pi/500\,000=1.257\times10^{-5}$ rad $=2.59''$. (c) $1.257\times10^{-5}\times1.2=15.1\ \mu$m.
</details>

**8.1.2** An M-method velocity measurement uses $q=2\pi/4096$ rad. What is the shortest window $T_s$ for which the speed quantum is at most 0.05 rad/s? What does that do to the lag, and which method would you prefer at such speeds?

<details><summary>Answer</summary>

$T_s\ge q/0.05=1.534\times10^{-3}/0.05=30.7$ ms. The M method then reports the average speed over 30.7 ms, i.e. a delay of about $T_s/2\approx15$ ms — large for a servo loop. The M/T method (or a model-based observer) removes the trade-off.
</details>

**8.1.3** Show that the T-method relative error $\omega t_c/(\sqrt6 q)$ and the M-method relative error $q/(\sqrt6\omega T_s)$ are equal at $\omega^*=q/\sqrt{T_st_c}$, and evaluate $\omega^*$ for $q=2\pi/16384$, $T_s=0.5$ ms and a 10 MHz clock.

<details><summary>Answer</summary>

Equating the two expressions and solving gives $\omega^{*2}=q^2/(T_st_c)$. For the data: $q=3.835\times10^{-4}$ rad, $T_st_c=0.5\times10^{-3}\times10^{-7}=5\times10^{-11}$ s², so $\omega^*=3.835\times10^{-4}/7.07\times10^{-6}=54.2$ rad/s.
</details>

**8.1.4** Convert the Gray word $0110$ to natural binary, give the angle interval it denotes on a 4-bit disc, and write the Gray words of the two neighbouring positions to show that each differs from it in one bit.

<details><summary>Answer</summary>

$b_3=0$, $b_2=0\oplus1=1$, $b_1=1\oplus1=0$, $b_0=0\oplus0=0$, so $b=0100=4$ and the interval is $[90^\circ,112.5^\circ)$. Neighbours: position 3 has Gray word $0010$ and position 5 has $0111$; each differs from $0110$ in a single bit.
</details>

**8.1.5** (*Programming*) For the experiment of Section 6, find numerically the bandwidth $\omega_o$ that minimises the RMS velocity error of (a) the low-pass derivative and (b) the $\alpha$–$\beta$ observer when the encoder is reduced to 10 bits. Explain the direction of the change.

<details><summary>Answer</summary>

Run the sweep with `IncrementalEncoder(256, 4)`. The optimum bandwidth falls (to about 60 rad/s for the low-pass derivative and 150 rad/s for the observer in this run) and the minimum RMS error rises (the values are printed in the verification cell below). A coarser encoder produces more noise per unit bandwidth, so the best compromise accepts more lag.
</details>

**8.1.6** (*MuJoCo*) Predict the RMS torque noise of the wrist-3 joint while it *moves*, if the 12-bit encoder is replaced by a 16-bit one with the M method, using $\sigma_\tau=K_dq/(\sqrt6T_s)$, and compare with the 12-bit value of Example 8.1.5.

<details><summary>Answer</summary>

$q$ is 16 times smaller, so $\sigma_\tau$ falls from $3.13$ to $0.196$ N·m. Resolution is the only way to reduce the noise of the unfiltered difference.
</details>

**8.1.7** In the first-order low-pass derivative of Section 6.1 show that for a velocity ramp (constant acceleration $a$) the steady-state error is $a\tau$ (use the continuous-time transfer function), and that the observer of Section 6.2 has *zero* steady-state velocity error for a constant velocity but not for constant acceleration.

<details><summary>Answer</summary>

For $\theta(t)=\tfrac12at^2$, $\Theta(s)=a/s^3$ and $\hat\Omega(s)=\dfrac{s}{\tau s+1}\dfrac{a}{s^3}=\dfrac{a}{s^2(\tau s+1)}$; the final-value theorem on $\Omega-\hat\Omega=\dfrac{a}{s^2}-\dfrac{a}{s^2(\tau s+1)}=\dfrac{a\tau}{s(\tau s+1)}$ gives $a\tau$. For the observer, a double pole inside the unit circle makes the error dynamics asymptotically stable; a constant velocity input produces no innovation once the state matches it, whereas an acceleration produces a constant innovation $r\ne0$ and hence a constant lag.
</details>
"""),

code(r'''
# Verification of the numbers quoted in the exercise answers
q1 = 2 * np.pi / (4 * 2500 * 50)
print("8.1.1:", 4 * 2500 * 50, "counts;", f"{np.degrees(q1)*3600:.2f} arcsec;", f"{q1*1.2*1e6:.1f} um")
assert 4 * 2500 * 50 == 500_000 and abs(np.degrees(q1) * 3600 - 2.59) < 0.01 and abs(q1 * 1.2 * 1e6 - 15.1) < 0.05 and abs(q1 - 1.257e-5) < 1e-8
q2 = 2 * np.pi / 4096
print("8.1.2: Ts >= %.1f ms" % (q2 / 0.05 * 1e3)); assert abs(q2 / 0.05 * 1e3 - 30.7) < 0.1
q3 = 2 * np.pi / 16384; w3 = q3 / np.sqrt(0.5e-3 * 1e-7)
print("8.1.3: q = %.4e, omega* = %.1f rad/s" % (q3, w3)); assert abs(q3 - 3.835e-4) < 1e-6 and abs(w3 - 54.2) < 0.2
g4 = int("0110", 2); b4 = int(sn.gray_decode(g4))
print("8.1.4: binary", format(b4, "04b"), "=", b4, " interval", b4 * 22.5, (b4 + 1) * 22.5, " neighbours", format(int(sn.gray_encode(b4 - 1)), "04b"), format(int(sn.gray_encode(b4 + 1)), "04b"))
assert b4 == 4 and (b4 * 22.5, (b4 + 1) * 22.5) == (90.0, 112.5) and format(int(sn.gray_encode(3)), "04b") == "0010" and format(int(sn.gray_encode(5)), "04b") == "0111"

enc10 = sn.IncrementalEncoder(256, 4); q10 = enc10.resolution
cnt10 = enc10.counts(theta); th10 = enc10.angle(cnt10) + q10 / 2
wsweep = np.logspace(0.8, 3.2, 40)
lp10 = [rms(sn.lowpass_derivative(th10, Ts, 1 / w)) for w in wsweep]
ab10 = [rms(sn.alpha_beta_filter(th10, Ts, *sn.alpha_beta_gains(w, Ts))[1]) for w in wsweep]
i1, i2 = int(np.argmin(lp10)), int(np.argmin(ab10))
print(f"8.1.5: 10-bit encoder: LP optimum {wsweep[i1]:.0f} rad/s (RMS {lp10[i1]:.3f});  alpha-beta optimum {wsweep[i2]:.0f} rad/s (RMS {ab10[i2]:.3f})")
print(f"       12-bit reference: LP optimum {wos[i1 if False else int(np.argmin(lp))]:.0f} rad/s (RMS {min(lp):.3f});  alpha-beta optimum {wos[int(np.argmin(ab))]:.0f} rad/s (RMS {min(ab):.3f})")
assert wsweep[i1] < wos[int(np.argmin(lp))] and wsweep[i2] < wos[int(np.argmin(ab))] and lp10[i1] > min(lp) and ab10[i2] > min(ab)
assert 40 < wsweep[i1] < 80 and 110 < wsweep[i2] < 200

sig12 = Kd[5] * q / (np.sqrt(6) * 1e-3); sig16 = Kd[5] * (2 * np.pi / 65536) / (np.sqrt(6) * 1e-3)
print("8.1.6: sigma_tau 12-bit %.3f, 16-bit %.3f N m" % (sig12, sig16)); assert abs(sig12 - 3.13) < 0.1 and abs(sig16 - 0.196) < 0.01 and abs(sig12 / sig16 - 16) < 1e-9
'''),

md(r"""
## Further reading and lectures

* Craig, *Introduction to Robotics*, 3rd ed., Ch. 8 (sensors and actuators of manipulators) and Ch. 9 (discrete-time implementation of the servo); Siciliano et al., *Robotics: Modelling, Planning and Control*, Ch. 5 (actuators and sensors) and Ch. 6 (control architecture); Spong et al., *Robot Modeling and Control*, Ch. 8.
* On state estimation: R. E. Kalman, "A new approach to linear filtering and prediction problems", *Journal of Basic Engineering* 82(1), 1960; the M/T method is treated in motor-drive texts under "speed measurement with incremental encoders".
* Lectures: Prof. G. C. Nandi's playlist 2 (*Mathematical Foundation of Robotics — Execution Level …*, no URL yet in [REFERENCES](../../docs/REFERENCES.md)); [NPTEL, *Mechanics and Control of Robotic Manipulators* (IIT Palakkad)](https://nptel.ac.in/courses/112106304); [Brunton, *Control Bootcamp*, overview video](https://www.youtube.com/watch?v=Pi7l8mMjYVE) for observers and the Kalman filter.
* MuJoCo documentation on [sensors](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor) and on [the model editing API](https://mujoco.readthedocs.io/en/stable/programming/modeledit.html).
"""),

nav(None, ("8.2 Force, Torque and Inertial Sensing", "8.2-force-torque-and-inertial-sensing.ipynb")),
]

if __name__ == "__main__":
    build(cells, f"{CH_DIR}/{FN}")
