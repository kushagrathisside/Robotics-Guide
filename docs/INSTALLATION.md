# Installation Guide

This guide explains how to obtain a working environment for the notebooks of
*Mathematical Foundations of Robotics* (MFoR) and *Robot Motion Control* (RMC).
It assumes no previous experience with Python. Three routes are described; you
need to follow only one of them.

| Route | Effort | Best for | Interactive 3D plots | MuJoCo viewer window |
|-------|--------|----------|----------------------|----------------------|
| **A. Google Colab** | None: runs in a web browser | A first look, or a computer on which you cannot install software | Yes, after one extra command | No (rendered images and videos only) |
| **B. Local, `venv` + `pip`** | About 10 minutes | Most students on Windows, macOS or Linux | Yes | Yes |
| **C. Local, conda / mamba** | About 15 minutes | Students who already use Anaconda, or who need Pinocchio on Windows | Yes | Yes |

Whichever route you choose, the software is the same. The core packages are
listed in [`requirements.txt`](../requirements.txt) (NumPy, SciPy, SymPy,
Matplotlib, ipywidgets, ipympl, JupyterLab, Pillow, and the MuJoCo simulator
with `mediapy`), and optional simulation libraries are listed in
[`requirements-sim.txt`](../requirements-sim.txt).

> **Which Python?** Use a 64-bit Python 3.11 or 3.12. These versions are
> supported by every package used in the course, including MuJoCo, whose
> project page asks for Python 3.11 or newer. Python 3.13 also works for the
> core material, but some optional libraries lag behind new Python releases.

---

## 1. Route A: Google Colab (no installation)

[Google Colab](https://colab.research.google.com) runs Jupyter notebooks on a
Google server; you need only a browser and a Google account. NumPy, SciPy,
SymPy, Matplotlib and Pillow are already installed there.

1. Open a notebook directly from GitHub. Replace `OWNER` with the account
   that hosts this repository and adjust the path to the notebook you want:

   ```text
   https://colab.research.google.com/github/OWNER/Robotics-Guide/blob/main/MFoR/<chapter>/<notebook>.ipynb
   ```

   Alternatively, in Colab choose *File → Open notebook → GitHub* and paste
   the repository URL.

2. Run the following cell first. It installs the packages that Colab lacks
   and fetches the repository so that the helper package `rgkit` can be
   imported.

   ```python
   !pip install -q mujoco mediapy ipympl
   !git clone -q https://github.com/OWNER/Robotics-Guide.git
   %cd Robotics-Guide
   ```

   `pip install mujoco` works on Colab without further configuration, because
   the MuJoCo wheel contains the compiled simulator.

3. **Interactive Matplotlib figures** (rotatable 3D plots) require Colab's
   custom widget manager. Run once per session:

   ```python
   from google.colab import output
   output.enable_custom_widget_manager()
   %matplotlib widget
   ```

4. **MuJoCo rendering on Colab.** Colab has no display, so MuJoCo must render
   offscreen. The official MuJoCo tutorial notebook selects a GPU runtime
   (*Runtime → Change runtime type → GPU*) and sets `MUJOCO_GL=egl`. On the
   default CPU runtime, software rendering with OSMesa is the reliable choice:

   ```python
   !apt-get -qq install -y libosmesa6 > /dev/null   # only if osmesa is missing
   %env MUJOCO_GL=osmesa
   %env PYOPENGL_PLATFORM=osmesa
   import mujoco                                     # import *after* setting the variables
   ```

   The environment variables must be set **before** `mujoco` is first
   imported. If you change them later, choose *Runtime → Restart session* and
   run the cells again.

Files that you create on Colab are lost when the session ends. Save your work
with *File → Save a copy in Drive*, or download it.

---

## 2. Route B: local installation with `venv` and `pip`

A *virtual environment* is a private folder that holds one Python interpreter
and the packages installed for one project. It prevents the course packages
from interfering with anything else on your computer.

### 2.1 Install Python and Git

| Operating system | Recommended procedure |
|------------------|-----------------------|
| **Windows 10/11** | Install Python 3.12 from [python.org](https://www.python.org/downloads/). On the first installer screen tick **"Add python.exe to PATH"**. Install [Git for Windows](https://git-scm.com/download/win). Use *PowerShell* or *Windows Terminal* for the commands below. |
| **macOS** | Install Python 3.12 from [python.org](https://www.python.org/downloads/) (a *universal2* installer) or with Homebrew: `brew install python@3.12`. Git is provided by the *Xcode Command Line Tools* (`xcode-select --install`). Do not use the system `/usr/bin/python3` for coursework. |
| **Linux (Debian/Ubuntu)** | `sudo apt update && sudo apt install python3 python3-venv python3-pip git`. On Fedora: `sudo dnf install python3 git`. |

Confirm the installation by opening a new terminal and typing
`python --version` (Windows) or `python3 --version` (macOS, Linux).

### 2.2 Obtain the repository

```bash
git clone https://github.com/OWNER/Robotics-Guide.git
cd Robotics-Guide
```

If you prefer not to use Git, download the ZIP archive from the GitHub page
(*Code → Download ZIP*) and extract it.

### 2.3 Create and activate the environment

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell reports that *running scripts is disabled on this system*, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate
again. In the older *Command Prompt* the activation command is
`.venv\Scripts\activate.bat`.

**macOS and Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When the environment is active, the prompt begins with `(.venv)`. You must
activate it in every new terminal before working on the course.

### 2.4 Install the packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# optional extras (read the comments inside the file first):
python -m pip install -r requirements-sim.txt
```

Writing `python -m pip` rather than `pip` guarantees that the packages go into
the interpreter that is currently active.

> **A faster alternative.** The tool [`uv`](https://github.com/astral-sh/uv)
> performs the same steps considerably faster:
> `uv venv --python 3.12 .venv`, activate as above, then
> `uv pip install -r requirements.txt`.

### 2.5 Register the kernel and start JupyterLab

```bash
python -m ipykernel install --user --name robotics-guide --display-name "Python (Robotics Guide)"
jupyter lab
```

JupyterLab opens in your web browser. Open a notebook and confirm that the
kernel shown in the upper-right corner is *Python (Robotics Guide)*.

---

## 3. Route C: conda or mamba

We recommend [Miniforge](https://github.com/conda-forge/miniforge), which
provides `conda` and the faster `mamba`, configured to use the community
`conda-forge` channel. Anaconda and Miniconda work equally well if you add
`-c conda-forge` to the commands.

```bash
mamba create -n robotics -c conda-forge python=3.12 \
      numpy scipy sympy matplotlib ipywidgets ipympl jupyterlab pillow ffmpeg pytest
mamba activate robotics            # with plain conda: conda activate robotics
python -m pip install mujoco mediapy
python -m ipykernel install --user --name robotics-guide --display-name "Python (Robotics Guide)"
jupyter lab
```

MuJoCo is installed with `pip` inside the conda environment; this is the
installation method documented by the MuJoCo project. Conda also installs the
`ffmpeg` program, which is needed to save animations as MP4 files.

Optional libraries:

```bash
mamba install -c conda-forge pinocchio      # the only easy route on Windows
python -m pip install roboticstoolbox-python
```

---

## 4. Platform-specific notes

| Platform | Notes |
|----------|-------|
| **Windows** | Use the 64-bit installer from python.org; MuJoCo publishes wheels for x86-64 Windows only (not for ARM-based Windows). Long paths inside OneDrive-synchronised folders occasionally cause installation errors; keep the repository in a short path such as `C:\courses\Robotics-Guide`. Pinocchio has no `pip` wheel for Windows: use conda (Route C). Drake does not support Windows natively; use WSL 2. |
| **Windows Subsystem for Linux (WSL 2)** | Follow the Linux instructions inside WSL. On Windows 11, WSLg provides a display, so `python -m mujoco.viewer` opens a window. Without WSLg, treat WSL as headless (Section 7.4). |
| **macOS on Apple Silicon (M1–M4)** | All core packages and MuJoCo provide native `arm64` wheels. Make sure that your Python is itself `arm64`: `python -c "import platform; print(platform.machine())"` should print `arm64`, not `x86_64` (the latter indicates an Intel Python running under Rosetta). Scripts that open MuJoCo's *passive* viewer must be started with `mjpython` (Section 7.3). |
| **macOS on Intel processors** | The core packages work as usual. MuJoCo, however, publishes Intel-Mac wheels only up to version 3.10.0; newer releases target Apple Silicon. Install with `python -m pip install --only-binary=mujoco mujoco`, which selects the newest release that has a pre-built wheel, or pin `mujoco==3.10.0`. The course code works with both. |
| **Linux** | MuJoCo wheels target the `manylinux` standard and run on all mainstream distributions. For headless rendering install `libegl1` and/or `libosmesa6` (Section 7.4). For the interactive viewer a desktop session with OpenGL is required. |
| **Chromebook / tablet** | Use Google Colab (Route A). |

---

## 5. JupyterLab or VS Code?

Both editors run the same notebooks; choose whichever you find more
comfortable.

| | JupyterLab | Visual Studio Code |
|---|---|---|
| Installation | Included in `requirements.txt`; start with `jupyter lab` | Install [VS Code](https://code.visualstudio.com/) and the *Python* and *Jupyter* extensions from Microsoft |
| Selecting the environment | Choose the kernel *Python (Robotics Guide)* | *Select Kernel* (upper right) → *Python Environments* → `.venv` or `robotics` |
| ipywidgets sliders | Supported | Supported |
| `%matplotlib widget` (ipympl) | Supported | Supported in recent versions of the Jupyter extension |
| Strengths | The reference environment for notebooks; identical to Colab's model | Integrated editor, debugger and Git tools; convenient for the `.py` scripts in `simulations/` |

If a figure or a slider does not appear in VS Code, first confirm that the
selected kernel is the course environment, and then try the same notebook in
JupyterLab to determine whether the difficulty lies with the editor.

---

## 6. Interactive figures and animations

### 6.1 ipywidgets and `%matplotlib widget`

Many notebooks use sliders (`ipywidgets`) and rotatable 3D figures
(`ipympl`). Both packages are in `requirements.txt`. With JupyterLab 4 and
ipywidgets 8 no further configuration is required: the browser extensions are
installed together with the Python packages.

At the top of a notebook that uses interactive figures, write

```python
%matplotlib widget
```

`%matplotlib widget` and `%matplotlib ipympl` are equivalent. Every figure then
becomes a live widget that can be rotated (3D axes), zoomed and panned. To
return to static images, use `%matplotlib inline`. Static images are also the
appropriate choice when a notebook is exported to HTML or PDF, because a
widget is displayed only while a kernel is running.

A minimal test:

```python
%matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
t = np.linspace(0, 4 * np.pi, 200)
ax.plot(np.cos(t), np.sin(t), t / 10)
plt.show()
```

If a rotatable helix appears, interactive plotting works. If instead a message
such as *Error displaying widget* appears, see Section 8.

### 6.2 Saving animations: Pillow and ffmpeg

Matplotlib animations (`matplotlib.animation.FuncAnimation`) can be shown and
saved in three ways:

| Goal | Code | Requirement |
|------|------|-------------|
| Show in the notebook | `from IPython.display import HTML; HTML(anim.to_jshtml())` | None |
| Save as GIF | `anim.save("motion.gif", writer="pillow", fps=20)` | Pillow (in `requirements.txt`) |
| Save as MP4 | `anim.save("motion.mp4", writer="ffmpeg", fps=30)` | The `ffmpeg` program |

`ffmpeg` is a separate program, not a Python package. Install it with
`winget install Gyan.FFmpeg` (Windows), `brew install ffmpeg` (macOS),
`sudo apt install ffmpeg` (Debian/Ubuntu) or `mamba install -c conda-forge
ffmpeg` (any platform). It is already present on Colab. The `mediapy`
function `show_video`, used to display MuJoCo videos, also relies on
`ffmpeg`; where `ffmpeg` is unavailable, save a GIF with Pillow instead.

---

## 7. MuJoCo

[MuJoCo](https://github.com/google-deepmind/mujoco) (*Multi-Joint dynamics
with Contact*) is the physics simulator used for the course's simulations. Its
role in the course is explained in [`docs/MUJOCO.md`](MUJOCO.md).

### 7.1 Installation

```bash
python -m pip install mujoco mediapy
```

The MuJoCo wheel contains the compiled simulator; no separate download or
licence key is required. Pre-built wheels are published for Linux (x86-64 and
AArch64), macOS (Apple Silicon) and Windows (x86-64). On an Intel Mac, use
`python -m pip install --only-binary=mujoco mujoco` (see Section 4).

### 7.2 Verification

```bash
python -c "import mujoco; print(mujoco.__version__)"
python simulations/mujoco/02_planar_2r_forward_kinematics.py
```

The second command loads the course's two-link arm, checks MuJoCo's forward
kinematics against the analytic formula, and prints *All forward-kinematics
and Jacobian checks passed*.

### 7.3 The interactive viewer

On a computer with a display:

```bash
python -m mujoco.viewer                                            # empty viewer; drag an XML file onto it
python -m mujoco.viewer --mjcf=simulations/mujoco/models/planar_2r.xml
```

From Python, `mujoco.viewer.launch(model)` opens a viewer that runs the
simulation itself and blocks until the window is closed, whereas
`mujoco.viewer.launch_passive(model, data)` returns immediately and lets your
own loop step the simulation, calling `viewer.sync()` to refresh the picture.

> **macOS.** `launch_passive` must be run under the `mjpython` launcher, which
> is installed with MuJoCo: `mjpython my_script.py` instead of
> `python my_script.py`. Otherwise MuJoCo raises
> `RuntimeError: launch_passive requires that the Python script be run under mjpython on macOS`.

The viewer is a desktop window and cannot be embedded in a notebook; inside
notebooks we render images and videos instead (see
[`docs/MUJOCO.md`](MUJOCO.md#7-using-mujoco-inside-jupyter-notebooks)).

### 7.4 Rendering without a display (servers, Colab, WSL, CI)

MuJoCo chooses an OpenGL back end when it is **first imported**, according to
the environment variable `MUJOCO_GL`:

| `MUJOCO_GL` | Platform | Meaning |
|-------------|----------|---------|
| *(unset)* or `glfw` | all | Default. Uses a (possibly hidden) window; needs a display. |
| `egl` | Linux | Headless rendering through EGL, on a GPU or with Mesa's software driver. Needs `libegl1`. |
| `osmesa` | Linux | Pure software rendering. Needs `libosmesa6`. Slow, but works everywhere. |
| `cgl` | macOS | Native headless context (chosen automatically on macOS). |
| `wgl` | Windows | Native Windows context. |

Set `PYOPENGL_PLATFORM` to the same value as `MUJOCO_GL` for `egl` and
`osmesa`. In a terminal:

```bash
sudo apt install libegl1 libosmesa6          # Debian/Ubuntu, once
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl   # or: osmesa
python simulations/mujoco/03_planar_2r_pd_control.py
```

In a notebook, set the variables in the very first cell:

```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")          # or "osmesa"
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
import mujoco
```

The scripts in [`simulations/mujoco/`](../simulations/mujoco/) detect a
headless Linux machine automatically, try `egl` and then `osmesa`, and skip
rendering (while still completing their numerical checks) if neither works.
On the machine on which these files were prepared (Ubuntu 24.04, no GPU), both
back ends worked once `libegl1` and `libosmesa6` had been installed, and
neither worked before.

### 7.5 Real robot models (MuJoCo Menagerie)

Several notebooks simulate real robots (the Dynamixel 2R arm, the Universal
Robots UR5e, the Franka Emika Panda and others) taken from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). These
models are not stored in this repository: they are large (about 30 MB each for
the UR5e and the Panda) and each has its own licence. The helper
`rgkit.robots.load_robot(...)` downloads a model the first time it is needed,
from a fixed commit of Menagerie, and keeps it in a cache.

* **First run.** An internet connection is required the first time each robot
  is loaded. The files come from `raw.githubusercontent.com`; if your network
  blocks that address, use the offline route below. On Colab the download
  works without any configuration (but the cache is lost when the session
  ends, so it is repeated in each new session).
* **Cache location.** By default `~/.cache/robotics-guide` (on Windows,
  `C:\Users\<you>\.cache\robotics-guide`). Set the environment variable
  `RG_CACHE` to choose another folder.
* **Offline use.** Clone Menagerie once and point `MUJOCO_MENAGERIE_PATH` at the
  clone; nothing is then downloaded:

  ```bash
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
  export MUJOCO_MENAGERIE_PATH="$PWD/mujoco_menagerie"          # macOS, Linux
  ```

  ```powershell
  $env:MUJOCO_MENAGERIE_PATH = "$PWD\mujoco_menagerie"          # Windows PowerShell
  ```

  To make the setting permanent, add the `export` line to `~/.bashrc` or
  `~/.zshrc`, or on Windows use *Settings → System → About → Advanced system
  settings → Environment Variables*. In a notebook you may instead write
  `os.environ["MUJOCO_MENAGERIE_PATH"] = "..."` before importing `rgkit.robots`.
* **Licences.** Robot licences are per model (Apache-2.0, BSD-3-Clause, MIT,
  ...). Each downloaded model folder contains its own `LICENSE` file; consult
  it before redistributing the model or images made from it.

A quick test:

```bash
python -c "from rgkit.robots import load_robot; m, d = load_robot('2r'); print('nq =', m.nq)"
```

which should print `nq = 2` (run it from the repository root).

---

## 8. Troubleshooting

| Symptom or error message | Probable cause | Remedy |
|--------------------------|----------------|--------|
| `python: command not found` (macOS/Linux) or the Microsoft Store opens (Windows) | Python not installed, or not on `PATH` | Use `python3` on macOS/Linux; on Windows reinstall from python.org with *Add python.exe to PATH*, or use `py`. |
| `ModuleNotFoundError: No module named 'numpy'` (or `mujoco`, `rgkit`) in a notebook | The notebook runs on a different kernel from the environment in which you installed the packages | Select the kernel *Python (Robotics Guide)*; run `import sys; print(sys.executable)` to see which interpreter is in use. For `rgkit`, start Jupyter from the repository root. |
| `running scripts is disabled on this system` (PowerShell) | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `ERROR: No matching distribution found for mujoco` | Python too old, 32-bit Python, or an unsupported platform (e.g. ARM Windows) | Install 64-bit Python 3.11 or 3.12; check with `python -c "import struct; print(8 * struct.calcsize('P'))"` (must print 64). |
| `Failed building wheel for mujoco`, or CMake errors while installing `mujoco` | No pre-built wheel for your platform (typically an Intel Mac with a recent MuJoCo release), so pip tries to compile from source | `python -m pip install --only-binary=mujoco mujoco`, or pin `mujoco==3.10.0` (Section 4). |
| `error: Microsoft Visual C++ 14.0 or greater is required` while installing `pybullet` | No pre-built wheel for your platform, so pip tries to compile | PyBullet is optional. Either skip it, install the *Microsoft C++ Build Tools*, or use Linux/WSL with Python 3.11. |
| `Error displaying widget: model not found`, or a figure appears as text such as `Canvas(toolbar=...)` | ipympl/ipywidgets not installed in the environment that runs JupyterLab, or the page was not reloaded | `python -m pip install ipympl ipywidgets` in the same environment as `jupyter lab`, restart JupyterLab, reload the browser tab. On Colab, call `output.enable_custom_widget_manager()`. |
| Sliders appear but nothing changes | The kernel was restarted or the cell has not been run | Run the cell again (*Run → Run All Cells*). |
| `MovieWriter ffmpeg unavailable; using Pillow instead.` | `ffmpeg` is not installed | Save as `.gif`, or install `ffmpeg` (Section 6.2). |
| `RuntimeError: Program 'ffmpeg' is not found` (from `mediapy`) | As above | Install `ffmpeg`, or show frames with `mediapy.show_images` / save a GIF with Pillow. |
| `GLFWError: ... X11: The DISPLAY environment variable is missing`, followed by `FatalError: an OpenGL platform library has not been loaded into this process` | MuJoCo tried to open a window on a headless machine | Set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` before importing `mujoco` (Section 7.4). |
| `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | `MUJOCO_GL=egl`, but the EGL library is missing | `sudo apt install libegl1` (plus a GPU driver, if available), or use `osmesa`. |
| `AttributeError: 'NoneType' object has no attribute 'glGetError'` with `MUJOCO_GL=osmesa` | OSMesa library missing | `sudo apt install libosmesa6`. |
| `RuntimeError: invalid value for environment variable MUJOCO_GL` | Misspelt value, or a value not valid on this OS (e.g. `egl` on Windows) | Use one of the values in the table of Section 7.4 for your platform. |
| `Exception ignored in: <function Renderer.__del__ ...>` followed by `EGLError` when a script or kernel exits | A `mujoco.Renderer` was not closed before the EGL context was torn down | Harmless. Avoid it by using `with mujoco.Renderer(model) as renderer:` or calling `renderer.close()`. |
| Changing `MUJOCO_GL` has no effect in a notebook | `mujoco` had already been imported | Restart the kernel, set the variable in the first cell, then import `mujoco`. |
| `RuntimeError: launch_passive requires that the Python script be run under mjpython on macOS` | macOS restriction on GUI threads | Run the script with `mjpython` instead of `python`. |
| `AttributeError: 'MjData' object has no attribute 'qM'`, or `TypeError: mj_fullM(): incompatible function arguments` | The signature of `mj_fullM` differs between MuJoCo releases | Use the version-independent snippet in [`docs/MUJOCO.md`](MUJOCO.md#5-dynamics-quantities-craig-chapter-6), or upgrade MuJoCo. |
| `urllib.error.URLError`, `HTTP Error 403`, a timeout or `CERTIFICATE_VERIFY_FAILED` from `rgkit.robots.load_robot` | No internet access, a firewall or proxy blocking `raw.githubusercontent.com`, or a missing certificate bundle | Connect to the internet for the first run, or use a local clone with `MUJOCO_MENAGERIE_PATH` (Section 7.5). Behind a proxy with its own certificate, set `SSL_CERT_FILE` to its CA bundle. |
| `FileNotFoundError: ... not found in Menagerie commit ...`, or an XML or mesh error when loading a downloaded robot | A damaged file left by an interrupted download, or a `MUJOCO_MENAGERIE_PATH` clone that lacks the model | Delete the robot's folder in the cache (`~/.cache/robotics-guide` or `$RG_CACHE`) and load it again; check that the clone contains the model's directory. |
| `ValueError: XML Error: ...` when loading a model | Malformed MJCF or a wrong path | Read the line number in the message; load models with a path relative to the script (`Path(__file__).parent / "models" / ...`). |
| Notebook output differs slightly from the text (e.g. in the 15th digit) | Floating-point round-off differs between processors and library versions | Differences below about `1e-12` are expected and harmless. |

If a problem persists, open an issue on the repository and include the output
of:

```bash
python -c "import sys, platform; print(sys.version); print(platform.platform())"
python -m pip list
```
