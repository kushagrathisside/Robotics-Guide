"""Real robot models for the notebooks, fetched on demand from MuJoCo Menagerie.

`MuJoCo Menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_ is
a curated collection of high-quality robot models released by Google
DeepMind and the robot manufacturers. Each model keeps its own licence
(Apache-2.0, BSD-3-Clause, MIT ...); this repository therefore does **not**
copy the models. :func:`fetch` downloads one model directory at a *pinned*
commit the first time it is needed, stores it in a cache folder together with
its ``LICENSE`` file, and re-uses it afterwards, so notebooks stay
reproducible and the repository stays small.

Where the cache lives (first match wins):

1. ``$MUJOCO_MENAGERIE_PATH`` — an existing local clone of Menagerie
   (works offline);
2. ``$RG_CACHE`` — any directory you choose;
3. ``~/.cache/robotics-guide`` (``%LOCALAPPDATA%`` is not special-cased; the
   home directory is used on every platform).
"""

from __future__ import annotations

import os
import re
import ssl
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

__all__ = ["MENAGERIE_SHA", "ROBOTS", "fetch", "load_robot", "frames_option", "describe", "cache_dir"]

#: Commit of google-deepmind/mujoco_menagerie that every notebook is tested against.
MENAGERIE_SHA = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
_RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{sha}/{path}"

#: Robots used in the notebooks. ``scene`` is the XML to load (it includes the robot plus a floor and light).
ROBOTS = {
    "2r": dict(robot="dynamixel_2r.xml", dir="dynamixel_2r", scene="scene.xml", title="Dynamixel 2R arm",
               dof=2, maker="Marc Duclusaud & Grégoire Passault", licence="MIT"),
    "ur5e": dict(robot="ur5e.xml", dir="universal_robots_ur5e", scene="scene.xml", title="Universal Robots UR5e",
                 dof=6, maker="Universal Robots", licence="BSD-3-Clause"),
    "panda": dict(robot="panda.xml", dir="franka_emika_panda", scene="scene.xml", title="Franka Emika Panda",
                  dof=7, maker="Franka Emika", licence="Apache-2.0"),
    "iiwa14": dict(robot="iiwa14.xml", dir="kuka_iiwa_14", scene="scene.xml", title="KUKA LBR iiwa 14",
                   dof=7, maker="KUKA", licence="see model LICENSE"),
    "sawyer": dict(robot="sawyer.xml", dir="rethink_robotics_sawyer", scene="scene.xml", title="Rethink Robotics Sawyer",
                   dof=7, maker="Rethink Robotics", licence="see model LICENSE"),
    "op3": dict(robot="op3.xml", dir="robotis_op3", scene="scene.xml", title="ROBOTIS OP3 humanoid",
                dof=20, maker="ROBOTIS", licence="see model LICENSE"),
}


def cache_dir() -> Path:
    """Directory in which downloaded models are stored."""
    env = os.environ.get("RG_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "robotics-guide"


def _ssl_context():
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        if os.environ.get(var) and Path(os.environ[var]).exists():
            return ssl.create_default_context(cafile=os.environ[var])
    return ssl.create_default_context()


def _download(path: str, dest: Path) -> bool:
    """Download one file of the pinned Menagerie commit; return False on HTTP 404."""
    url = _RAW.format(sha=MENAGERIE_SHA, path=path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    dest.write_bytes(data)
    return True


_FILE_ATTRS = ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")


def _referenced_files(xml_path: Path):
    """Yield (path relative to the model directory, True) for every file the XML needs."""
    root = ET.parse(xml_path).getroot()
    base = xml_path.parent
    compiler = root.find("compiler")
    meshdir = (compiler.get("meshdir", "") if compiler is not None else "")
    texturedir = (compiler.get("texturedir", "") if compiler is not None else "")
    assetdir = (compiler.get("assetdir", "") if compiler is not None else "")
    for el in root.iter():
        for attr in _FILE_ATTRS:
            f = el.get(attr)
            if not f:
                continue
            if el.tag == "include" or el.tag == "model":
                yield (base / f, "xml")
            elif el.tag == "mesh" or el.tag == "skin":
                yield (base / (meshdir or assetdir) / f, "asset")
            elif el.tag == "texture":
                yield (base / (texturedir or assetdir) / f, "asset")
            elif el.tag == "hfield":
                yield (base / assetdir / f, "asset")
            else:
                yield (base / f, "asset")


def fetch(name: str, verbose: bool = True) -> Path:
    """Make sure the Menagerie model ``name`` (a key of :data:`ROBOTS`) is on disk; return its directory."""
    spec = ROBOTS[name]
    local = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if local and (Path(local) / spec["dir"] / spec["scene"]).exists():
        return Path(local) / spec["dir"]

    root = cache_dir() / f"menagerie-{MENAGERIE_SHA[:8]}"
    model_dir = root / spec["dir"]
    marker = model_dir / ".complete"
    if marker.exists():
        return model_dir

    if verbose:
        print(f"Downloading {spec['title']} from MuJoCo Menagerie ({MENAGERIE_SHA[:8]}) ...", file=sys.stderr)
    todo, done = [model_dir / spec["scene"]], set()
    for extra in ("LICENSE", "README.md"):
        _download(f"{spec['dir']}/{extra}", model_dir / extra)
    while todo:
        p = todo.pop()
        if p in done:
            continue
        done.add(p)
        rel = p.relative_to(root).as_posix()
        if not p.exists() and not _download(rel, p):
            raise FileNotFoundError(f"{rel} not found in Menagerie commit {MENAGERIE_SHA[:8]}")
        if p.suffix.lower() == ".xml":
            for q, kind in _referenced_files(p):
                q = Path(os.path.normpath(q))
                if q not in done:
                    todo.append(q)
    marker.write_text(MENAGERIE_SHA + "\n")
    return model_dir


def load_robot(name: str, scene: str = "light", frames: bool = False):
    """Return ``(model, data)`` for a robot of :data:`ROBOTS`, downloading it if necessary.

    ``scene="light"`` wraps the manufacturer's model in this repository's light scene (soft sky, chequered
    floor) so that figures look the same in every chapter; ``scene="menagerie"`` uses the scene file shipped
    with the model; ``scene="bare"`` loads the robot alone. The robot is put in its ``home`` keyframe when
    the model defines one. ``frames=True`` thins the frame axes drawn by :func:`frames_option`.
    """
    from . import mujoco_utils as mju  # configure OpenGL before importing mujoco

    d = fetch(name)
    spec = ROBOTS[name]
    if scene == "menagerie":
        xml = d / spec["scene"]
    elif scene == "bare":
        xml = d / spec["robot"]
    else:
        xml = d / "rg_scene_light.xml"
        xml.write_text(
            f'<mujoco model="{name} (light scene)">\n  <include file="{spec["robot"]}"/>\n'
            f'  {mju.scene_assets()}\n  <worldbody>\n    {mju.floor(z=0.0, size=3)}\n  </worldbody>\n</mujoco>\n')
    model = mju.mujoco.MjModel.from_xml_path(str(xml))
    data = mju.mujoco.MjData(model)
    if model.nkey:
        mju.mujoco.mj_resetDataKeyframe(model, data, 0)
    mju.mujoco.mj_forward(model, data)
    return model, data


def frames_option(model, length: float = 0.10, width: float = 0.006, kind: str = "body"):
    """Return a ``MjvOption`` that draws the coordinate frame of every body (or site, geom, joint-less ...).

    ``length`` and ``width`` are in metres; they are stored in ``model.vis.scale`` (as fractions of the
    model's size) so call this once per model before rendering.
    """
    from . import mujoco_utils as mju

    mj = mju.mujoco
    size = max(model.stat.meansize, 1e-6)
    model.vis.scale.framelength = length / size
    model.vis.scale.framewidth = width / size
    opt = mj.MjvOption()
    opt.frame = {"body": mj.mjtFrame.mjFRAME_BODY, "site": mj.mjtFrame.mjFRAME_SITE,
                 "geom": mj.mjtFrame.mjFRAME_GEOM, "world": mj.mjtFrame.mjFRAME_WORLD}[kind]
    return opt


def describe(name: str) -> str:
    """One-line attribution string for figure captions and README tables."""
    s = ROBOTS[name]
    return (f"{s['title']} — model from [MuJoCo Menagerie](https://github.com/google-deepmind/"
            f"mujoco_menagerie/tree/{MENAGERIE_SHA[:8]}/{s['dir']}) ({s['maker']}; licence: {s['licence']})")
