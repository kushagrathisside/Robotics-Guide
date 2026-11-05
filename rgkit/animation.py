"""Animation helper that works both live in Jupyter and on GitHub.

GitHub renders the *saved outputs* of a notebook but cannot run JavaScript,
so an interactive matplotlib animation would appear blank there. The
:func:`animate` function therefore behaves in two ways:

* **Live (default)** — returns an HTML/JavaScript player with play, pause
  and scrub controls (``FuncAnimation.to_jshtml``).
* **Static** (environment variable ``RG_STATIC=1``, used when the notebooks
  are executed for publication) — displays a *filmstrip* of key frames as a
  single PNG, which GitHub shows, and optionally writes an animated GIF that
  the chapter README embeds.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation

__all__ = ["animate", "is_static", "filmstrip"]


def is_static() -> bool:
    """True when notebooks are being executed for publication (``RG_STATIC=1``)."""
    return os.environ.get("RG_STATIC", "0") == "1"


def _grab(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _crop_common(images, pad=8, thresh=245):
    """Crop a list of same-size images to the union of their non-white regions."""
    mask = np.zeros(images[0].shape[:2], dtype=bool)
    for img in images:
        mask |= (img < thresh).any(axis=2)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return images
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, mask.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, mask.shape[1])
    return [img[r0:r1, c0:c1] for img in images]


def filmstrip(images, titles=None, width=12.0):
    """Show RGB images (animation key frames, MuJoCo renders) side by side.

    Wide images (e.g. two-panel animations) are laid out on a 2-column grid
    so that each frame stays legible.
    """
    n = len(images)
    h, w = images[0].shape[:2]
    cols = 2 if (w / h > 1.5 and n > 2) else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width, rows * width / cols * h / w + 0.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for i, img in enumerate(images):
        axes[i].imshow(img)
        if titles is not None:
            axes[i].set_title(titles[i], fontsize=10)
    fig.tight_layout()
    plt.show()


def animate(fig, update, frames, interval=50, gif=None, n_keyframes=4, keyframe_titles=None,
            fps=None):
    """Animate ``update(frame)`` on ``fig``.

    Parameters
    ----------
    fig : matplotlib Figure to animate.
    update : callable taking one element of ``frames`` and redrawing the figure.
    frames : sequence of frame values (e.g. ``np.linspace(0, 1, 60)``).
    interval : milliseconds between frames in the live player.
    gif : optional path; in static mode with ``RG_SAVE_ASSETS=1`` the GIF is
        written there (also written in live mode if given explicitly).
    n_keyframes : number of frames in the static filmstrip.
    """
    frames = list(frames)
    if is_static():
        if gif and os.environ.get("RG_SAVE_ASSETS", "0") == "1":
            _save_gif(fig, update, frames, gif, fps or max(1, round(1000 / interval)))
        idx = np.linspace(0, len(frames) - 1, n_keyframes).round().astype(int)
        imgs = []
        for i in idx:
            update(frames[i])
            imgs.append(_grab(fig))
        plt.close(fig)
        filmstrip(_crop_common(imgs), keyframe_titles)
        return None

    from IPython.display import HTML
    anim = mpl_animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    if gif:
        anim.save(gif, writer=mpl_animation.PillowWriter(fps=fps or max(1, round(1000 / interval))))
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


def _save_gif(fig, update, frames, path, fps):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = []
    for f in frames:
        update(f)
        raw.append(_grab(fig))
    imgs = []
    for arr in _crop_common(raw):
        img = Image.fromarray(arr)
        if img.width > 440:                      # keep README GIFs small
            img = img.resize((440, int(img.height * 440 / img.width)), Image.LANCZOS)
        imgs.append(img)
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=48) for im in imgs]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / fps),
                loop=0, optimize=True)
