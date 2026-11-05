"""
Figures. write_png needs only the standard library; the labeled figures use matplotlib (upstream's pin, imported
inside the function so `import wm_pipeline` stays light).
"""

import struct
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

# What the colors in CarDreamer's birdeye_wpt mean (birdeye_renderer.py:85-201, birdeye_handler.py:90-115).
BEV_LEGEND = "BEV colors: red = ego car, blue = ego's route, green = other cars, orange = other cars' planned paths, grey = road"


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal RGB PNG writer: (H, W, 3) uint8 in, a PNG file out."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 image, got {rgb.shape} {rgb.dtype}")
    height, width, _ = rgb.shape
    rows = b"".join(b"\x00" + np.ascontiguousarray(rgb[y]).tobytes() for y in range(height))

    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def real_vs_imagined_figure(comparisons: Sequence, every: int, seconds_per_step: float, title: str):
    """
    A labeled matplotlib figure: for each comparison, its real BEVs in one row and the imagined ones below, every
    `every`-th step of the horizon from the `every`-th on. Under each imagined frame: its mean pixel error and the
    no-motion guess's error.

    :param comparisons: objects with start, real, imagined, model_error and no_change_error, e.g. ImaginationComparison.
    """
    if not comparisons:
        raise ValueError("nothing to draw")
    import matplotlib

    matplotlib.use("Agg")  # draw straight to a file: no window, which WSL may not be able to open
    import matplotlib.pyplot as plt

    shown = list(range(every - 1, comparisons[0].real.shape[0], every))
    n_rows, n_cols = 2 * len(comparisons), len(shown)
    # About 1.3 inches per frame: at 100 dpi that is close to the BEV's own 128 pixels, so nothing gets blurred.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols + 1.2, 1.5 * n_rows + 1.3), squeeze=False)
    for block, comparison in enumerate(comparisons):
        real_axes, imagined_axes = axes[2 * block], axes[2 * block + 1]
        for column, k in enumerate(shown):
            real_axes[column].imshow(comparison.real[k], interpolation="nearest")
            imagined_axes[column].imshow(comparison.imagined[k], interpolation="nearest")
            imagined_axes[column].set_xlabel(f"{comparison.model_error[k]:.1f} | {comparison.no_change_error[k]:.1f}", fontsize=7)
            if block == 0:  # times only along the top: lower down they would collide with the error labels above
                real_axes[column].set_title(f"+{(k + 1) * seconds_per_step:.1f} s", fontsize=8)
        real_axes[0].set_ylabel(f"step {comparison.start}\n({comparison.start * seconds_per_step:.1f} s)\nreal", fontsize=8)
        imagined_axes[0].set_ylabel("imagined", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    caption = [
        "Columns: time after the start step.",
        "Under each imagined frame: its mean pixel error against the real frame | the error of guessing that nothing moves (0-255, lower is better).",
        BEV_LEGEND,
    ]
    fig.text(0.5, 0.005, "\n".join(caption), ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    return fig


def save_figure(fig, path: Path) -> None:
    """Save as a PNG small enough for GitHub (upstream's pre-commit hook rejects files over 500 KB), without clipping."""
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"optimize": True})
    plt.close(fig)
