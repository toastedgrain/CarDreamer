"""
Figures as plain arrays, written as PNG with the standard library only, so no image package is needed.
"""

import struct
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

WHITE = 255
ROW_GAP = 2  # pixels between a real row and its imagined row
BLOCK_GAP = 8  # pixels between start steps


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


def real_vs_imagined_grid(comparisons: Sequence, every: int) -> np.ndarray:
    """
    One block per comparison: its real BEVs in the top row, the imagined ones below, every `every`-th step of the
    horizon from the `every`-th on. Blocks are stacked top to bottom, separated by white bands.

    :param comparisons: objects with `real` and `imagined`, both (H, 128, 128, 3) uint8, e.g. ImaginationComparison.
    """
    if not comparisons:
        raise ValueError("nothing to draw")
    shown = slice(every - 1, None, every)
    blocks = []
    for comparison in comparisons:
        real_row = np.concatenate(list(comparison.real[shown]), axis=1)
        imagined_row = np.concatenate(list(comparison.imagined[shown]), axis=1)
        line = np.full((ROW_GAP, real_row.shape[1], 3), WHITE, np.uint8)
        blocks.append(np.concatenate([real_row, line, imagined_row], axis=0))
    gap = np.full((BLOCK_GAP, blocks[0].shape[1], 3), WHITE, np.uint8)
    parts = [part for block in blocks for part in (block, gap)][:-1]
    return np.concatenate(parts, axis=0)
