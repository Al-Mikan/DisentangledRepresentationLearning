#!/usr/bin/env python3
"""
Extract 5 evenly spaced frames from a variable-length MP4,
tile them horizontally, and save as a single PNG.

Usage:
  python extract_tile5.py input.mp4 output.png
  python extract_tile5.py input.mp4 output.png --height 240
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def _evenly_spaced_indices(n_frames: int, k: int) -> List[int]:
    """Return k indices evenly spaced over [0, n_frames-1]."""
    if n_frames <= 0:
        return [0] * k
    if k <= 1:
        return [max(0, n_frames - 1)]
    # Use linspace so first=0 and last=n_frames-1
    idx = np.linspace(0, n_frames - 1, num=k)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, n_frames - 1)
    return idx.tolist()


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def extract_and_tile(
    video_path: str,
    out_path: str,
    k: int = 5,
    target_height: int = 240,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # may be 0 for some codecs
    indices = _evenly_spaced_indices(n_frames if n_frames > 0 else 1_000_000_000, k)

    frames: List[np.ndarray] = []
    last_good: Optional[np.ndarray] = None

    # If CAP_PROP_FRAME_COUNT is unreliable, we still try indices;
    # missing reads are replaced by last successful frame.
    for idx in indices:
        fr = _read_frame_at(cap, idx)
        if fr is None:
            # fallback: try sequential read from current position
            ok, fr2 = cap.read()
            fr = fr2 if ok else None

        if fr is None and last_good is not None:
            fr = last_good.copy()

        if fr is None:
            # As a last resort, rewind and grab first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr2 = cap.read()
            if not ok or fr2 is None:
                cap.release()
                raise RuntimeError("Could not read any frame from the video.")
            fr = fr2

        last_good = fr
        frames.append(fr)

    cap.release()

    # Resize all frames to the same height, preserving aspect ratio
    resized: List[np.ndarray] = []
    for fr in frames:
        h, w = fr.shape[:2]
        if h == 0 or w == 0:
            raise RuntimeError("Encountered an empty frame.")
        new_w = int(round(w * (target_height / h)))
        fr_rs = cv2.resize(fr, (new_w, target_height), interpolation=cv2.INTER_AREA)
        resized.append(fr_rs)

    tiled = np.concatenate(resized, axis=1)

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(out_path, tiled)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_mp4", help="Path to input mp4")
    ap.add_argument("output_png", help="Path to output png")
    ap.add_argument("--height", type=int, default=240, help="Tile height (px). Default: 240")
    ap.add_argument("--k", type=int, default=5, help="Number of frames to sample. Default: 5")
    args = ap.parse_args()

    extract_and_tile(args.input_mp4, args.output_png, k=args.k, target_height=args.height)


if __name__ == "__main__":
    main()
