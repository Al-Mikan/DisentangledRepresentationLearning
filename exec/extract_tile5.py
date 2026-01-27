#!/usr/bin/env python3
"""
Extract k evenly spaced frames from a variable-length MP4,
tile them horizontally, and save as exec/frame/{video_name}.png.

Usage:
  python extract_tile5.py input.mp4
  python extract_tile5.py input.mp4 --height 240 --k 5
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
    video_path: Path,
    k: int = 5,
    target_height: int = 240,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _evenly_spaced_indices(
        n_frames if n_frames > 0 else 1_000_000_000, k
    )

    frames: List[np.ndarray] = []
    last_good: Optional[np.ndarray] = None

    for idx in indices:
        fr = _read_frame_at(cap, idx)

        if fr is None:
            ok, fr2 = cap.read()
            fr = fr2 if ok else None

        if fr is None and last_good is not None:
            fr = last_good.copy()

        if fr is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr2 = cap.read()
            if not ok or fr2 is None:
                cap.release()
                raise RuntimeError("Could not read any frame from the video.")
            fr = fr2

        last_good = fr
        frames.append(fr)

    cap.release()

    # Resize to fixed height
    resized: List[np.ndarray] = []
    for fr in frames:
        h, w = fr.shape[:2]
        new_w = int(round(w * (target_height / h)))
        fr_rs = cv2.resize(fr, (new_w, target_height), interpolation=cv2.INTER_AREA)
        resized.append(fr_rs)

    tiled = np.concatenate(resized, axis=1)

    # ===== output path: exec/frame/{video_name}.png =====
    out_dir = Path("exec/frame")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{video_path.stem}.png"

    if not cv2.imwrite(str(out_path), tiled):
        raise RuntimeError(f"Failed to write image: {out_path}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_mp4", type=Path, help="Path to input mp4")
    ap.add_argument("--height", type=int, default=240, help="Tile height (px)")
    ap.add_argument("--k", type=int, default=5, help="Number of frames to sample")
    args = ap.parse_args()

    out = extract_and_tile(
        args.input_mp4,
        k=args.k,
        target_height=args.height,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
