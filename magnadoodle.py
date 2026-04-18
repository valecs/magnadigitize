#!/usr/bin/env python3
"""
magnadoodle.py — Digitize a Magna Doodle photo into a clean PNG and SVG.

The trick: the visible honeycomb is a periodic structure at a single spatial
scale (the cell pitch), while the drawing varies at a larger scale. A Gaussian
blur with sigma ~= cell pitch averages out the hex grid while leaving the
drawing intact. Subtracting a much heavier blur removes shadows / vignetting.
Otsu's threshold then gives a clean binary mask, which potrace converts to SVG.

Usage:
    python3 magnadoodle.py photo.jpg
    python3 magnadoodle.py photo.jpg --crop 200 300 1800 2400
    python3 magnadoodle.py photo.jpg --cell 12 --debug
    python3 magnadoodle.py photo.jpg --svg-only

Dependencies:
    pip install opencv-python numpy scipy
    apt install potrace                 # for SVG output

Author: written for Vale, April 2026
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


# ──────────────────────────────────────────────────────────────────────────────
# Cell-pitch detection via radial FFT
# ──────────────────────────────────────────────────────────────────────────────
def estimate_cell_pitch(gray: np.ndarray,
                        pitch_min: float = 8,
                        pitch_max: float = 35) -> float:
    """Estimate the hex-cell pitch (center-to-center distance) in pixels.

    The honeycomb produces a strong ring in the FFT magnitude at radius
    f = N / pitch, where N is image size. We find the dominant ring.
    """
    h, w = gray.shape
    s = min(h, w)
    crop = gray[(h - s) // 2:(h - s) // 2 + s,
                (w - s) // 2:(w - s) // 2 + s].astype(np.float32)
    crop -= crop.mean()
    win = np.outer(np.hanning(s), np.hanning(s))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(crop * win)))

    cy, cx = s // 2, s // 2
    yy, xx = np.indices(spec.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    r_max = s // 2
    radial = np.bincount(r.ravel(), spec.ravel())[:r_max] / (
             np.bincount(r.ravel())[:r_max] + 1e-9)

    f_lo = max(3, int(s / pitch_max))
    f_hi = min(r_max - 1, int(s / pitch_min))
    band = radial[f_lo:f_hi].copy()
    baseline = ndimage.gaussian_filter1d(band, sigma=8)
    peaks = band - baseline
    f_peak = f_lo + int(np.argmax(peaks))
    return s / f_peak


# ──────────────────────────────────────────────────────────────────────────────
# Find the drawing surface (low-saturation + bright region, eroded to peel
# off the silver metallic trim that otherwise leaks into the mask)
# ──────────────────────────────────────────────────────────────────────────────
def find_drawing_region(
    img_bgr: np.ndarray,
) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """Return (mask, bbox) for the drawing surface, or (None, full-frame).

    The Magna Doodle drawing area is uniquely low-saturation and bright. We
    threshold on blurred saturation + value, take the largest connected blob,
    then erode ~4% to pull inward past the silver trim and any shadow that
    bleeds onto the surface edge.
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    blur_s = cv2.GaussianBlur(s, (31, 31), 0)
    blur_v = cv2.GaussianBlur(v, (31, 31), 0)
    raw = ((blur_s < 60) & (blur_v > 100)).astype(np.uint8) * 255
    erode_px = max(8, int(min(H, W) * 0.04))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (erode_px * 2 + 1, erode_px * 2 + 1))
    eroded = cv2.erode(raw, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, 8)
    if n <= 1:
        return None, (0, 0, W, H)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == biggest).astype(np.uint8) * 255
    x = stats[biggest, cv2.CC_STAT_LEFT]
    y = stats[biggest, cv2.CC_STAT_TOP]
    w = stats[biggest, cv2.CC_STAT_WIDTH]
    h = stats[biggest, cv2.CC_STAT_HEIGHT]
    return region, (x, y, x + w, y + h)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def digitize(
    img_bgr: np.ndarray,
    cell_pitch: float | None = None,
    region: np.ndarray | None = None,
    invert_output: bool = True,
    debug_dir: Path | None = None,
) -> np.ndarray:
    """Return a clean uint8 binary image: drawing=0 on background=255
    (or inverted if invert_output=False).

    If `region` is provided (uint8 0/255 mask of the drawing surface), pixels
    outside it are neutralized before thresholding and the mask is AND-ed back
    against it at the end. Otsu's threshold is computed only over in-region
    pixels so the drawing's contrast isn't drowned out by the dark frame.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr

    if region is not None:
        # Replace out-of-region pixels with a bright surface-like value so the
        # blackhat response there is ~0 and doesn't bias anything.
        bright = int(np.percentile(gray[region > 0], 90))
        gray = np.where(region > 0, gray, bright).astype(np.uint8)

    if cell_pitch is None:
        cell_pitch = estimate_cell_pitch(gray)
        print(f"  detected cell pitch ≈ {cell_pitch:.1f} px")

    # 1. Gaussian blur averages out the hex grid (sigma ~ half a cell pitch).
    sigma = max(2.0, cell_pitch * 0.45)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)

    # 2. Black-hat: local bright-bg minus image pulls out dark strokes directly,
    #    without needing to estimate a global shadow model. Kernel is a few
    #    cells wide so it responds to strokes but ignores the drawing surface.
    bh_size = int(cell_pitch * 6) | 1
    bh = cv2.morphologyEx(
        smooth, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_size, bh_size)))
    flat = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Otsu on in-region pixels only (prevents the surrounding frame from
    #    dominating the histogram), then apply threshold globally.
    if region is not None:
        in_mask = region > 0
        thr_val, _ = cv2.threshold(flat[in_mask], 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(flat, int(thr_val), 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, region)
    else:
        _, mask = cv2.threshold(flat, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Morphological cleanup: drop tiny specks, close 1-2 px gaps.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = int((cell_pitch * 0.7) ** 2)
    cleaned = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    mask = cleaned

    if debug_dir:
        debug_dir.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(debug_dir / '01_gray.png'), gray)
        cv2.imwrite(str(debug_dir / '02_smooth.png'), smooth)
        cv2.imwrite(str(debug_dir / '04_flat.png'), flat)
        cv2.imwrite(str(debug_dir / '05_mask.png'), mask)
        if region is not None:
            cv2.imwrite(str(debug_dir / '00_region.png'), region)

    return mask if not invert_output else cv2.bitwise_not(mask)


# ──────────────────────────────────────────────────────────────────────────────
# SVG export via potrace
# ──────────────────────────────────────────────────────────────────────────────
def png_to_svg(png_path: Path, svg_path: Path, threshold: float = 0.5) -> None:
    """Convert a binary PNG to SVG using potrace.

    potrace expects a PBM/PGM/PPM file. We convert via mkbitmap-less route:
    just save as PGM then run potrace.
    """
    if shutil.which('potrace') is None:
        raise RuntimeError(
            "potrace not found. Install with: apt install potrace  (or brew install potrace)"
        )
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    pgm_path = png_path.with_suffix('.pgm')
    cv2.imwrite(str(pgm_path), img)
    # potrace flags:
    #   -s         SVG output
    #   -t 2       turdsize: drop blobs smaller than N pixels
    #   -O 0.4     curve optimization tolerance
    #   -a 1.0     corner threshold
    subprocess.run(
        ['potrace', '-s', '-t', '4', '-O', '0.4', '-a', '1.0',
         '-o', str(svg_path), str(pgm_path)],
        check=True,
    )
    pgm_path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', type=Path, help='Input photo of Magna Doodle')
    p.add_argument('-o', '--output', type=Path,
                   help='Output basename (default: <input>_clean)')
    p.add_argument('--crop', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'),
                   help='Manual crop box; default is auto-detect')
    p.add_argument('--no-crop', action='store_true', help='Skip auto-cropping')
    p.add_argument('--cell', type=float,
                   help='Hex cell pitch in px (default: auto-detect)')
    p.add_argument('--no-svg', action='store_true', help='Skip SVG export')
    p.add_argument('--svg-only', action='store_true', help='Only write the SVG')
    p.add_argument('--debug', action='store_true',
                   help='Write intermediate images to <output>_debug/')
    args = p.parse_args()

    if not args.image.exists():
        print(f"error: file not found: {args.image}", file=sys.stderr)
        return 1

    out_base = args.output or args.image.with_name(args.image.stem + '_clean')
    out_base = Path(out_base)

    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: could not read image: {args.image}", file=sys.stderr)
        return 1
    print(f"Loaded {args.image.name}: {img.shape[1]}×{img.shape[0]}")

    # Crop + drawing-surface mask
    if args.crop:
        x0, y0, x1, y1 = args.crop
        region = None
    elif args.no_crop:
        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
        region = None
    else:
        region_full, (x0, y0, x1, y1) = find_drawing_region(img)
        print(f"  auto-crop to ({x0},{y0})–({x1},{y1})")
        region = region_full[y0:y1, x0:x1] if region_full is not None else None
    cropped = img[y0:y1, x0:x1]

    debug_dir = out_base.with_name(out_base.name + '_debug') if args.debug else None
    clean = digitize(cropped, cell_pitch=args.cell, region=region,
                     debug_dir=debug_dir)

    png_path = out_base.with_suffix('.png')
    if not args.svg_only:
        cv2.imwrite(str(png_path), clean)
        print(f"  wrote {png_path}")

    if not args.no_svg:
        svg_path = out_base.with_suffix('.svg')
        # potrace traces black-on-white → invert our convention
        tmp_png = out_base.with_name(out_base.name + '_tmp.png')
        cv2.imwrite(str(tmp_png), clean)
        try:
            png_to_svg(tmp_png, svg_path)
            print(f"  wrote {svg_path}")
        finally:
            tmp_png.unlink(missing_ok=True)

    if args.svg_only and png_path.exists():
        png_path.unlink()

    return 0


if __name__ == '__main__':
    sys.exit(main())
