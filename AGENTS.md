# AGENTS.md

Context for AI coding agents (and humans) working on this project.

## What this is

`magnadoodle` is a small Python tool that digitizes photographs of children's [Magna Doodle](https://en.wikipedia.org/wiki/Magna_Doodle) drawings into clean black-and-white PNG and SVG files. The central challenge is the honeycomb hex-cell pattern visible through the drawing surface — a fine periodic structure that generic raster-to-vector tools (Adobe Capture, Inkscape's Trace Bitmap, vectorizer.ai) cannot suppress, so they render the grid right along with the drawing.

## Goals and non-goals

**Goals.** Turn a typical phone photo of a Magna Doodle into a clean, pattern-free SVG suitable for printing, scaling, or archiving. Handle the usual photo artifacts: uneven lighting, hand shadows, JPEG compression, the colored plastic frame around the drawing surface. Run on a laptop in under a second per image. Require zero manual preprocessing for typical cases.

**Non-goals.** Perspective rectification beyond what a roughly-square handheld photo provides. Color or grayscale preservation (the drawing is monochrome by construction). Real-time mobile capture. Preserving stroke pressure or temporal order (the toy records neither).

## Layout

```
magnadoodle.py      # single-file module with the pipeline and CLI
pyproject.toml      # uv / hatchling project config
AGENTS.md           # this file
```

That's the whole project. Keep it flat.

## Pipeline

The core insight: the honeycomb is periodic at one spatial scale (the cell pitch, typically 15–30 px in a phone photo), while the drawing varies at a much larger scale. We separate them by filtering.

1. **Auto-crop** to the drawing surface (`auto_crop()`) — largest bright connected component after heavy Gaussian blur + Otsu. The Magna Doodle's off-white drawing area is substantially brighter than its colored frame.
2. **Estimate cell pitch** (`estimate_cell_pitch()`) — radial average of the FFT magnitude spectrum; the hex grid shows up as a dominant ring. A Hann window reduces edge artifacts, and a smoothed-baseline subtraction finds the peak robustly even on noisy photos.
3. **Smooth out the hex grid** — Gaussian blur with σ ≈ 0.45 × cell pitch. Big enough to average across neighboring cells, small enough to preserve drawing edges.
4. **Flatten background** — subtract a morphological close with a large elliptical kernel to kill shadows and vignettes. The close is applied to the already-smoothed image so the drawing becomes bright on a dark field.
5. **Threshold** — Otsu, which works cleanly here because the flattened histogram is bimodal.
6. **Cleanup** — morphological open with a 3×3 ellipse, then a connected-components filter dropping anything smaller than ~(0.7 × cell pitch)².
7. **Vectorize** — shell out to `potrace` with `-s -t 4 -O 0.4 -a 1.0`. We shell out rather than use pypotrace because it's one fewer thing to install and the CLI interface is stable.

Design alternatives we evaluated and rejected:

- **Median filter.** Effective at killing the hex pattern but erodes thin features (eyes, dots on "i"s, thin smile lines).
- **FFT notch filter.** Produces visible ringing (Gibbs phenomenon) from hard-edged notches. Would be viable with Gaussian-shaped notches — noted as a future direction below.
- **Adaptive thresholding without flattening.** Introduces block artifacts that look worse than Otsu on the flattened image.

## Setup and dev

```bash
uv sync                                 # creates .venv, installs deps, builds package
uv run magnadoodle photo.jpg            # run the CLI
uv run magnadoodle photo.jpg --debug    # also write intermediate stages to *_debug/
```

System dependency: `potrace` (for SVG output). Install via `apt install potrace`, `brew install potrace`, or `pacman -S potrace`. If missing, pass `--no-svg`.

## Testing

There's no formal test suite. For regression testing, generate a synthetic Magna Doodle image with a small fixture script that:

- renders a flat-top hex grid (circumradius ~11 px) as faint polylines on a light background,
- draws a simple face (ellipse head, circle eye with pupil, nose, arc smile) with `cv2.ellipse`/`cv2.circle`,
- adds a soft Gaussian shadow blob and a vertical gradient to mimic handheld lighting,
- adds mild Gaussian noise and re-encodes as JPEG at quality 85.

The synthetic image is a faithful stand-in for real photos: same hex topology, similar stroke-to-cell size ratios, same lighting characteristics. When changing any stage in the pipeline, verify the synthetic result first before real photos — it's fast to iterate on and catches regressions cleanly.

For real photos, `--debug` writes `01_gray.png` through `05_mask.png`. The two to scrutinize are `04_flat.png` (should show the drawing as a clean bright shape on a dark field, hex pattern invisible) and `05_mask.png` (should be crisp binary with no speckle).

## Tuning

Parameters with good defaults but worth knowing about:

| Knob | Location | Effect |
|------|----------|--------|
| `--cell N` | CLI | Override cell pitch detection. Bump up if grid is leaking through; drop if thin features vanish. |
| `sigma = 0.45 * cell_pitch` | `digitize()` | Gaussian blur width. Scales with cell pitch. |
| `bg_kernel_frac = 0.05` | `digitize()` kwarg | Background morphology kernel as fraction of min(H, W). Larger = more aggressive shadow removal but may eat broad strokes. |
| `min_area = (cell * 0.7)**2` | `digitize()` | Drops specks smaller than this many pixels. Raise if there are spurious dots; lower if small intentional features (dots, nostril points) disappear. |
| potrace `-t 4` | `png_to_svg()` | "Turdsize" — drops paths enclosing fewer than N pixels. |
| potrace `-O 0.4 -a 1.0` | `png_to_svg()` | Curve optimization and corner threshold. Bump `-O` up for smoother curves, `-a` up for gentler corners. |

## Gotchas

- **Don't vendor potrace or switch to pypotrace.** pypotrace binds to a specific C library version that breaks on modern Python; the subprocess call is intentionally boring and has been stable since 2001.
- **Don't swap Otsu for adaptive thresholding.** We already did the flattening; adaptive thresholding on a flattened image introduces block artifacts.
- **`bg_size` has two floors:** `int(min_dim * 0.05) | 1` and `int(cell * 6) | 1`, whichever is larger. The `| 1` forces odd values (OpenCV kernels require odd). Don't remove either floor; the first protects against tiny inputs, the second against huge cell pitches.
- **Hatchling needs `only-include = ["magnadoodle.py"]`.** Without it, hatchling looks for a `magnadoodle/` package directory and fails. This is the single-file-module idiom.
- **The image is drawing-BLACK on WHITE at the end of `digitize()`.** `invert_output=True` produces scan-like output. potrace traces black-on-white, so keep this convention end-to-end.
- **`main()` returns `int`, not `None`.** The `[project.scripts]` entry point and `sys.exit(main())` both rely on this. Don't "simplify" to returning None.

## Possible future directions

In rough priority order:

1. **Batch processing** — glob-pattern input, parallelized over cores with `multiprocessing.Pool`. Natural fit for archiving a backlog of photos.
2. **Perspective correction** — detect the rectangular drawing frame (Canny + Hough or contour-based), warp to a rectangle before cropping. Would make handheld photos more forgiving.
3. **FFT notch path as a second mode** — for drawings with features thinner than a cell pitch. Use Gaussian-shaped notches to avoid the ringing we saw with hard-edged notches.
4. **Web or iOS Shortcut frontend** — photograph → upload → SVG returned. Worth doing only if this becomes a frequently-used tool.
5. **Skeleton-based stroke reconstruction** — skeletonize the mask and walk connected components as polylines. Useful if someone wants to animate the drawing being re-drawn, or export to a plotter.

Do not add color handling, stroke-width variation, or "pencil look" stylization. The Magna Doodle produces uniform-width monochrome strokes by physics; faking otherwise would misrepresent the source.
