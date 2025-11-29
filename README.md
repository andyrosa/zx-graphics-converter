ZX Graphics Converter
=====================

Small tool to downscale images to ZX Spectrum resolution (256x192) and palette. Each 8x8 attribute block is limited to two colors of the same brightness level, matching Spectrum hardware.

Features
- Palette-accurate ZX rendering with attribute blocks.
- Two distance modes: perceptual (Lab/ΔE2000) and simple RGB.
- Default dual-output mode to compare both metrics.
- Batch conversion for common image types (jpg/jpeg/webp).

Requirements
- Python 3.10+
- Pillow (`pip install pillow`)

Usage
```
python zx_convert.py <input_image> [output_image] [--mode cie|rgb|both]
```
- `--mode cie` (default for single output) uses Lab + ΔE2000.
- `--mode rgb` uses squared Euclidean distance in RGB.
- `--mode both` (default when no output is specified) writes two files: `<base>_zx_cie` and `<base>_zx_rgb`.
- If `output_image` is omitted, names are derived from the input (with mode suffixes).

Batch conversion
```
python zx_convert.py --mode both
```
With no positional args, all `*.jpg`, `*.jpeg`, and `*.webp` files in the script directory are processed, skipping any that already have `_zx`, `_zx_cie`, or `_zx_rgb` in the name.

How it works
- Images are resized to 256x192 (Lanczos).
- For each 8x8 block, the tool chooses the best two palette colors within the same brightness (normal or bright) using the selected distance metric.
- Each pixel in the block is assigned to whichever of the two chosen colors is closer under that metric.

Notes
- ΔE2000 generally preserves perceived hue/brightness relationships better than RGB distance; use RGB if you want to see the naïve comparison.
- Outputs are standard image files using the input extension (jpg/webp, etc.) with `_zx_*` suffixes.
- RGB looks better; the algorithm should be more global.