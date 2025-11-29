ZX Graphics Converter
=====================

Converts images to ZX Spectrum resolution (256×192) and palette. Each 8×8 attribute block is limited to two colors from the same brightness level (normal or bright), matching Spectrum hardware constraints.

Algorithm
---------
1. Resize image to 256×192 using Lanczos downsampling.
2. Split into 8×8 blocks (32×24 = 768 blocks total).
3. For each block, brute-force search all color pairs within each brightness group (normal 0-7, bright 8-15) to find the pair that minimizes total error across all 64 pixels.
4. Assign each pixel to whichever of the two chosen colors is closer.

This guarantees the optimal 2-color approximation for each block under the selected distance metric.

Distance Modes
--------------
- `cie`: Linear CIEDE2000 (ΔE00) in Lab color space. Perceptually uniform.
- `rgb`: Squared Euclidean distance in RGB. Faster.

Requirements
------------
- Python 3.10+
- Pillow, NumPy (`pip install pillow numpy`)

Usage
-----
```
python zx_convert.py [input_image] [output_image] [--mode cie|rgb|both]
```

Examples:
```
python zx_convert.py photo.jpg                    # outputs photo_zx_cie.jpg and photo_zx_rgb.jpg
python zx_convert.py photo.jpg --mode rgb         # outputs photo_zx_rgb.jpg only
python zx_convert.py photo.jpg out.jpg --mode rgb # outputs out.jpg
python zx_convert.py                              # batch mode: all jpg/jpeg/webp in script directory
```

Batch mode skips files with `_zx`, `_zx_cie`, or `_zx_rgb` suffixes.

Notes
-----
- RGB mode often produces visually better results because squared distance penalizes large errors more heavily.
- Output uses the input file's extension (jpg, webp, etc.) with `_zx_*` suffix.
- The 15-color ZX palette has 8 normal and 8 bright colors (black appears in both).