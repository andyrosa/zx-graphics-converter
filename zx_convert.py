"""
Convert an image to ZX Spectrum resolution (256x192) and color palette (8 colors with 2 brightness levels, minus black which is the same at both) at 32 x 24.
Supports either perceptual (CIEDE2000 in Lab) or simple RGB distance per image.
"""

from PIL import Image
import os
import argparse

# ZX Spectrum palette (RGB values)
# Normal brightness (0) and bright (1) versions
ZX_PALETTE = [
    (0, 0, 0),       # 0: Black
    (0, 0, 215),     # 1: Blue
    (215, 0, 0),     # 2: Red
    (215, 0, 215),   # 3: Magenta
    (0, 215, 0),     # 4: Green
    (0, 215, 215),   # 5: Cyan
    (215, 215, 0),   # 6: Yellow
    (215, 215, 215), # 7: White
    # Bright versions
    (0, 0, 0),       # 8: Black (same as normal)
    (0, 0, 255),     # 9: Bright Blue
    (255, 0, 0),     # 10: Bright Red
    (255, 0, 255),   # 11: Bright Magenta
    (0, 255, 0),     # 12: Bright Green
    (0, 255, 255),   # 13: Bright Cyan
    (255, 255, 0),   # 14: Bright Yellow
    (255, 255, 255), # 15: Bright White
]

ZX_WIDTH = 256
ZX_HEIGHT = 192
ATTR_BLOCK_SIZE = 8  # Color attributes applied in 8x8 blocks


def rgb_squared_distance(c1, c2):
    """Squared Euclidean distance in RGB."""
    return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2


def perceptual_distance(lab1, lab2):
    """Wrapper to make swapping distance metrics easier."""
    return delta_e_ciede2000(lab1, lab2)


def delta_e_ciede2000(lab1, lab2):
    """Compute CIEDE2000 color difference between two Lab colors.

    Implementation based on the formula from the CIEDE2000 standard (approximate).
    Expects lab tuples (L, a, b).
    Returns a non-negative float; lower values mean more similar.
    """
    import math

    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1: Compute C' and h'
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1 - math.sqrt((C_bar ** 7) / (C_bar ** 7 + 25 ** 7))) if C_bar != 0 else 0
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)

    def _hp(a_prime, b):
        if a_prime == 0 and b == 0:
            return 0.0
        angle = math.degrees(math.atan2(b, a_prime))
        return angle + 360 if angle < 0 else angle

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)

    # Step 2: Delta L', Delta C', Delta H'
    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        if dhp > 180:
            dhp -= 360
        elif dhp < -180:
            dhp += 360

    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    # Step 3: Calculate weighting functions
    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0

    # Compute h_bar
    if C1p * C2p == 0:
        hp_bar = h1p + h2p
    else:
        hp_bar = h1p + h2p
        if abs(h1p - h2p) > 180:
            hp_bar += 360 if (h1p + h2p) < 360 else -360
        hp_bar /= 2.0

    T = 1 - 0.17 * math.cos(math.radians(hp_bar - 30)) + 0.24 * math.cos(math.radians(2 * hp_bar)) + 0.32 * math.cos(math.radians(3 * hp_bar + 6)) - 0.20 * math.cos(math.radians(4 * hp_bar - 63))

    delta_ro = 30 * math.exp(-((hp_bar - 275) / 25) ** 2)
    Rc = 2 * math.sqrt((Cp_bar ** 7) / (Cp_bar ** 7 + 25 ** 7))

    Sl = 1 + ((0.015 * ((Lp_bar - 50) ** 2)) / math.sqrt(20 + ((Lp_bar - 50) ** 2)))
    Sc = 1 + 0.045 * Cp_bar
    Sh = 1 + 0.015 * Cp_bar * T

    Rt = -math.sin(math.radians(2 * delta_ro)) * Rc

    # Step 4: Combine
    kL = kC = kH = 1.0
    dE = math.sqrt((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2 + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))

    return dE


def _rgb_to_linear(c):
    # convert sRGB channel (0..1) to linear RGB
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb):
    """Convert an (R,G,B) tuple (0-255) to CIE L*a*b* tuple.

    Returns (L, a, b) where L in 0..100 (approximately), a/b roughly -128..127.
    """
    # sRGB [0,255] -> [0,1]
    r, g, b = [v / 255.0 for v in rgb]

    # Convert to linear RGB
    r_lin = _rgb_to_linear(r)
    g_lin = _rgb_to_linear(g)
    b_lin = _rgb_to_linear(b)

    # Linear RGB to XYZ (D65)
    # Using the sRGB color space conversion matrix
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # Scale to the reference white
    # Reference white D65
    xr = x / 0.95047
    yr = y / 1.00000
    zr = z / 1.08883

    def f(t):
        if t > 0.008856:
            return t ** (1.0 / 3.0)
        return (7.787 * t) + (16.0 / 116.0)

    fx = f(xr)
    fy = f(yr)
    fz = f(zr)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return (L, a, b)


# Precompute Lab representations of the ZX palette for speed
ZX_PALETTE_LAB = [rgb_to_lab(c) for c in ZX_PALETTE]


def find_nearest_zx_color(rgb):
    """Find the nearest ZX Spectrum color to the given RGB value."""
    rgb_lab = rgb_to_lab(rgb)
    min_dist = float('inf')
    nearest_idx = 0
    for idx, zx_lab in enumerate(ZX_PALETTE_LAB):
        dist = perceptual_distance(rgb_lab, zx_lab)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx
    return nearest_idx


def find_best_two_colors_for_block(block_rgbs, block_labs, distance_mode):
    """
    Find the best two ZX Spectrum colors for an 8x8 block.
    Both colors must be from the same brightness level.
    Returns (ink_color_idx, paper_color_idx, is_bright)
    """
    best_error = float('inf')
    best_ink = 0
    best_paper = 0
    best_bright = False

    # Try normal brightness (indices 0-7) and bright (indices 8-15)
    for bright in [False, True]:
        offset = 8 if bright else 0
        color_range = range(offset, offset + 8)

        # Try all pairs of colors in this brightness level
        for ink_idx in color_range:
            ink_lab = ZX_PALETTE_LAB[ink_idx]
            ink_rgb = ZX_PALETTE[ink_idx]
            for paper_idx in color_range:
                paper_lab = ZX_PALETTE_LAB[paper_idx]
                paper_rgb = ZX_PALETTE[paper_idx]
                total_error = 0
                if distance_mode == 'cie':
                    for lab in block_labs:
                        # For each pixel, pick whichever of ink/paper is closer in Lab space
                        dist_ink = perceptual_distance(lab, ink_lab)
                        dist_paper = perceptual_distance(lab, paper_lab)
                        total_error += min(dist_ink, dist_paper)
                else:
                    for rgb in block_rgbs:
                        dist_ink = rgb_squared_distance(rgb, ink_rgb)
                        dist_paper = rgb_squared_distance(rgb, paper_rgb)
                        total_error += min(dist_ink, dist_paper)

                if total_error < best_error:
                    best_error = total_error
                    best_ink = ink_idx
                    best_paper = paper_idx
                    best_bright = bright
    
    return best_ink, best_paper, best_bright


def convert_to_zx_spectrum(input_path, output_path, distance_mode='cie'):
    """
    Convert an image to ZX Spectrum resolution and colors with attribute blocks.
    distance_mode: 'cie' for CIEDE2000 in Lab, 'rgb' for raw RGB distance.
    """
    img = Image.open(input_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to ZX Spectrum resolution
    img_resized = img.resize((ZX_WIDTH, ZX_HEIGHT), Image.Resampling.LANCZOS)
    
    # Create output image
    output = Image.new('RGB', (ZX_WIDTH, ZX_HEIGHT))
    
    # Process each 8x8 attribute block
    for block_y in range(ZX_HEIGHT // ATTR_BLOCK_SIZE):
        for block_x in range(ZX_WIDTH // ATTR_BLOCK_SIZE):
            # Collect all pixels in this block
            block_rgbs = []
            block_labs = []
            for py in range(ATTR_BLOCK_SIZE):
                for px in range(ATTR_BLOCK_SIZE):
                    x = block_x * ATTR_BLOCK_SIZE + px
                    y = block_y * ATTR_BLOCK_SIZE + py
                    pixel = img_resized.getpixel((x, y))
                    block_rgbs.append(pixel)
                    if distance_mode == 'cie':
                        block_labs.append(rgb_to_lab(pixel))

            # Find best two colors for this block
            ink_idx, paper_idx, _ = find_best_two_colors_for_block(block_rgbs, block_labs, distance_mode)
            ink_color = ZX_PALETTE[ink_idx]
            paper_color = ZX_PALETTE[paper_idx]
            ink_lab = ZX_PALETTE_LAB[ink_idx]
            paper_lab = ZX_PALETTE_LAB[paper_idx]

            # Apply colors to block - each pixel gets whichever of ink/paper is closer
            idx = 0
            for py in range(ATTR_BLOCK_SIZE):
                for px in range(ATTR_BLOCK_SIZE):
                    x = block_x * ATTR_BLOCK_SIZE + px
                    y = block_y * ATTR_BLOCK_SIZE + py
                    if distance_mode == 'cie':
                        lab = block_labs[idx]
                        dist_ink = perceptual_distance(lab, ink_lab)
                        dist_paper = perceptual_distance(lab, paper_lab)
                    else:
                        rgb = block_rgbs[idx]
                        dist_ink = rgb_squared_distance(rgb, ink_color)
                        dist_paper = rgb_squared_distance(rgb, paper_color)
                    new_color = ink_color if dist_ink < dist_paper else paper_color
                    output.putpixel((x, y), new_color)
                    idx += 1

    output.save(output_path)
    print(f"Converted image saved to: {output_path}")
    print(f"Resolution: {ZX_WIDTH}x{ZX_HEIGHT}")
    

if __name__ == '__main__':
    import glob

    parser = argparse.ArgumentParser(description="Convert images to ZX Spectrum palette.")
    parser.add_argument("input", nargs="?", help="Input image file")
    parser.add_argument("output", nargs="?", help="Output path (used only when mode is not 'both')")
    parser.add_argument("--mode", choices=["cie", "rgb", "both"], default="both", help="Distance metric to use")
    args = parser.parse_args()

    def build_outputs(base_path, ext, mode, explicit_out=None):
        if mode == "both":
            base = os.path.splitext(explicit_out)[0] if explicit_out else base_path
            return [
                (f"{base}_zx_cie{ext}", "cie"),
                (f"{base}_zx_rgb{ext}", "rgb"),
            ]
        else:
            target = explicit_out if explicit_out else f"{base_path}_zx_{mode}{ext}"
            return [(target, mode)]

    if args.input:
        input_file = args.input
        base, ext = os.path.splitext(args.output if args.output else input_file)
        outputs = build_outputs(base, ext, args.mode, explicit_out=args.output)
        for output_file, mode in outputs:
            convert_to_zx_spectrum(input_file, output_file, distance_mode=mode)
    else:
        # Batch mode - convert all jpg and webp files without existing _zx outputs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        patterns = ['*.jpg', '*.jpeg', '*.webp']
        for pattern in patterns:
            for input_file in glob.glob(os.path.join(script_dir, pattern)):
                base, ext = os.path.splitext(input_file)
                if base.endswith('_zx') or base.endswith('_zx_cie') or base.endswith('_zx_rgb'):
                    continue
                print(f"Converting: {os.path.basename(input_file)}")
                outputs = build_outputs(base, ext, args.mode)
                for output_file, mode in outputs:
                    convert_to_zx_spectrum(input_file, output_file, distance_mode=mode)
