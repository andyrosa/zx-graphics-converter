"""
Convert an image to ZX Spectrum resolution and color palette 
"""

from PIL import Image
import numpy as np
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


def _rgb_to_linear(channel_value):
    # convert sRGB channel (0..1) to linear RGB
    if channel_value <= 0.04045:
        return channel_value / 12.92
    return ((channel_value + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb):
    """Convert an (R,G,B) tuple (0-255) to CIE L*a*b* tuple.

    Returns (L, a, b) where L in 0..100 (approximately), a/b roughly -128..127.
    """
    # sRGB [0,255] -> [0,1]
    r, g, b = [value / 255.0 for value in rgb]

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
ZX_PALETTE_LAB = [rgb_to_lab(color) for color in ZX_PALETTE]

# NumPy arrays for vectorized operations
ZX_PALETTE_RGB_ARRAY = np.array(ZX_PALETTE, dtype=np.float32)
ZX_PALETTE_LAB_ARRAY = np.array(ZX_PALETTE_LAB, dtype=np.float32)




def delta_e_ciede2000_block(lab_block, reference_lab_color):
    """
    Vectorized CIEDE2000 between many Lab pixels and a single Lab palette color.

    lab_block: numpy array (num_pixels, 3)
    reference_lab_color: iterable of length 3 (L, a, b)
    returns: numpy array (num_pixels,) of ΔE00 distances
    """
    L1 = lab_block[:, 0]
    a1 = lab_block[:, 1]
    b1 = lab_block[:, 2]

    L2 = float(reference_lab_color[0])
    a2 = float(reference_lab_color[1])
    b2 = float(reference_lab_color[2])

    # Step 1: chroma
    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_bar = (C1 + C2) / 2.0

    C_bar_pow7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar_pow7 / (C_bar_pow7 + 25.0 ** 7)))

    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    C2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)

    # Step 2: hue angles
    h1_prime = np.degrees(np.arctan2(b1, a1_prime))
    h1_prime = np.where(h1_prime < 0.0, h1_prime + 360.0, h1_prime)

    h2_prime = np.degrees(np.arctan2(b2, a2_prime))
    h2_prime = np.where(h2_prime < 0.0, h2_prime + 360.0, h2_prime)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    delta_h_prime = h2_prime - h1_prime
    zero_chroma_mask = (C1_prime * C2_prime) == 0.0
    delta_h_prime = np.where(zero_chroma_mask, 0.0, delta_h_prime)
    delta_h_prime = np.where(delta_h_prime > 180.0, delta_h_prime - 360.0, delta_h_prime)
    delta_h_prime = np.where(delta_h_prime < -180.0, delta_h_prime + 360.0, delta_h_prime)

    delta_H_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(
        np.radians(delta_h_prime / 2.0)
    )

    # Step 3: means and weighting
    L_prime_bar = (L1 + L2) / 2.0
    C_prime_bar = (C1_prime + C2_prime) / 2.0

    hue_sum = h1_prime + h2_prime
    hue_difference = np.abs(h1_prime - h2_prime)
    non_zero_chroma_mask = ~zero_chroma_mask

    h_prime_bar = hue_sum.copy()
    adjust_mask = non_zero_chroma_mask & (hue_difference > 180.0)

    h_prime_bar = np.where(
        adjust_mask & (hue_sum < 360.0),
        h_prime_bar + 360.0,
        h_prime_bar,
    )
    h_prime_bar = np.where(
        adjust_mask & (hue_sum >= 360.0),
        h_prime_bar - 360.0,
        h_prime_bar,
    )
    h_prime_bar = np.where(
        non_zero_chroma_mask,
        h_prime_bar / 2.0,
        h_prime_bar,
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_prime_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_prime_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_prime_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_prime_bar - 63.0))
    )

    delta_theta = 30.0 * np.exp(-((h_prime_bar - 275.0) / 25.0) ** 2)
    R_C = 2.0 * np.sqrt((C_prime_bar ** 7) / (C_prime_bar ** 7 + 25.0 ** 7))

    S_L = 1.0 + (0.015 * (L_prime_bar - 50.0) ** 2) / np.sqrt(
        20.0 + (L_prime_bar - 50.0) ** 2
    )
    S_C = 1.0 + 0.045 * C_prime_bar
    S_H = 1.0 + 0.015 * C_prime_bar * T

    R_T = -np.sin(2.0 * np.radians(delta_theta)) * R_C

    delta_E = np.sqrt(
        (delta_L_prime / S_L) ** 2
        + (delta_C_prime / S_C) ** 2
        + (delta_H_prime / S_H) ** 2
        + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )

    return delta_E


def get_block_distances(block_pixels, mode):
    """
    Calculate distances from every pixel in the block to every palette color.
    Returns: numpy array (num_pixels, 16)
    """
    num_pixels = block_pixels.shape[0]

    if mode == "cie":
        distances = np.zeros((num_pixels, 16), dtype=np.float32)
        # Convert block to Lab
        lab_block = np.empty((num_pixels, 3), dtype=np.float32)
        for i in range(num_pixels):
            lab_block[i] = rgb_to_lab(tuple(block_pixels[i]))

        for i in range(16):
            distances[:, i] = delta_e_ciede2000_block(lab_block, ZX_PALETTE_LAB_ARRAY[i])
        return distances
    else:
        # RGB: Squared Euclidean distance
        # block: (N, 3), palette: (16, 3)
        # (N, 1, 3) - (1, 16, 3) -> (N, 16, 3)
        diff = (
            block_pixels[:, np.newaxis, :].astype(np.float32)
            - ZX_PALETTE_RGB_ARRAY[np.newaxis, :, :]
        )
        return np.sum(diff * diff, axis=2)


def process_8x8_block(block_pixels, mode):
    """
    Find best 2 colors for the block and return the quantized pixels.
    """
    distances = get_block_distances(block_pixels, mode)

    best_error = np.inf
    best_c1, best_c2 = 0, 0

    # Iterate over the two brightness groups (0-7 and 8-15)
    for group_start in (0, 8):
        # Extract distances for this group: (64, 8)
        group_dists = distances[:, group_start : group_start + 8]

        # Brute force all pairs in this group
        for i in range(8):
            d1 = group_dists[:, i]
            # Optimization: j starts at i (colors can be same)
            for j in range(i, 8):
                d2 = group_dists[:, j]
                # Error is sum of min distance for each pixel
                total_error = np.sum(np.minimum(d1, d2))

                if total_error < best_error:
                    best_error = total_error
                    best_c1 = group_start + i
                    best_c2 = group_start + j

    # Reconstruct the block using the best two colors
    mask = distances[:, best_c1] <= distances[:, best_c2]

    c1 = ZX_PALETTE_RGB_ARRAY[best_c1]
    c2 = ZX_PALETTE_RGB_ARRAY[best_c2]

    # (64, 3)
    result = np.where(mask[:, np.newaxis], c1, c2)
    return result.astype(np.uint8)


def convert_to_zx_spectrum(input_path, output_path, distance_mode="cie"):
    """
    Convert an image to ZX Spectrum resolution and colors with attribute blocks.

    Uses NumPy-accelerated brute-force search over palette colors, per brightness group.
    For each 8x8 block, finds the best 2-color approximation by minimizing
    total error across all 64 pixels.

    distance_mode: 'cie' for linear CIEDE2000, 'rgb' for squared RGB.
    """
    img = Image.open(input_path)

    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to ZX Spectrum resolution
    img_resized = img.resize((ZX_WIDTH, ZX_HEIGHT), Image.Resampling.LANCZOS)

    # Convert to numpy array for fast processing
    img_array = np.array(img_resized, dtype=np.uint8)
    output_array = np.zeros_like(img_array)

    # Process each 8x8 attribute block
    blocks_x = ZX_WIDTH // ATTR_BLOCK_SIZE
    blocks_y = ZX_HEIGHT // ATTR_BLOCK_SIZE

    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            # Extract block pixels
            y_start = block_y * ATTR_BLOCK_SIZE
            y_end = y_start + ATTR_BLOCK_SIZE
            x_start = block_x * ATTR_BLOCK_SIZE
            x_end = x_start + ATTR_BLOCK_SIZE

            block = img_array[y_start:y_end, x_start:x_end]
            block_pixels = block.reshape(64, 3)

            # Process block
            result_pixels = process_8x8_block(block_pixels, distance_mode)

            # Write back to output
            output_array[y_start:y_end, x_start:x_end] = result_pixels.reshape(8, 8, 3)

    # Save output
    output_img = Image.fromarray(output_array)
    output_img.save(output_path)
    file_size = os.path.getsize(output_path)
    print(f"Saved: {output_path} ({file_size:,} bytes)")


if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser(description="Convert images to ZX Spectrum palette.")
    parser.add_argument("input", nargs="?", help="Input image file")
    parser.add_argument(
        "output", nargs="?", help="Output path (used only when mode is not 'both')"
    )
    parser.add_argument(
        "--mode",
        choices=["cie", "rgb", "both"],
        default="both",
        help="Distance metric to use",
    )
    args = parser.parse_args()

    def build_outputs(base_path, ext, mode, explicit_out=None):
        if mode == "both":
            base = os.path.splitext(explicit_out)[0] if explicit_out else base_path
            return [
                (f"{base}_zx_cie{ext}", "cie"),
                (f"{base}_zx_rgb{ext}", "rgb"),
            ]
        else:
            target = (
                explicit_out
                if explicit_out
                else f"{base_path}_zx_{mode}{ext}"
            )
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
        patterns = ["*.jpg", "*.jpeg", "*.webp"]
        for pattern in patterns:
            for input_file in glob.glob(os.path.join(script_dir, pattern)):
                base, ext = os.path.splitext(input_file)
                if (
                    base.endswith("_zx")
                    or base.endswith("_zx_cie")
                    or base.endswith("_zx_rgb")
                ):
                    continue
                print(f"Converting: {os.path.basename(input_file)}")
                outputs = build_outputs(base, ext, args.mode)
                for output_file, mode in outputs:
                    convert_to_zx_spectrum(input_file, output_file, distance_mode=mode)
