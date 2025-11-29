"""
Convert an image to ZX Spectrum resolution (256x192) and color palette.
The ZX Spectrum has 15 colors (8 colors with 2 brightness levels, minus black which is the same at both).
"""

from PIL import Image
import sys
import os

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


def color_distance(c1, c2):
    """Calculate squared Euclidean distance between two RGB colors."""
    return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2


def find_nearest_zx_color(rgb):
    """Find the nearest ZX Spectrum color to the given RGB value."""
    min_dist = float('inf')
    nearest_idx = 0
    for idx, zx_color in enumerate(ZX_PALETTE):
        dist = color_distance(rgb, zx_color)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx
    return nearest_idx


def find_best_two_colors_for_block(pixels_in_block):
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
            for paper_idx in color_range:
                total_error = 0
                for pixel in pixels_in_block:
                    # For each pixel, pick whichever of ink/paper is closer
                    dist_ink = color_distance(pixel, ZX_PALETTE[ink_idx])
                    dist_paper = color_distance(pixel, ZX_PALETTE[paper_idx])
                    total_error += min(dist_ink, dist_paper)
                
                if total_error < best_error:
                    best_error = total_error
                    best_ink = ink_idx
                    best_paper = paper_idx
                    best_bright = bright
    
    return best_ink, best_paper, best_bright


def convert_to_zx_spectrum(input_path, output_path):
    """Convert an image to ZX Spectrum resolution and colors with attribute blocks."""
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
            pixels_in_block = []
            for py in range(ATTR_BLOCK_SIZE):
                for px in range(ATTR_BLOCK_SIZE):
                    x = block_x * ATTR_BLOCK_SIZE + px
                    y = block_y * ATTR_BLOCK_SIZE + py
                    pixels_in_block.append(img_resized.getpixel((x, y)))
            
            # Find best two colors for this block
            ink_idx, paper_idx, _ = find_best_two_colors_for_block(pixels_in_block)
            ink_color = ZX_PALETTE[ink_idx]
            paper_color = ZX_PALETTE[paper_idx]
            
            # Apply colors to block - each pixel gets whichever of ink/paper is closer
            for py in range(ATTR_BLOCK_SIZE):
                for px in range(ATTR_BLOCK_SIZE):
                    x = block_x * ATTR_BLOCK_SIZE + px
                    y = block_y * ATTR_BLOCK_SIZE + py
                    pixel = img_resized.getpixel((x, y))
                    
                    dist_ink = color_distance(pixel, ink_color)
                    dist_paper = color_distance(pixel, paper_color)
                    new_color = ink_color if dist_ink < dist_paper else paper_color
                    output.putpixel((x, y), new_color)
    
    output.save(output_path)
    print(f"Converted image saved to: {output_path}")
    print(f"Resolution: {ZX_WIDTH}x{ZX_HEIGHT}")
    print(f"Colors: ZX Spectrum 15-color palette")


if __name__ == '__main__':
    import glob
    
    if len(sys.argv) >= 2:
        # Single file mode
        input_file = sys.argv[1]
        if len(sys.argv) >= 3:
            output_file = sys.argv[2]
        else:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_zx{ext}"
        convert_to_zx_spectrum(input_file, output_file)
    else:
        # Batch mode - convert all jpg and webp files without _zx
        script_dir = os.path.dirname(os.path.abspath(__file__))
        patterns = ['*.jpg', '*.jpeg', '*.webp']
        
        for pattern in patterns:
            for input_file in glob.glob(os.path.join(script_dir, pattern)):
                base, ext = os.path.splitext(input_file)
                if base.endswith('_zx'):
                    continue
                output_file = f"{base}_zx{ext}"
                print(f"Converting: {os.path.basename(input_file)}")
                convert_to_zx_spectrum(input_file, output_file)
