#!/usr/bin/env python3
"""
Script to create a demo image for README.md showing before/after document scanning.
"""

import argparse
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from doc_scanner import scanner


def draw_corners(image, corners, color=(0, 255, 0), thickness=3):
    """Draw detected corners on the image."""
    if corners is None:
        return image
    
    img_copy = image.copy()
    for corner in corners:
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(img_copy, (x, y), 10, color, thickness)
        cv2.circle(img_copy, (x, y), 15, color, 2)
    
    # Draw lines connecting corners
    if len(corners) == 4:
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(img_copy, [pts], True, color, 2)
    
    return img_copy


def create_demo_image(input_path, output_path, max_width=1200):
    """Create a before/after demo image for the README."""
    # Load original image
    original = cv2.imread(input_path)
    if original is None:
        raise ValueError(f"Could not load image from {input_path}")
    
    # Resize for processing if needed
    height, width = original.shape[:2]
    if height > width:
        resize_ratio = width / 500
    else:
        resize_ratio = height / 500
    
    resized = cv2.resize(
        original,
        (0, 0),
        interpolation=cv2.INTER_AREA,
        fx=1 / resize_ratio,
        fy=1 / resize_ratio,
    )
    
    # Convert to HSV and scan
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    intensity_scanner = scanner(hsv[:, :, 2])
    intensity_scanner.scan()
    
    if intensity_scanner.corners is None:
        saturation_scanner = scanner(hsv[:, :, 1])
        saturation_scanner.scan()
        if saturation_scanner.corners is not None:
            active_scanner = saturation_scanner
        else:
            raise ValueError("Could not detect document corners")
    else:
        active_scanner = intensity_scanner
    
    # Get warped result
    warped = active_scanner.warp(original, scale=resize_ratio)
    
    # Create three versions: original, with corners, and warped
    orig_clean = resized.copy()
    orig_with_corners = resized.copy()
    
    if active_scanner.corners:
        # Get corner coordinates from Frame object
        corners_coords = active_scanner.corners.coordinates()
        corners_resized = [(int(c[0]), int(c[1])) for c in corners_coords]
        orig_with_corners = draw_corners(orig_with_corners, corners_resized)
    
    # Resize images for demo (make them same height)
    target_height = 400
    orig_h, orig_w = orig_clean.shape[:2]
    warp_h, warp_w = warped.shape[:2]
    
    orig_ratio = target_height / orig_h
    warp_ratio = target_height / warp_h
    
    orig_display = cv2.resize(orig_clean, None, fx=orig_ratio, fy=orig_ratio, interpolation=cv2.INTER_AREA)
    corners_display = cv2.resize(orig_with_corners, None, fx=orig_ratio, fy=orig_ratio, interpolation=cv2.INTER_AREA)
    warp_display = cv2.resize(warped, None, fx=warp_ratio, fy=warp_ratio, interpolation=cv2.INTER_AREA)
    
    # Get dimensions for layout
    orig_h, orig_w = orig_display.shape[:2]
    corners_h, corners_w = corners_display.shape[:2]
    warp_h, warp_w = warp_display.shape[:2]
    
    # Add padding and labels
    padding = 20
    label_height = 40
    total_width = orig_w + corners_w + warp_w + padding * 4
    total_height = max(orig_h, corners_h, warp_h) + label_height + padding * 2
    
    # Create canvas
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Place images
    y_offset = label_height + padding
    
    # Original (clean)
    x1 = padding
    canvas[y_offset:y_offset + orig_h, x1:x1 + orig_w] = orig_display
    
    # Original with corners
    x2 = x1 + orig_w + padding
    canvas[y_offset:y_offset + corners_h, x2:x2 + corners_w] = corners_display
    
    # Warped result
    x3 = x2 + corners_w + padding
    canvas[y_offset:y_offset + warp_h, x3:x3 + warp_w] = warp_display
    
    # Convert to PIL for text rendering
    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_canvas)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()
    
    # Add labels
    text_color = (50, 50, 50)
    draw.text((x1, padding), "Original", fill=text_color, font=font)
    draw.text((x2, padding), "Detected Corners", fill=text_color, font=font)
    draw.text((x3, padding), "Cropped", fill=text_color, font=font)
    
    # Convert back to OpenCV format and save
    result = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)
    
    # Resize if too wide
    if total_width > max_width:
        scale = max_width / total_width
        new_width = int(total_width * scale)
        new_height = int(total_height * scale)
        result = cv2.resize(result, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(output_path, result)
    print(f"Demo image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a demo image for README.md")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/images/segment/1526719858.jpg",
        help="Input image path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./demo.png",
        help="Output demo image path",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1200,
        help="Maximum width of output image",
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_demo_image(args.input, args.output, args.max_width)
