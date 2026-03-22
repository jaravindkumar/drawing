"""
ImageProcessor: Remove background and produce a binary silhouette mask.

Steps:
1. rembg.remove() strips the background, leaving an RGBA image.
2. Extract the alpha channel as the foreground mask.
3. Otsu threshold on grayscale as fallback / refinement.
4. Return a binary (0/255) numpy array.
"""

import numpy as np
import cv2
from rembg import remove


def process_image(image_bytes: bytes) -> np.ndarray:
    """
    Accept raw image bytes, return a binary mask (uint8, values 0 or 255).
    """
    # Remove background via rembg
    output_bytes = remove(image_bytes)

    # Decode the RGBA PNG that rembg returns
    nparr = np.frombuffer(output_bytes, np.uint8)
    img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img_rgba is None:
        raise ValueError("Failed to decode rembg output image")

    # Use alpha channel as the primary mask if RGBA
    if img_rgba.shape[2] == 4:
        alpha = img_rgba[:, :, 3]
        # Threshold: pixels with alpha > 10 are foreground
        _, binary_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    else:
        # Fallback: grayscale + Otsu
        gray = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Morphological closing to fill small holes inside the shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return binary_mask
