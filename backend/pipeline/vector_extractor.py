"""
VectorExtractor: Find the outer contour of a binary mask and simplify it
to 20-40 anchor points normalised to [0, 1]².

Steps:
1. cv2.findContours – pick the largest contour.
2. Adaptive Ramer-Douglas-Peucker via binary search so the resulting
   point count lands in [target_min, target_max].
3. Normalise to [0, 1]² with origin at top-left.
"""

import numpy as np
import cv2
from rdp import rdp

TARGET_MIN = 20
TARGET_MAX = 40


def extract_anchors(binary_mask: np.ndarray) -> list[list[float]]:
    """
    Return normalised anchor points as [[x, y], ...] in [0,1]².
    Raises ValueError if no contour is found.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("No contour found in the processed image")

    # Largest contour by area
    largest = max(contours, key=cv2.contourArea)
    pts = largest[:, 0, :].astype(float)  # shape (N, 2)

    # Adaptive RDP: binary search on epsilon
    simplified = _adaptive_rdp(pts, TARGET_MIN, TARGET_MAX)

    # Normalise
    h, w = binary_mask.shape[:2]
    norm = simplified.copy()
    norm[:, 0] /= max(w - 1, 1)
    norm[:, 1] /= max(h - 1, 1)
    # Clamp to [0, 1]
    norm = np.clip(norm, 0.0, 1.0)

    return norm.tolist()


def _adaptive_rdp(
    pts: np.ndarray,
    target_min: int,
    target_max: int,
    max_iters: int = 50,
) -> np.ndarray:
    """
    Binary search over RDP epsilon to get point count in [target_min, target_max].
    """
    eps_lo, eps_hi = 0.0, float(max(pts[:, 0].max(), pts[:, 1].max()))

    # Quick bounds check: even eps=0 produces too few points
    raw_count = len(pts)
    if raw_count <= target_max:
        return pts

    best = pts
    for _ in range(max_iters):
        eps_mid = (eps_lo + eps_hi) / 2.0
        simplified = rdp(pts, epsilon=eps_mid)
        count = len(simplified)
        if target_min <= count <= target_max:
            return np.array(simplified)
        if count > target_max:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid
            best = np.array(simplified)
        if eps_hi - eps_lo < 1e-6:
            break

    # If best is still empty or too small, fall back to evenly sampled points
    if len(best) < target_min:
        indices = np.linspace(0, len(pts) - 1, target_min, dtype=int)
        best = pts[indices]

    return best
