"""
ShapeScaler: Map normalised [0,1]² anchor points to real-world lat/lon
coordinates centred on a given start point.

Approach:
- Scale the unit square so the longest axis equals `target_distance_m`.
- Optionally cap the footprint so it fits within `radius_m` of the centre.
- Apply optional bearing rotation.
- Convert metres → lat/lon using a simple local tangent plane approximation
  (accurate enough for distances < ~50 km):
      dlat = dy_m / 111320
      dlon = dx_m / (111320 * cos(lat0_rad))
"""

import math
import numpy as np


def scale_anchors(
    norm_anchors: list[list[float]],
    start_lat: float,
    start_lon: float,
    target_distance_m: float,
    radius_m: float,
    bearing_deg: float = 0.0,
) -> list[list[float]]:
    """
    Return [[lat, lon], ...] for each anchor.

    Parameters
    ----------
    norm_anchors     : [[x, y], ...] in [0, 1]²
    start_lat/lon    : route start / shape centre
    target_distance_m: desired total perimeter / scale length in metres
    radius_m         : maximum allowed radius from centre
    bearing_deg      : clockwise rotation from North in degrees
    """
    pts = np.array(norm_anchors, dtype=float)  # (N, 2)

    # Centre on (0, 0)
    pts -= 0.5

    # Scale so the longest axis span equals target_distance_m
    span = pts.max(axis=0) - pts.min(axis=0)
    max_span = max(span[0], span[1])
    if max_span < 1e-9:
        raise ValueError("Anchors have zero span — degenerate shape")
    scale = target_distance_m / max_span
    pts *= scale  # now in metres, centred at origin

    # Cap to radius
    dists = np.linalg.norm(pts, axis=1)
    max_dist = dists.max()
    if max_dist > radius_m:
        pts *= radius_m / max_dist

    # Apply rotation (bearing is clockwise from North; x=East, y=North)
    # Convert bearing → standard math angle
    angle_rad = math.radians(-bearing_deg)  # negate for CW→CCW
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = (rot @ pts.T).T  # (N, 2): [east_m, north_m]

    # Convert metres → lat/lon using local tangent plane
    lat0_rad = math.radians(start_lat)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(lat0_rad)

    latlon = []
    for east_m, north_m in pts:
        lat = start_lat + north_m / meters_per_deg_lat
        lon = start_lon + east_m / meters_per_deg_lon
        latlon.append([lat, lon])

    return latlon
