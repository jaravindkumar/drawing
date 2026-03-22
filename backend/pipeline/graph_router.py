"""
GraphRouter: Load an OSMnx street graph, snap anchor lat/lons to nearest
nodes, build a sequential shortest-path route, and compute a fidelity score.

Caching strategy:
- Try Redis (localhost:6379) first, storing the serialised graph as pickle bytes.
- Fall back to an in-memory dict if Redis is unavailable.
- Cache key: MD5 of "lat4dp_lon4dp_radius_mode"
- TTL: 24 hours (Redis) / unlimited (in-memory, lives for process lifetime)

Fidelity score (0-100):
  For each target anchor, find the min distance to any route node.
  Average those distances, normalise by the bbox diagonal of the anchors,
  return 100 * (1 - normalised_mean_deviation).
"""

import hashlib
import math
import pickle
import time
from typing import Optional

import networkx as nx
import numpy as np
import osmnx as osmnx_lib

try:
    import redis as redis_lib

    _redis_client: Optional[redis_lib.Redis] = redis_lib.Redis(
        host="localhost", port=6379, db=0, socket_connect_timeout=1
    )
    # Test connection
    _redis_client.ping()
    _redis_available = True
except Exception:
    _redis_client = None
    _redis_available = False

_memory_cache: dict = {}
_CACHE_TTL = 86400  # 24 hours in seconds


def _cache_key(lat: float, lon: float, radius_m: float, mode: str) -> str:
    raw = f"{lat:.4f}_{lon:.4f}_{radius_m:.0f}_{mode}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[object]:
    if _redis_available and _redis_client is not None:
        try:
            data = _redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception:
            pass
    entry = _memory_cache.get(key)
    if entry:
        obj, ts = entry
        if time.time() - ts < _CACHE_TTL:
            return obj
        del _memory_cache[key]
    return None


def _cache_set(key: str, obj: object) -> None:
    if _redis_available and _redis_client is not None:
        try:
            _redis_client.setex(key, _CACHE_TTL, pickle.dumps(obj))
            return
        except Exception:
            pass
    _memory_cache[key] = (obj, time.time())


def _load_graph(
    lat: float, lon: float, radius_m: float, mode: str
) -> nx.MultiDiGraph:
    key = _cache_key(lat, lon, radius_m, mode)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    network_type = "walk" if mode == "run" else "bike"
    G = osmnx_lib.graph_from_point(
        (lat, lon),
        dist=radius_m,
        network_type=network_type,
        simplify=True,
    )
    _cache_set(key, G)
    return G


def build_route(
    anchors_latlon: list[list[float]],
    center_lat: float,
    center_lon: float,
    radius_m: float,
    mode: str = "run",
) -> tuple[list[list[float]], float]:
    """
    Build a sequential route through anchor points on the OSM graph.

    Returns
    -------
    route_coords : [[lat, lon], ...]
    fidelity_score : float 0-100
    """
    G = _load_graph(center_lat, center_lon, radius_m, mode)

    # Snap each anchor to its nearest graph node
    nearest_nodes = []
    for lat, lon in anchors_latlon:
        node = osmnx_lib.nearest_nodes(G, lon, lat)
        nearest_nodes.append(node)

    # Build route by sequential shortest paths between consecutive nodes
    route_nodes: list[int] = []
    for i in range(len(nearest_nodes) - 1):
        src = nearest_nodes[i]
        dst = nearest_nodes[i + 1]
        try:
            segment = nx.shortest_path(G, src, dst, weight="length")
        except nx.NetworkXNoPath:
            # If no path, just append endpoints
            segment = [src, dst]
        if route_nodes and segment[0] == route_nodes[-1]:
            segment = segment[1:]
        route_nodes.extend(segment)

    # Optionally close the loop back to start
    if len(nearest_nodes) >= 2 and nearest_nodes[0] != nearest_nodes[-1]:
        try:
            closing = nx.shortest_path(
                G, nearest_nodes[-1], nearest_nodes[0], weight="length"
            )
            if closing and closing[0] == route_nodes[-1]:
                closing = closing[1:]
            route_nodes.extend(closing)
        except nx.NetworkXNoPath:
            pass

    # Extract lat/lon for each node
    route_coords = [
        [G.nodes[n]["y"], G.nodes[n]["x"]] for n in route_nodes
    ]

    fidelity = _compute_fidelity(anchors_latlon, route_coords)

    return route_coords, fidelity


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _compute_fidelity(
    target_anchors: list[list[float]],
    route_coords: list[list[float]],
) -> float:
    if not route_coords or not target_anchors:
        return 0.0

    route_arr = np.array(route_coords)  # (M, 2) lat/lon
    target_arr = np.array(target_anchors)  # (N, 2) lat/lon

    # Bounding box diagonal of target anchors (metres)
    lat_range = target_arr[:, 0].max() - target_arr[:, 0].min()
    lon_range = target_arr[:, 1].max() - target_arr[:, 1].min()
    mid_lat = target_arr[:, 0].mean()
    diag_m = math.sqrt(
        (lat_range * 111320) ** 2
        + (lon_range * 111320 * math.cos(math.radians(mid_lat))) ** 2
    )
    if diag_m < 1.0:
        return 100.0

    # For each target anchor, find min distance to any route node
    deviations = []
    for tlat, tlon in target_arr:
        dists = [
            _haversine_m(tlat, tlon, rlat, rlon)
            for rlat, rlon in route_coords
        ]
        deviations.append(min(dists))

    mean_dev = np.mean(deviations)
    normalised = mean_dev / diag_m
    score = max(0.0, 100.0 * (1.0 - normalised))
    return round(score, 1)


def compute_actual_distance(route_coords: list[list[float]]) -> float:
    """Return total route distance in metres."""
    total = 0.0
    for i in range(len(route_coords) - 1):
        lat1, lon1 = route_coords[i]
        lat2, lon2 = route_coords[i + 1]
        total += _haversine_m(lat1, lon1, lat2, lon2)
    return round(total, 1)
