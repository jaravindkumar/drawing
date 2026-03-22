"""
RouteExporter: Convert a list of [lat, lon] route coordinates to:
  - GPX file (gpxpy)
  - KML file (simplekml, cyan line, width 4)
  - Google Maps URL (max 9 waypoints, evenly decimated)
  - GeoJSON LineString for map display

All binary outputs are returned as base64-encoded strings.
"""

import base64
import math
from datetime import datetime

import gpxpy
import gpxpy.gpx
import numpy as np
import simplekml


def export_all(
    route_coords: list[list[float]],
    mode: str = "run",
) -> dict:
    """
    Parameters
    ----------
    route_coords : [[lat, lon], ...]
    mode         : "run" or "cycle"

    Returns
    -------
    dict with keys: gpx_b64, kml_b64, google_maps_url, geojson
    """
    return {
        "gpx_b64": _export_gpx(route_coords),
        "kml_b64": _export_kml(route_coords),
        "google_maps_url": _export_google_maps_url(route_coords, mode),
        "geojson": _export_geojson(route_coords),
    }


def _export_gpx(route_coords: list[list[float]]) -> str:
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for lat, lon in route_coords:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))

    gpx_str = gpx.to_xml()
    return base64.b64encode(gpx_str.encode("utf-8")).decode("utf-8")


def _export_kml(route_coords: list[list[float]]) -> str:
    kml = simplekml.Kml()
    ls = kml.newlinestring(name="RouteArt")
    # KML coords are (lon, lat[, alt])
    ls.coords = [(lon, lat) for lat, lon in route_coords]
    ls.style.linestyle.color = simplekml.Color.cyan
    ls.style.linestyle.width = 4
    kml_str = kml.kml()
    return base64.b64encode(kml_str.encode("utf-8")).decode("utf-8")


def _export_google_maps_url(
    route_coords: list[list[float]], mode: str
) -> str:
    if not route_coords:
        return ""

    travel_mode = "walking" if mode == "run" else "bicycling"
    origin = f"{route_coords[0][0]},{route_coords[0][1]}"
    destination = f"{route_coords[-1][0]},{route_coords[-1][1]}"

    # Decimate to exactly 9 intermediate waypoints using np.linspace
    if len(route_coords) > 2:
        n_waypoints = min(9, len(route_coords) - 2)
        indices = np.linspace(1, len(route_coords) - 2, n_waypoints, dtype=int)
        waypoints = [f"{route_coords[i][0]},{route_coords[i][1]}" for i in indices]
        wp_str = "|".join(waypoints)
    else:
        wp_str = ""

    url = (
        f"https://www.google.com/maps/dir/?api=1"
        f"&origin={origin}"
        f"&destination={destination}"
    )
    if wp_str:
        url += f"&waypoints={wp_str}"
    url += f"&travelmode={travel_mode}"

    return url


def _export_geojson(route_coords: list[list[float]]) -> dict:
    # GeoJSON uses [lon, lat] order
    coordinates = [[lon, lat] for lat, lon in route_coords]
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
        "properties": {},
    }
