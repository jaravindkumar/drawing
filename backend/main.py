"""
RouteArt FastAPI backend.

Endpoints
---------
GET  /health   — liveness check
POST /preview  — image → scaled anchor lat/lons (no OSM)
POST /generate — image → full routed GPX/KML/GeoJSON + fidelity score
"""

import base64
import math
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.image_processor import process_image
from pipeline.vector_extractor import extract_anchors
from pipeline.shape_scaler import scale_anchors
from pipeline.graph_router import build_route, compute_actual_distance
from pipeline.route_exporter import export_all

app = FastAPI(title="RouteArt", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PreviewRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes")
    start_lat: float
    start_lon: float
    distance_m: float = Field(5000.0, gt=0)
    radius_m: float = Field(2000.0, gt=0)
    bearing_deg: float = Field(0.0, ge=0.0, le=360.0)


class PreviewResponse(BaseModel):
    anchors_latlon: list[list[float]]
    warning: Optional[str] = None


class GenerateRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes")
    start_lat: float
    start_lon: float
    distance_m: float = Field(5000.0, gt=0)
    radius_m: float = Field(2000.0, gt=0)
    mode: str = Field("run", pattern="^(run|cycle)$")
    bearing_deg: float = Field(0.0, ge=0.0, le=360.0)
    fidelity_w: float = Field(1.0, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    route_geojson: dict
    fidelity_score: float
    actual_distance: float
    exports: dict
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_image(image_b64: str) -> bytes:
    try:
        return base64.b64decode(image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")


def _run_image_pipeline(
    image_bytes: bytes,
    start_lat: float,
    start_lon: float,
    distance_m: float,
    radius_m: float,
    bearing_deg: float,
) -> tuple[list[list[float]], Optional[str]]:
    """Run image → anchors pipeline, return (anchors_latlon, warning)."""
    binary_mask = process_image(image_bytes)
    norm_anchors = extract_anchors(binary_mask)

    warning = None
    anchor_count = len(norm_anchors)
    if anchor_count < 10:
        warning = f"Shape is very simple ({anchor_count} anchors); route may not resemble the image."

    anchors_latlon = scale_anchors(
        norm_anchors, start_lat, start_lon, distance_m, radius_m, bearing_deg
    )
    return anchors_latlon, warning


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/preview", response_model=PreviewResponse)
def preview(req: PreviewRequest):
    image_bytes = _decode_image(req.image_b64)
    try:
        anchors_latlon, warning = _run_image_pipeline(
            image_bytes,
            req.start_lat,
            req.start_lon,
            req.distance_m,
            req.radius_m,
            req.bearing_deg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PreviewResponse(anchors_latlon=anchors_latlon, warning=warning)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    image_bytes = _decode_image(req.image_b64)
    try:
        anchors_latlon, warning = _run_image_pipeline(
            image_bytes,
            req.start_lat,
            req.start_lon,
            req.distance_m,
            req.radius_m,
            req.bearing_deg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        route_coords, fidelity_score = build_route(
            anchors_latlon,
            center_lat=req.start_lat,
            center_lon=req.start_lon,
            radius_m=req.radius_m,
            mode=req.mode,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Routing failed: {exc}"
        )

    actual_distance = compute_actual_distance(route_coords)
    exports = export_all(route_coords, req.mode)
    route_geojson = exports.pop("geojson")

    return GenerateResponse(
        route_geojson=route_geojson,
        fidelity_score=fidelity_score,
        actual_distance=actual_distance,
        exports=exports,
        warning=warning,
    )
