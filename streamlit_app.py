"""
RouteArt — Streamlit UI

Calls backend pipeline functions directly (no HTTP server required).
Map: streamlit-folium with CartoDB dark tiles.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import base64
import io
import math

import folium
import streamlit as st
from streamlit_folium import st_folium

from pipeline.image_processor import process_image, _get_rembg_session
from pipeline.vector_extractor import extract_anchors
from pipeline.shape_scaler import scale_anchors
from pipeline.graph_router import build_route, compute_actual_distance
from pipeline.route_exporter import export_all

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RouteArt",
    page_icon="🗺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Dark accent overrides */
  [data-testid="stSidebar"] { background-color: #1a1a1a; }
  h1 { color: #00e5ff !important; }
  .metric-label { color: #888 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading background removal model…")
def _cached_rembg_session():
    return _get_rembg_session()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in [
    ("start_lat", None),
    ("start_lon", None),
    ("anchors_latlon", None),
    ("route_coords", None),
    ("fidelity_score", None),
    ("actual_distance", None),
    ("exports", None),
    ("warning", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("RouteArt")
    st.caption("Turn any image into a real-world GPS route")
    st.divider()

    # Step 1: pin status
    st.markdown("**① Start Point**")
    if st.session_state.start_lat is not None:
        st.success(
            f"📍 {st.session_state.start_lat:.5f}, {st.session_state.start_lon:.5f}"
        )
    else:
        st.info("Click the map to set a start point")

    st.divider()

    # Step 2: mode
    st.markdown("**② Activity Mode**")
    mode = st.radio("Mode", ["Run 🏃", "Cycle 🚴"], horizontal=True, label_visibility="collapsed")
    mode_key = "run" if "Run" in mode else "cycle"

    st.divider()

    # Step 3: sliders
    st.markdown("**③ Shape Parameters**")
    distance_km = st.slider("Distance (km)", 1.0, 30.0, 5.0, 0.5)
    radius_km = st.slider("Search Radius (km)", 0.5, 10.0, 2.0, 0.5)
    bearing_deg = st.slider("Rotation (°)", 0, 360, 0, 5)

    st.divider()

    # Step 4: image upload
    st.markdown("**④ Shape Image**")
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, use_container_width=True, caption="Shape preview")
    else:
        image_bytes = None

    st.divider()

    # Action buttons
    can_act = st.session_state.start_lat is not None and image_bytes is not None

    col_prev, col_gen = st.columns(2)
    with col_prev:
        preview_clicked = st.button(
            "Preview", disabled=not can_act, use_container_width=True
        )
    with col_gen:
        generate_clicked = st.button(
            "Generate", disabled=not can_act, use_container_width=True, type="primary"
        )

    # Warning
    if st.session_state.warning:
        st.warning(st.session_state.warning)

    # Results
    if st.session_state.fidelity_score is not None:
        st.divider()
        st.markdown("**Results**")
        c1, c2 = st.columns(2)
        c1.metric("Fidelity", f"{st.session_state.fidelity_score:.1f}%")
        dist_km = st.session_state.actual_distance / 1000
        c2.metric("Distance", f"{dist_km:.2f} km")

    # Exports
    if st.session_state.exports:
        st.divider()
        st.markdown("**Export**")
        gmaps_url = st.session_state.exports.get("google_maps_url", "")
        if gmaps_url:
            st.link_button("🗺 Open in Google Maps", gmaps_url, use_container_width=True)

        kml_b64 = st.session_state.exports.get("kml_b64", "")
        if kml_b64:
            kml_bytes = base64.b64decode(kml_b64)
            st.download_button(
                "📥 Download KML",
                data=kml_bytes,
                file_name="routeart.kml",
                mime="application/vnd.google-earth.kml+xml",
                use_container_width=True,
            )

        gpx_b64 = st.session_state.exports.get("gpx_b64", "")
        if gpx_b64:
            gpx_bytes = base64.b64decode(gpx_b64)
            st.download_button(
                "📥 Download GPX",
                data=gpx_bytes,
                file_name="routeart.gpx",
                mime="application/gpx+xml",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def run_image_pipeline(image_bytes, start_lat, start_lon, distance_m, radius_m, bearing):
    binary_mask = process_image(image_bytes, session=_cached_rembg_session())
    norm_anchors = extract_anchors(binary_mask)
    warning = None
    if len(norm_anchors) < 10:
        warning = f"Shape is very simple ({len(norm_anchors)} anchors); route may not closely match."
    anchors_latlon = scale_anchors(
        norm_anchors, start_lat, start_lon, distance_m, radius_m, bearing
    )
    return anchors_latlon, warning


# ---------------------------------------------------------------------------
# Handle Preview
# ---------------------------------------------------------------------------
if preview_clicked and image_bytes and st.session_state.start_lat is not None:
    with st.spinner("Extracting shape contour…"):
        try:
            anchors, warning = run_image_pipeline(
                image_bytes,
                st.session_state.start_lat,
                st.session_state.start_lon,
                distance_km * 1000,
                radius_km * 1000,
                bearing_deg,
            )
            st.session_state.anchors_latlon = anchors
            st.session_state.warning = warning
            st.session_state.route_coords = None  # clear old route
            st.session_state.exports = None
            st.session_state.fidelity_score = None
            st.session_state.actual_distance = None
        except Exception as e:
            st.error(f"Preview failed: {e}")

# ---------------------------------------------------------------------------
# Handle Generate
# ---------------------------------------------------------------------------
if generate_clicked and image_bytes and st.session_state.start_lat is not None:
    with st.spinner("Routing on street network… this may take 30-60 seconds on first run."):
        try:
            anchors, warning = run_image_pipeline(
                image_bytes,
                st.session_state.start_lat,
                st.session_state.start_lon,
                distance_km * 1000,
                radius_km * 1000,
                bearing_deg,
            )
            st.session_state.anchors_latlon = anchors

            route_coords, fidelity = build_route(
                anchors,
                center_lat=st.session_state.start_lat,
                center_lon=st.session_state.start_lon,
                radius_m=radius_km * 1000,
                mode=mode_key,
            )
            actual_distance = compute_actual_distance(route_coords)
            exports = export_all(route_coords, mode_key)
            # remove geojson from exports dict (used internally)
            exports.pop("geojson", None)

            st.session_state.route_coords = route_coords
            st.session_state.fidelity_score = fidelity
            st.session_state.actual_distance = actual_distance
            st.session_state.exports = exports
            st.session_state.warning = warning
            st.session_state.anchors_latlon = None  # hide preview once route is shown
        except Exception as e:
            st.error(f"Generate failed: {e}")

# ---------------------------------------------------------------------------
# Build Folium map
# ---------------------------------------------------------------------------
# Determine map centre and zoom
if st.session_state.route_coords:
    lats = [c[0] for c in st.session_state.route_coords]
    lons = [c[1] for c in st.session_state.route_coords]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    zoom = 13
elif st.session_state.anchors_latlon:
    lats = [c[0] for c in st.session_state.anchors_latlon]
    lons = [c[1] for c in st.session_state.anchors_latlon]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]
    zoom = 13
elif st.session_state.start_lat is not None:
    center = [st.session_state.start_lat, st.session_state.start_lon]
    zoom = 14
else:
    center = [20, 0]
    zoom = 2

m = folium.Map(
    location=center,
    zoom_start=zoom,
    tiles="CartoDB dark_matter",
    width="100%",
)

# Start pin
if st.session_state.start_lat is not None:
    folium.CircleMarker(
        location=[st.session_state.start_lat, st.session_state.start_lon],
        radius=7,
        color="#00e5ff",
        fill=True,
        fill_color="#00e5ff",
        fill_opacity=0.9,
        tooltip="Start",
    ).add_to(m)

# Preview shape overlay (dashed)
if st.session_state.anchors_latlon:
    pts = st.session_state.anchors_latlon + [st.session_state.anchors_latlon[0]]
    folium.PolyLine(
        locations=pts,
        color="#888888",
        weight=2,
        dash_array="8 5",
        tooltip="Shape preview",
    ).add_to(m)

# Route line
if st.session_state.route_coords:
    folium.PolyLine(
        locations=st.session_state.route_coords,
        color="#00e5ff",
        weight=4,
        opacity=0.9,
        tooltip="Route",
    ).add_to(m)

    # Start / end markers
    folium.CircleMarker(
        location=st.session_state.route_coords[0],
        radius=6,
        color="#00ff88",
        fill=True,
        fill_color="#00ff88",
        tooltip="Route start",
    ).add_to(m)
    folium.CircleMarker(
        location=st.session_state.route_coords[-1],
        radius=6,
        color="#ff4444",
        fill=True,
        fill_color="#ff4444",
        tooltip="Route end",
    ).add_to(m)

# ---------------------------------------------------------------------------
# Render map & capture clicks
# ---------------------------------------------------------------------------
st.markdown("### 🗺 Map  —  click to set start point")
map_output = st_folium(m, use_container_width=True, height=620, returned_objects=["last_clicked"])

# Update start point from click
if map_output and map_output.get("last_clicked"):
    clicked = map_output["last_clicked"]
    new_lat = clicked["lat"]
    new_lon = clicked["lng"]
    if new_lat != st.session_state.start_lat or new_lon != st.session_state.start_lon:
        st.session_state.start_lat = new_lat
        st.session_state.start_lon = new_lon
        # Reset downstream state when pin moves
        st.session_state.anchors_latlon = None
        st.session_state.route_coords = None
        st.session_state.exports = None
        st.session_state.fidelity_score = None
        st.session_state.actual_distance = None
        st.session_state.warning = None
        st.rerun()
