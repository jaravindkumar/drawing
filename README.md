# RouteArt

Turn any image into a real-world running or cycling GPS route that follows roads.

## How It Works

1. Upload an image of a shape (logo, letter, animal, etc.)
2. Set a start point on the map
3. Choose Run or Cycle mode
4. Set distance, radius, and rotation
5. Preview the shape overlay, then generate the route
6. Export to Google Maps, GPX, or KML

## Stack

- **Backend**: Python FastAPI
- **Frontend**: Vanilla JS + MapLibre GL JS
- **Image processing**: OpenCV + rembg
- **Routing**: OSMnx + NetworkX
- **Export**: gpxpy + simplekml

## Run Instructions

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in your browser (or serve it via any static file server).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/preview` | Returns shape anchor points scaled to real-world coordinates |
| POST | `/generate` | Returns full route with GPX/KML/Google Maps exports |
| GET | `/health` | Health check |

## Optional: Redis Cache

Install and run Redis locally for faster repeated OSM graph lookups:

```bash
redis-server
```

Falls back to in-memory cache automatically if Redis is unavailable.
