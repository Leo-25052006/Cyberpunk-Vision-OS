import asyncio
import json
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

import incidents  # NEW (Track B): incident rules, evidence, SQLite storage, dispatch simulation

BASE_DIR = Path(__file__).resolve().parent          # NEW: paths work no matter where you run python from
VIDEO_PATH = str(BASE_DIR / "campus_video.mp4")  

model = YOLO(str(BASE_DIR / "yolov8l.pt"))

current_headcount = 0
ZONE_CAPACITY_LIMIT = 10
IS_SCANNING = True
SHOW_OVERLAYS = True

current_fps = 0.0
current_inference_ms = 0.0

# NEW: ONE shared vision worker (replaces the per-viewer generate_frames loop)
#  - runs YOLO once per frame no matter how many dashboards are open
#  - keeps a rolling 5-second buffer of frames for incident replay
#  - feeds the headcount into the incident engine
class FrameHub:
    """Holds the latest JPEG so every /video_feed viewer can stream it."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0

    def publish(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.seq += 1
            self.cond.notify_all()

    def wait_next(self, last_seq, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout)
            return self.seq, self.jpeg


hub = FrameHub()
stop_event = threading.Event()

CLIP_FRAMES = 20
CLIP_INTERVAL_S = 0.25


def encode(img, quality=85, width=None):
    if width and img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def vision_loop():
    global current_headcount, current_fps, current_inference_ms
    cap = cv2.VideoCapture(VIDEO_PATH)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frame_interval = 1.0 / src_fps
    clip_buffer = deque(maxlen=CLIP_FRAMES)
    last_sample = 0.0
    last_tick = 0.0
    last_loop = time.time()
    fail_count = 0

    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fail_count += 1
            if fail_count > 3:
                time.sleep(1)
            continue
        fail_count = 0
        now = time.time()

        if IS_SCANNING:
            # YOLO AI
            t0 = time.time()
            results = model(frame, classes=[0], conf=0.20, imgsz=640, verbose=False)
            current_headcount = len(results[0].boxes)
            annotated = results[0].plot()
            current_inference_ms = (time.time() - t0) * 1000
            shown = annotated if SHOW_OVERLAYS else frame

            if now - last_sample >= CLIP_INTERVAL_S:
                clip_buffer.append(encode(annotated, quality=70, width=640))
                last_sample = now

            incidents.engine.update(
                current_headcount, ZONE_CAPACITY_LIMIT, now,
                lambda: (encode(annotated, quality=88), list(clip_buffer)),
            )
        else:
            current_headcount = 0
            shown = frame

        hub.publish(encode(shown))

        if now - last_tick >= 1.0:
            incidents.tick_dispatch(now)
            last_tick = now

        elapsed = time.time() - last_loop
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        loop_time = time.time() - last_loop
        current_fps = 0.9 * current_fps + 0.1 * (1.0 / loop_time) if current_fps else 1.0 / loop_time
        last_loop = time.time()

    cap.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    incidents.init_db()
    stop_event.clear()
    worker = threading.Thread(target=vision_loop, daemon=True, name="vision-worker")
    worker.start()
    yield
    stop_event.set()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

incidents.EVIDENCE_DIR.mkdir(exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(incidents.EVIDENCE_DIR)), name="evidence") 


def mjpeg_stream():
    seq = 0
    while True:
        seq, jpeg = hub.wait_next(seq)
        if jpeg is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


# Control API Endpoints
@app.get("/api/toggle_scan")
def toggle_scan():
    global IS_SCANNING
    IS_SCANNING = not IS_SCANNING
    return {"scanning": IS_SCANNING}


@app.get("/api/toggle_overlay")
def toggle_overlay():
    global SHOW_OVERLAYS
    SHOW_OVERLAYS = not SHOW_OVERLAYS
    return {"overlays": SHOW_OVERLAYS}


@app.get("/api/set_capacity/{value}")
def set_capacity(value: int):
    global ZONE_CAPACITY_LIMIT
    ZONE_CAPACITY_LIMIT = max(1, value)
    return {"capacity": ZONE_CAPACITY_LIMIT}

# incident REST API
class DispatchBody(BaseModel):
    unit_id: Optional[str] = None


@app.get("/api/incidents")
def api_list_incidents(limit: int = 100):
    return {"server_time": time.time(), "incidents": incidents.list_incidents(limit), "stats": incidents.stats()}


@app.get("/api/incidents/{incident_id}")
def api_get_incident(incident_id: int):
    try:
        incident = incidents.get_incident(incident_id)
    except LookupError as e:
        raise HTTPException(404, str(e))
    return {"server_time": time.time(), "incident": incident, "units": incidents.units_status(),
            "incident_pos": incidents.INCIDENT_POS}


@app.post("/api/incidents/{incident_id}/dispatch")
def api_dispatch(incident_id: int, body: Optional[DispatchBody] = None):
    try:
        return {"incident": incidents.dispatch(incident_id, body.unit_id if body else None)}
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/incidents/{incident_id}/resolve")
def api_resolve(incident_id: int):
    try:
        return {"incident": incidents.resolve(incident_id)}
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/incidents/{incident_id}/dismiss")
def api_dismiss(incident_id: int):
    try:
        return {"incident": incidents.dismiss(incident_id)}
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/incidents/reset")
def api_reset():
    incidents.reset_all()
    return {"ok": True}


@app.get("/api/units")
def api_units():
    return {"units": incidents.units_status()}


@app.get("/api/events")
def api_events(limit: int = 200):
    return {"events": incidents.list_events(limit)}


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            alert = "WARNING: Congestion Detected!" if current_headcount >= ZONE_CAPACITY_LIMIT else "Normal"
            if not IS_SCANNING:
                alert = "System Paused"

            s = incidents.stats()
            data = {
                "headcount": current_headcount,
                "status": alert,
                "capacity": ZONE_CAPACITY_LIMIT,
                "scanning": IS_SCANNING,
                # NEW (Track B + performance)
                "incident_version": incidents.version(),
                "open_incidents": s["open"],
                "critical_open": s["critical_open"],
                "fps": round(current_fps, 1) if IS_SCANNING else 0,
                "inference_ms": round(current_inference_ms, 1) if IS_SCANNING else 0,
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1)
    except Exception as e:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
