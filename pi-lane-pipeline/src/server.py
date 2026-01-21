from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
from typing import Optional
import time
from .inference import run_lane_inference
from .camera import CameraCapture

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = CameraCapture()
active_connections = set()

@app.websocket("/ws/lane-metadata")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)

    try:
        while True:
            frame_start = time.time()
            
            # Capture frame
            frame = camera.capture_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            
            camera_ms = (time.time() - frame_start) * 1000
            
            # Run inference
            result = run_lane_inference(frame)
            
            # Format metadata with Phase 0 structure
            metadata = {
                # Phase 0 MVP outputs
                "lane_count": result.get("lane_count", 1),
                "current_lane_index": result.get("current_lane_index", 0),
                "lane_centers": result.get("lane_centers", []),
                "confidence": result.get("confidence", 0.0),
                
                # Telemetry metrics (Phase 1 requirement)
                "fps_camera": camera.get_fps(),
                "inference_ms": result.get("inference_ms", 0.0),
                
                # Metadata
                "timestamp": int(datetime.now().timestamp()),
                "model_version": "lane_v0.1"
            }
            
            # Broadcast to all connected clients
            await websocket.send_json(metadata)
            
            # Target 5-10 Hz
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "fps": camera.get_fps(),
        "model_loaded": interpreter is not None if 'interpreter' in globals() else False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)