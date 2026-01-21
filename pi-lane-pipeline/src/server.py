try:
    from . import imp_compat  # Compatibility shim for Python 3.12+
except (ImportError, ModuleNotFoundError):
    pass 

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
from typing import Optional
import time
import traceback
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
    print(f"✅ WebSocket client connected. Total: {len(active_connections)}")

    try:
        while True:
            try:
                # Capture frame
                frame = camera.capture_frame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Run inference with error handling
                try:
                    result = run_lane_inference(frame)
                except Exception as inf_error:
                    print(f"❌ Inference error: {inf_error}")
                    print(traceback.format_exc())
                    # Return safe defaults
                    result = {
                        "lane_count": 1,
                        "current_lane_index": 0,
                        "lane_centers": [],
                        "confidence": 0.0,
                        "inference_ms": 0.0
                    }
                
                # Format metadata
                try:
                    metadata = {
                        "lane_count": result.get("lane_count", 1),
                        "current_lane_index": result.get("current_lane_index", 0),
                        "lane_centers": result.get("lane_centers", []),
                        "confidence": result.get("confidence", 0.0),
                        "fps_camera": camera.get_fps() if camera else 0.0,
                        "inference_ms": result.get("inference_ms", 0.0),
                        "timestamp": int(datetime.now().timestamp()),
                        "model_version": "lane_v0.1"
                    }
                except Exception as meta_error:
                    print(f"❌ Metadata formatting error: {meta_error}")
                    continue  # Skip this frame
                
                # Send to client
                try:
                    await websocket.send_json(metadata)
                except Exception as send_error:
                    print(f"❌ WebSocket send error: {send_error}")
                    break  # Connection closed, exit loop
                
                # Target 5-10 Hz
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                print("⚠️  WebSocket cancelled")
                break
            except Exception as loop_error:
                print(f"❌ Error in WebSocket loop: {loop_error}")
                print(traceback.format_exc())
                await asyncio.sleep(0.1)  # Small delay before retry
                continue  # Continue loop, don't crash

    except Exception as e:
        print(f"❌ WebSocket endpoint error: {e}")
        print(traceback.format_exc())
    finally:
        active_connections.discard(websocket)
        print(f"⚠️  WebSocket disconnected. Remaining: {len(active_connections)}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "fps": camera.get_fps() if camera else 0.0,
        "camera_available": camera.camera is not None if camera else False,
        "active_connections": len(active_connections)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)