from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
from typing import Optional
from .inference import run_lane_inference
from .camera import CameraCapture

app = FastAPI()

#CORS for phone app
#allows any frontend(Android, iOS, Web) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#single shared camera instance
camera = CameraCapture()
#tracks connected WebSocket clients
active_connections = set()

#continuously: 
#captures a camera frame
#runs lane detection
#sends JSON metadata to the phone
#target: 5-10 Hz (5-10 updates per second)
@app.websocket("/ws/lane-metadata")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)

    try:
        while True:
            #Capture frame
            frame = camera.capture_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            #Run inference: send image frame to AI model
            result = run_lane_inference(frame)

            #Format metadata: expected output, converts raw inference output into clean, structured metadata
            metadata = {
                "lane_count": result.get("lane_count", 0),
                "recommended_lanes": result.get("recommended_lanes", []),
                "confidence": result.get("confidence", 0.0),
                "timestamp": int(datetime.now().timestamp()),
                "model_version": "lane_v0.1"
            }
            
            #Broadcast to all connected clients
            await websocket.send_json(metadata)

            #Target 5-10 Hz (100-200ms per frame): prevents overheating / overload
            await asyncio.sleep(0.1)

    #catches disconnects or runtime errors
    except Exception as e:
        print(f"WebSocket error: {e}")
    #remove client from active list when disconnected
    finally:
        active_connections.discard(websocket)

@app.get("/health")
async def health():
    return {"status": "ok", "fps": camera.get_fps()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)