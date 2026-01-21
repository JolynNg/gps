Code Explanation

1. mask_processor.py — Lane mask to metrics converter
    Purpose: Converts a binary lane mask (from AI model) into lane metrics.
    How it works:
    Input: Binary mask (white pixels = lanes, black = background)
    Process:
    Normalizes mask to 0–255
    Focuses on bottom half (where lanes are most visible)
    Creates histogram: counts lane pixels per column
    Finds peaks: lane boundaries (vertical lines)
    Groups nearby peaks to avoid duplicates
    Calculates lane_count: gaps between boundaries
    Finds lane_centers: midpoint between boundaries
    Determines current_lane_index: lane closest to image center
    Calculates confidence: mask quality + peak strength
    Output: {lane_count, current_lane_index, lane_centers, confidence}
    Example: If mask has 3 vertical lines → 2 lanes detected → lane_centers = [midpoint1, midpoint2]


2. inference.py — AI model inference runner
    Purpose: Runs the TFLite lane detection model on camera frames.
    How it works:
    Model loading:
    Tries tflite_runtime (Pi-optimized), falls back to tensorflow
    Loads model from models/lane_detection.tflite
    If no model found → generates dummy mask for testing
    Inference pipeline:
    Preprocess: Resize frame to model input size (640x480), normalize
    Run model: Feed frame to TFLite interpreter
    Get mask: Extract segmentation mask from model output
    Postprocess: Handle batch/channel dimensions, resize to original size
    Process mask: Calls mask_processor.py to get lane metrics
    Measure time: Records inference_ms for performance monitoring
    Fallback: If model fails or doesn't exist, returns dummy mask with 3 fake lane lines.

3. server.py — WebSocket server for real-time streaming
    Purpose: Captures frames, runs inference, streams results to phone app via WebSocket.
    How it works:
    WebSocket endpoint (/ws/lane-metadata):
    Accepts connection from phone
    Continuous loop (5–10 Hz):
    Captures frame from camera
    Runs inference
    Formats metadata JSON
    Sends to phone
    Sleeps 0.1s (10 Hz max)
    Metadata structure (Phase 0 + Phase 1):
        { "lane_count": 3,    
          "current_lane_index": 1,   
          "lane_centers": [160, 320, 480],   
          "confidence": 0.85,   
          "fps_camera": 8.5,   
          "inference_ms": 45.2,    
          "timestamp": 1234567890,    
          "model_version": "lane_v0.1" 
         }
    Health endpoint: Returns camera FPS and model status

4. lane_recommender.py — Navigation-based lane recommendation (Phase 5)
    Purpose: Converts lane metrics + navigation instructions into recommended lanes.
    How it works:
    Input: lane_count, current_lane_index, navigation_maneuver, distance_to_turn
    Rule engine:
    "exit_right" → recommend rightmost lanes (especially if <200m away)
    "left" → recommend left lanes
    "straight" → recommend middle lanes (if 3+ lanes)
    Default → stay in current lane
    Output: List of recommended lane numbers (1-indexed for UI)
    Note: Not used yet in Phase 0; will be integrated in Phase 5.


Pi Camera → capture_frame()
    ↓
inference.py → run_lane_inference()
    ↓
mask_processor.py → process_lane_mask()
    ↓
server.py → WebSocket JSON
    ↓
piClient.ts → parse & callback
    ↓
App.tsx → update store + send telemetry
    ↓
LaneBar.tsx → display lanes