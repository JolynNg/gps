
try:
    from . import imp_compat
except (ImportError, ModuleNotFoundError):
    pass

import numpy as np
from typing import Dict, Any, Optional
import time
import os
import traceback
from .mask_processor import process_lane_mask

# Lazy import TensorFlow
tf = None
interpreter = None
input_details = None
output_details = None

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lane_detection.tflite")
INPUT_SIZE = (800, 288)  
NORMALIZE = True
INPUT_MEAN = 127.5
INPUT_STD = 127.5

def load_model(model_path: Optional[str] = None):
    """Load TFLite model."""
    global interpreter, tf, input_details, output_details
    
    if interpreter is not None:
        return
    
    if tf is None:
        try:
            import tflite_runtime.interpreter as tflite
            tf = tflite
            use_tflite_runtime = True
        except ImportError:
            import tensorflow as tf
            use_tflite_runtime = False
    
    model_path = model_path or MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}. Using dummy mask.")
        return
    
    try:
        # Handle both tflite_runtime (Pi) and tensorflow (Mac)
        if use_tflite_runtime:
            interpreter = tf.Interpreter(model_path=model_path)
        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ Model loaded: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        interpreter = None

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess frame for model input."""
    import cv2
    resized = cv2.resize(frame, INPUT_SIZE)
    processed = resized.astype(np.float32)
    
    if NORMALIZE:
        processed = (processed - INPUT_MEAN) / INPUT_STD
    else:
        processed = processed / 255.0
    
    processed = np.expand_dims(processed, axis=0)
    return processed

# def keypoints_to_mask(keypoints_output, frame_shape):
#     """
#     Convert UFLD keypoints to binary mask.
#     UFLD outputs shape: (batch, num_rows, num_cols, num_lanes)
#     Shape (1, 101, 56, 4) means: 101 rows, 56 grid points, 4 lanes
#     """
#     import cv2
#     h, w = frame_shape
#     mask = np.zeros((h, w), dtype=np.uint8)
    
#     # Remove batch dimension: (101, 56, 4)
#     if len(keypoints_output.shape) == 4:
#         keypoints_output = keypoints_output[0]
    
#     # Shape should now be (num_rows, num_cols, num_lanes)
#     num_rows, num_cols, num_lanes = keypoints_output.shape
    
#     # Generate row indices
#     row_indices = np.linspace(h//2, h-1, num_rows, dtype=np.int32)
    
#     # Process each lane
#     for lane_idx in range(num_lanes):
#         lane_data = keypoints_output[:, :, lane_idx]  # Shape: (101, 56)
        
#         # Get x-coordinates: take argmax along the column dimension
#         x_grid_positions = np.argmax(lane_data, axis=1)  # Shape: (101,)
        
#         # Get max values to determine validity
#         max_values = np.max(lane_data, axis=1)
        
#         # Use adaptive threshold: lower threshold to catch more lanes
#         # Also check if the max value is significantly above the mean
#         mean_max = np.mean(max_values)
#         threshold = max(0.01, mean_max * 0.3)  # Adaptive threshold
        
#         # Filter valid points
#         valid_mask = max_values > threshold
        
#         if np.any(valid_mask):
#             # Scale from grid (0-55) to pixel coordinates
#             x_coords = (x_grid_positions[valid_mask] * w / num_cols).astype(np.int32)
#             y_coords = row_indices[valid_mask]
            
#             points = np.column_stack((x_coords, y_coords))
            
#             # Draw thicker lines for better mask coverage
#             for i in range(len(points) - 1):
#                 cv2.line(mask, tuple(points[i]), tuple(points[i+1]), 255, 5)  # Thicker: 5 instead of 3
            
#             # Also draw circles at key points for better coverage
#             for point in points:
#                 cv2.circle(mask, tuple(point), 3, 255, -1)
    
#     return mask

def keypoints_to_lane_metrics(keypoints_output, frame_shape):
    """
    Directly extract lane metrics from keypoints without converting to mask.
    More reliable for lane counting.
    """
    h, w = frame_shape
    
    # Remove batch dimension
    if len(keypoints_output.shape) == 4:
        keypoints_output = keypoints_output[0]
    
    num_rows, num_cols, num_lanes = keypoints_output.shape
    
    # Count valid lanes and get their x-coordinates at bottom of image
    valid_lanes = []
    lane_x_positions = []
    
    for lane_idx in range(num_lanes):
        lane_data = keypoints_output[:, :, lane_idx]  # Shape: (101, 56)
        max_values = np.max(lane_data, axis=1)
        mean_confidence = np.mean(max_values)
        
        # Lane is valid if mean confidence is above threshold
        if mean_confidence > 0.05:  # Adjust threshold as needed
            x_grid = np.argmax(lane_data, axis=1)
            # Get x-coordinate at bottom of image (last few rows, average for stability)
            bottom_rows = x_grid[-10:] if len(x_grid) >= 10 else x_grid
            # Filter out invalid grid positions (0 or very low)
            valid_bottom = bottom_rows[bottom_rows > 0]
            if len(valid_bottom) > 0:
                # Average x position at bottom
                avg_x_grid = np.mean(valid_bottom)
                x_pixel = int(avg_x_grid * w / num_cols)
                valid_lanes.append(lane_idx)
                lane_x_positions.append(x_pixel)
    
        # Sort lane positions by x-coordinate (left to right)
    if len(lane_x_positions) > 0:
        sorted_indices = np.argsort(lane_x_positions)
        lane_x_positions = [lane_x_positions[i] for i in sorted_indices]
    
    # Calculate lane centers (midpoints between adjacent lane lines)
    # These represent actual lanes (spaces between boundaries)
    lane_centers = []
    if len(lane_x_positions) >= 2:
        for i in range(len(lane_x_positions) - 1):
            center = (lane_x_positions[i] + lane_x_positions[i + 1]) // 2
            lane_centers.append(center)
    
    # Lane count = number of actual lanes (centers), not lane lines
    lane_count = len(lane_centers) if len(lane_centers) > 0 else 1
    
    # Find current lane (closest lane center to image center)
    current_lane_index = 0
    if len(lane_centers) > 0:
        image_center_x = w // 2
        distances = [abs(center - image_center_x) for center in lane_centers]
        current_lane_index = np.argmin(distances)
    elif len(lane_x_positions) > 0:
        # If no centers (only 1 lane line detected), use the lane position itself
        image_center_x = w // 2
        distances = [abs(pos - image_center_x) for pos in lane_x_positions]
        current_lane_index = np.argmin(distances)
    
    # Calculate confidence from valid lanes
    if len(valid_lanes) > 0:
        confidences = [np.mean(np.max(keypoints_output[:, :, i], axis=1)) for i in valid_lanes]
        confidence = float(np.mean(confidences))
    else:
        confidence = 0.0
    
    return {
        "lane_count": max(1, lane_count),  # At least 1 lane
        "current_lane_index": current_lane_index,
        "lane_centers": lane_centers,
        "confidence": min(1.0, confidence)
    }

def run_lane_inference(frame: np.ndarray) -> Dict[str, Any]:
    """
    Run lane detection inference.
    
    Returns:
        Dictionary with:
        - lane_mask (optional, for debugging)
        - lane_count
        - current_lane_index
        - lane_centers
        - confidence
        - inference_ms (for telemetry)
    """
    start_time = time.time()
    
    # If no model, return dummy mask
    if interpreter is None:
        load_model()
        if interpreter is None:
            # Generate dummy mask for testing
            try:
                h, w = frame.shape[:2]
                dummy_mask = np.zeros((h, w), dtype=np.uint8)
                # Draw some fake lanes
                import cv2
                cv2.line(dummy_mask, (w//4, h), (w//4, h//2), 255, 20)
                cv2.line(dummy_mask, (w//2, h), (w//2, h//2), 255, 20)
                cv2.line(dummy_mask, (3*w//4, h), (3*w//4, h//2), 255, 20)
                
                result = process_lane_mask(dummy_mask, frame.shape[:2])
                result["inference_ms"] = 0.0
                return result
            except Exception as e:
                print(f"❌ Error generating dummy mask: {e}")
                print(traceback.format_exc())
                # Return safe defaults
                return {
                    "lane_count": 1,
                    "current_lane_index": 0,
                    "lane_centers": [],
                    "confidence": 0.0,
                    "inference_ms": 0.0
                }
    
    try:
        # Preprocess
        processed_frame = preprocess_frame(frame)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        interpreter.invoke()

         # Get keypoints output (UFLD format - not a mask!)
        keypoints_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Use direct keypoints-to-metrics (more reliable than mask conversion)
        result = keypoints_to_lane_metrics(keypoints_output, frame.shape[:2])

        # Add inference time
        inference_ms = (time.time() - start_time) * 1000
        result["inference_ms"] = inference_ms
        
        return result
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        # Return safe defaults
        return {
            "lane_count": 1,
            "current_lane_index": 0,
            "lane_centers": [],
            "confidence": 0.0,
            "inference_ms": 0.0
        }