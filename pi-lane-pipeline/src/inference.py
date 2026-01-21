
try:
    from . import imp_compat
except (ImportError, ModuleNotFoundError):
    pass

import numpy as np
from typing import Dict, Any, Optional
import time
import os
from .mask_processor import process_lane_mask

# Lazy import TensorFlow
tf = None
interpreter = None
input_details = None
output_details = None

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lane_detection.tflite")
INPUT_SIZE = (640, 480)  # Adjust to your model
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
        except ImportError:
            import tensorflow as tf
    
    model_path = model_path or MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}. Using dummy mask.")
        return
    
    try:
        interpreter = tf.Interpreter(model_path=model_path)
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
            h, w = frame.shape[:2]
            dummy_mask = np.zeros((h, w), dtype=np.uint8)
            # Draw some fake lanes
            cv2 = __import__('cv2')
            cv2.line(dummy_mask, (w//4, h), (w//4, h//2), 255, 20)
            cv2.line(dummy_mask, (w//2, h), (w//2, h//2), 255, 20)
            cv2.line(dummy_mask, (3*w//4, h), (3*w//4, h//2), 255, 20)
            
            result = process_lane_mask(dummy_mask, frame.shape[:2])
            result["inference_ms"] = 0.0
            return result
    
    try:
        # Preprocess
        processed_frame = preprocess_frame(frame)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        interpreter.invoke()
        
        # Get mask output (assuming first output is the mask)
        mask_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Postprocess mask (remove batch dimension, handle shape)
        if len(mask_output.shape) == 4:
            mask = mask_output[0]  # Remove batch dim
        else:
            mask = mask_output
        
        # If mask is multi-channel, take first channel or argmax
        if len(mask.shape) == 3:
            if mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                # Multi-class: take argmax or first class
                mask = np.argmax(mask, axis=2)
        
        # Resize mask to original frame size if needed
        if mask.shape[:2] != frame.shape[:2]:
            import cv2
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
        # Process mask to get lane metrics
        result = process_lane_mask(mask, frame.shape[:2])
        
        # Add inference time
        inference_ms = (time.time() - start_time) * 1000
        result["inference_ms"] = inference_ms
        
        # Optional: include mask for debugging (can be removed for production)
        # result["lane_mask"] = mask.tolist()  # Too large for JSON, skip
        
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