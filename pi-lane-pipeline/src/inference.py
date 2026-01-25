try:
    from . import imp_compat  # optional compatibility shim
except (ImportError, ModuleNotFoundError):
    pass

import os
import time
import traceback
from typing import Dict, Any, Optional

import numpy as np

# Lazy import TensorFlow / TFLite Runtime
tf = None
interpreter = None
input_details = None
output_details = None
use_tflite_runtime = None  # set at load_model()

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lane_detection.tflite")
INPUT_SIZE = (800, 288)  # (width, height) expected by this TFLite model
NORMALIZE = True
INPUT_MEAN = 127.5
INPUT_STD = 127.5

# Debug overlay settings (set env var LANE_DEBUG_OVERLAY=1 to enable)
DEBUG_OVERLAY = os.getenv("LANE_DEBUG_OVERLAY", "0") == "1"
DEBUG_OVERLAY_DIR = os.getenv("LANE_DEBUG_OVERLAY_DIR", os.path.join(os.getcwd(), "debug_outputs"))

# If you still want mask-based processing elsewhere, keep this import
# (not used in the keypoints->metrics path below).
try:
    from .mask_processor import process_lane_mask  # noqa: F401
except Exception:
    process_lane_mask = None


def load_model(model_path: Optional[str] = None):
    """Load TFLite model (supports tflite_runtime on Pi and tensorflow on desktop)."""
    global interpreter, tf, input_details, output_details, use_tflite_runtime

    if interpreter is not None:
        return

    if tf is None:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
            tf = tflite
            use_tflite_runtime = True
        except ImportError:
            import tensorflow as tensorflow  # type: ignore
            tf = tensorflow
            use_tflite_runtime = False

    model_path = model_path or MODEL_PATH

    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}. Inference will fallback to safe defaults.")
        return

    try:
        if use_tflite_runtime:
            interpreter = tf.Interpreter(model_path=model_path)
        else:
            interpreter = tf.lite.Interpreter(model_path=model_path)

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ Model loaded: {model_path}")
        # Optional: print input/output shapes once
        try:
            print(f"ℹ️  Input details:  {input_details[0]}")
            print(f"ℹ️  Output details: {output_details[0]}")
        except Exception:
            pass
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        interpreter = None


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess frame for model input.
    NOTE: OpenCV frames are usually BGR. Many models expect RGB.
    If your overlay is consistently shifted/wrong, try uncommenting the BGR->RGB conversion.
    """
    import cv2

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame, INPUT_SIZE)  # (w,h)
    processed = resized.astype(np.float32)

    if NORMALIZE:
        processed = (processed - INPUT_MEAN) / INPUT_STD
    else:
        processed = processed / 255.0

    # Add batch dim
    processed = np.expand_dims(processed, axis=0)
    return processed


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def save_keypoints_overlay(
    frame_bgr: np.ndarray,
    keypoints_output: np.ndarray,
    out_path: str,
    *,
    prob_threshold: float = 0.50,
    draw_lines: bool = True,
) -> None:
    import os
    import cv2
    import numpy as np

    h, w = frame_bgr.shape[:2]

    kp = keypoints_output
    if kp.ndim == 4:
        kp = kp[0]  # (rows, cols, lanes)

    num_rows, num_cols, num_lanes = kp.shape

    # Try full-height first; if it looks vertically shifted, switch back to bottom-half
    y_coords = np.linspace(0, h - 1, num_rows).astype(np.int32)
    # y_coords = np.linspace(h // 2, h - 1, num_rows).astype(np.int32)

    vis = frame_bgr.copy()

    for lane_idx in range(num_lanes):
        lane_logits = kp[:, :, lane_idx]          # (rows, cols)
        lane_prob = softmax(lane_logits, axis=1)  # (rows, cols)

        x_grid = np.argmax(lane_prob, axis=1)     # (rows,)
        pmax = np.max(lane_prob, axis=1)          # (rows,) in [0,1]

        good_rows = pmax > prob_threshold
        rows = np.where(good_rows)[0]
        if len(rows) < 2:
            continue

        # Build raw points
        pts = []
        for r in rows:
            x = int(x_grid[r] * (w - 1) / (num_cols - 1))
            y = int(y_coords[r])
            pts.append((x, y))

        # --- Change 3: continuity filter (REMOVE crazy jumps) ---
        filtered = []
        max_dx = int(0.08 * w)  # allow up to 8% width jump per step
        for p in pts:
            if not filtered:
                filtered.append(p)
                continue
            if abs(p[0] - filtered[-1][0]) <= max_dx:
                filtered.append(p)

        pts = filtered
        if len(pts) < 2:
            continue

        # --- Change 4: smoothing (polyfit) to remove staircase ---
        # Fit x as a quadratic function of y (lane curves)
        if len(pts) >= 6:
            xs = np.array([p[0] for p in pts], dtype=np.float32)
            ys = np.array([p[1] for p in pts], dtype=np.float32)

            try:
                coeff = np.polyfit(ys, xs, 2)  # x = a*y^2 + b*y + c
                ys2 = np.linspace(ys.min(), ys.max(), 50).astype(np.int32)
                xs2 = (coeff[0] * ys2 * ys2 + coeff[1] * ys2 + coeff[2]).astype(np.int32)

                # Clamp to image bounds
                xs2 = np.clip(xs2, 0, w - 1)
                pts = list(zip(xs2.tolist(), ys2.tolist()))
            except Exception:
                # If polyfit fails, keep original pts
                pass

        # Draw after filtering/smoothing
        for (x, y) in pts:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

        if draw_lines and len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i + 1], (0, 255, 0), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def keypoints_to_lane_metrics(
    keypoints_output: np.ndarray,
    frame_shape: tuple,
    *,
    prob_threshold: float = 0.30,
    bottom_k: int = 10,
    min_lane_line_separation_px: int = 40,
) -> Dict[str, Any]:
    """
    Directly extract lane metrics from keypoints without converting to a mask.

    - Converts per-row logits -> probabilities via softmax across num_cols.
    - Uses bottom rows for stable x-position.
    - Returns:
        lane_lines_x: sorted lane boundary x positions (pixels)
        lane_count: number of lanes (spaces) = max(1, len(lane_lines_x) - 1)
        lane_centers: centers between adjacent lane lines
        current_lane_index: nearest center to image center
        confidence: mean confidence of selected lane lines (0..1)
    """
    h, w = frame_shape

    kp = keypoints_output
    if kp.ndim == 4:
        kp = kp[0]  # (rows, cols, lanes)

    num_rows, num_cols, num_lanes = kp.shape

    lane_lines_x = []
    lane_line_conf = []

    for lane_idx in range(num_lanes):
        lane_logits = kp[:, :, lane_idx]          # (rows, cols)
        lane_prob = softmax(lane_logits, axis=1)  # (rows, cols)

        x_grid = np.argmax(lane_prob, axis=1)     # (rows,)
        pmax = np.max(lane_prob, axis=1)          # (rows,) in [0,1]

        # Focus on bottom rows (closer to car, generally more reliable)
        b = bottom_k if num_rows >= bottom_k else num_rows
        xg_bottom = x_grid[-b:]
        p_bottom = pmax[-b:]

        good = p_bottom > prob_threshold
        if not np.any(good):
            continue

        # Use median for robustness against outliers
        xg_med = float(np.median(xg_bottom[good]))
        conf_mean = float(np.mean(p_bottom[good]))

        # Map grid -> pixel
        x_px = int(xg_med * (w - 1) / (num_cols - 1))

        lane_lines_x.append(x_px)
        lane_line_conf.append(conf_mean)

    # Sort lane lines left -> right
    if lane_lines_x:
        order = np.argsort(lane_lines_x)
        lane_lines_x = [lane_lines_x[i] for i in order]
        lane_line_conf = [lane_line_conf[i] for i in order]

    # Deduplicate lane lines that are too close (collapse to one)
    filtered_lines = []
    filtered_conf = []
    for x, c in zip(lane_lines_x, lane_line_conf):
        if not filtered_lines:
            filtered_lines.append(x)
            filtered_conf.append(c)
            continue
        if abs(x - filtered_lines[-1]) >= min_lane_line_separation_px:
            filtered_lines.append(x)
            filtered_conf.append(c)
        else:
            # Keep the one with higher confidence
            if c > filtered_conf[-1]:
                filtered_lines[-1] = x
                filtered_conf[-1] = c

    lane_lines_x = filtered_lines
    lane_line_conf = filtered_conf

    # Lanes are spaces between lane lines (boundaries)
    lane_count = max(1, len(lane_lines_x) - 1)

    # Centers between adjacent lane lines
    lane_centers = []
    if len(lane_lines_x) >= 2:
        for i in range(len(lane_lines_x) - 1):
            lane_centers.append((lane_lines_x[i] + lane_lines_x[i + 1]) // 2)

    # Current lane index: closest lane center to image center
    image_center_x = w // 2
    current_lane_index = 0
    if lane_centers:
        distances = [abs(c - image_center_x) for c in lane_centers]
        current_lane_index = int(np.argmin(distances))
    elif lane_lines_x:
        # Fallback: if only one line detected, pick nearest line (not ideal, but safe)
        distances = [abs(x - image_center_x) for x in lane_lines_x]
        current_lane_index = int(np.argmin(distances))

    confidence = float(np.mean(lane_line_conf)) if lane_line_conf else 0.0

    return {
        "lane_lines_x": lane_lines_x,                # boundaries in pixels
        "lane_count": lane_count,                    # lanes = boundaries - 1
        "current_lane_index": current_lane_index,    # index into lane_centers
        "lane_centers": lane_centers,                # lane centers in pixels
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
    }


def run_lane_inference(frame: np.ndarray) -> Dict[str, Any]:
    """
    Run lane detection inference.

    Returns dict with:
      - lane_count
      - current_lane_index
      - lane_centers
      - lane_lines_x (debug/useful)
      - confidence
      - inference_ms
      - debug_overlay_path (only when LANE_DEBUG_OVERLAY=1)
    """
    start_time = time.time()
    
    print("DEBUG_OVERLAY =", DEBUG_OVERLAY)
    print("DEBUG_OVERLAY_DIR =", DEBUG_OVERLAY_DIR)

    if interpreter is None:
        load_model()
        if interpreter is None:
            # Safe defaults when model isn't available
            return {
                "lane_count": 1,
                "current_lane_index": 0,
                "lane_centers": [],
                "lane_lines_x": [],
                "confidence": 0.0,
                "inference_ms": 0.0,
            }

    try:
        processed_frame = preprocess_frame(frame)

        interpreter.set_tensor(input_details[0]["index"], processed_frame)
        interpreter.invoke()

        keypoints_output = interpreter.get_tensor(output_details[0]["index"])

        # Optional debug overlay saved to disk
        debug_overlay_path = None
        if DEBUG_OVERLAY:
            os.makedirs(DEBUG_OVERLAY_DIR, exist_ok=True)
            ts = int(time.time() * 1000)
            debug_overlay_path = os.path.join(DEBUG_OVERLAY_DIR, f"overlay_{ts}.jpg")
            try:
                save_keypoints_overlay(frame, keypoints_output, debug_overlay_path, prob_threshold=0.30)
            except Exception as e:
                print(f"⚠️  Failed to save overlay: {e}")

        # Metrics directly from keypoints
        result = keypoints_to_lane_metrics(
            keypoints_output,
            frame.shape[:2],
            prob_threshold=0.30,
            bottom_k=10,
            min_lane_line_separation_px=40,
        )

        inference_ms = (time.time() - start_time) * 1000.0
        result["inference_ms"] = float(inference_ms)

        if debug_overlay_path is not None:
            result["debug_overlay_path"] = debug_overlay_path

        return result

    except Exception as e:
        print(f"❌ Inference error: {e}")
        print(traceback.format_exc())
        return {
            "lane_count": 1,
            "current_lane_index": 0,
            "lane_centers": [],
            "lane_lines_x": [],
            "confidence": 0.0,
            "inference_ms": float((time.time() - start_time) * 1000.0),
        }
