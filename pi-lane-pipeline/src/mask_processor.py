"""
Convert lane mask to lane_count, current_lane_index, lane_centers.
Phase 0 MVP: Simple histogram-based lane detection.
"""
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple

def process_lane_mask(mask: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
    """
    Convert binary lane mask to lane metrics.
    
    Args:
        mask: Binary lane mask (H, W) or (H, W, 1), values 0-255 or 0-1
        frame_shape: Original frame shape (H, W)
        
    Returns:
        Dictionary with lane_count, current_lane_index, lane_centers, confidence
    """
    # Ensure mask is 2D and binary
    if len(mask.shape) == 3:
        mask = mask[:, :, 0] if mask.shape[2] == 1 else mask[:, :, 1]
    
    # Normalize to 0-255 if needed
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # Focus on bottom half (where lanes matter most)
    h, w = mask.shape
    bottom_half = mask[h//2:, :]
    
    # Create histogram: count lane pixels per column
    histogram = np.sum(bottom_half, axis=0)
    
    # Find peaks (lane boundaries)
    # Use threshold to filter noise
    threshold = np.max(histogram) * 0.3
    peaks = []
    
    # Simple peak detection
    for i in range(1, len(histogram) - 1):
        if histogram[i] > threshold:
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                peaks.append(i)
    
    # Group nearby peaks (within 50 pixels) as same lane boundary
    if len(peaks) > 0:
        peaks = _group_peaks(peaks, distance=50)
    
    # Lane count = number of gaps between boundaries
    # Minimum 2 boundaries = 1 lane, 3 boundaries = 2 lanes, etc.
    lane_count = max(1, len(peaks) - 1) if len(peaks) >= 2 else 1
    
    # Calculate lane centers
    lane_centers = []
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            center = (peaks[i] + peaks[i+1]) // 2
            lane_centers.append(center)
    
    # Determine current lane (assume camera is centered)
    # Find which lane center is closest to image center
    current_lane_index = 0
    if len(lane_centers) > 0:
        image_center_x = w // 2
        distances = [abs(center - image_center_x) for center in lane_centers]
        current_lane_index = np.argmin(distances)
    
    # Calculate confidence based on mask quality
    # Higher = more lane pixels, clearer peaks
    mask_coverage = np.sum(mask > 0) / (h * w)
    peak_strength = np.mean([histogram[p] for p in peaks]) / 255.0 if peaks else 0.0
    confidence = min(1.0, (mask_coverage * 0.5 + peak_strength * 0.5))
    
    return {
        "lane_count": int(lane_count),
        "current_lane_index": int(current_lane_index),
        "lane_centers": [int(c) for c in lane_centers],
        "confidence": float(confidence)
    }

def _group_peaks(peaks: List[int], distance: int = 50) -> List[int]:
    """Group nearby peaks together."""
    if not peaks:
        return []
    
    peaks = sorted(peaks)
    grouped = [peaks[0]]
    
    for peak in peaks[1:]:
        if peak - grouped[-1] > distance:
            grouped.append(peak)
        else:
            # Merge: take the stronger peak (keep the last one for simplicity)
            grouped[-1] = peak
    
    return grouped