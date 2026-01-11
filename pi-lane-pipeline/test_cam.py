#!/usr/bin/env python3
"""Live camera preview with OpenCV (fixed colors)"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera import CameraCapture
import cv2
import time

print("Starting live preview (press 'q' to quit)...")
camera = CameraCapture()

if camera.camera is None:
    print("❌ Camera failed!")
    exit(1)

print("✓ Camera initialized")

try:
    while True:
        frame = camera.capture_frame()
        if frame is not None:
            # Picamera2 returns RGB, OpenCV needs BGR
            # Convert RGB to BGR for correct colors
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Optional: Resize for display
            display_frame = cv2.resize(frame_bgr, (800, 600))
            
            # Show frame
            cv2.imshow('Camera Preview (Press Q to quit)', display_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        time.sleep(0.05)  # Small delay
        
except KeyboardInterrupt:
    print("\nStopped by user")

cv2.destroyAllWindows()
camera.release()
print("✓ Preview closed")