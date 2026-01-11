#!/usr/bin/env python3
"""Camera preview - shows live feed"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera import CameraCapture
import cv2
import time

print("Starting camera preview (press 'q' to quit)...")
camera = CameraCapture()

if camera.camera is None:
    print("❌ Camera failed to initialize!")
    exit(1)

print("✓ Camera initialized - showing preview...")

try:
    while True:
        frame = camera.capture_frame()
        if frame is not None:
            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Resize for display (optional)
            display_frame = cv2.resize(frame_bgr, (800, 600))
            
            # Show frame
            cv2.imshow('Camera Preview', display_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\nStopped by user")

cv2.destroyAllWindows()
camera.release()
print("✓ Preview closed")