#!/usr/bin/env python3
"""Quick camera test script"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera import CameraCapture
import time

print("Testing camera...")
camera = CameraCapture()

if camera.camera is None:
    print("❌ Camera failed to initialize!")
    exit(1)

print("✓ Camera initialized")
time.sleep(1)

for i in range(5):
    frame = camera.capture_frame()
    if frame is not None:
        print(f"Frame {i+1}: {frame.shape}, FPS: {camera.get_fps():.1f}")
    else:
        print(f"Frame {i+1}: FAILED")
    time.sleep(0.2)

camera.release()
print("✓ Test complete!")