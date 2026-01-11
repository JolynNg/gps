#!/usr/bin/env python3
"""Camera test - save with PIL (correct colors)"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera import CameraCapture
from PIL import Image
import time

print("Testing camera...")
camera = CameraCapture()

if camera.camera is None:
    print("❌ Camera failed!")
    exit(1)

print("✓ Camera initialized")
time.sleep(2)

frame = camera.capture_frame()

if frame is not None:
    # Convert numpy array to PIL Image (PIL expects RGB)
    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    
    # Save with PIL (preserves RGB correctly)
    image.save('test_camera_pil.jpg', 'JPEG')
    print("✓ Image saved with PIL: test_camera_pil.jpg")
    
    # Also try saving with OpenCV for comparison
    import cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test_camera_opencv.jpg', frame_bgr)
    print("✓ Image saved with OpenCV: test_camera_opencv.jpg")
    print("  Compare both images to see which has correct colors")
else:
    print("❌ Failed to capture")

camera.release()
print("✓ Done!")