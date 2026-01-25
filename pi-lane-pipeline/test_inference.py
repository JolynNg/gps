import sys
import os
sys.path.insert(0, 'src')
import numpy as np
import cv2
from src.inference import load_model, run_lane_inference

# Load model
print("Loading model...")
load_model()

# Test with a sample image
test_img_path = '../dataset/tusimple/train_set/clips/0313-2/890/1.jpg'
if os.path.exists(test_img_path):
    print(f"Loading test image: {test_img_path}")
    frame = cv2.imread(test_img_path)
    if frame is not None:
        print("Running inference...")
        result = run_lane_inference(frame)
        print(f"\n✅ Results:")
        print(f"   Lane count: {result['lane_count']}")
        print(f"   Current lane index: {result['current_lane_index']}")
        print(f"   Lane centers: {result['lane_centers']}")
        print(f"   Number of centers: {len(result['lane_centers'])}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Inference time: {result['inference_ms']:.1f}ms")

    else:
        print("❌ Could not load image")
else:
    print(f"⚠️  Test image not found")
    print("   Model should still work with dummy mask")