import os
import sys
import cv2

os.environ["LANE_DEBUG_OVERLAY"] = "1"
os.environ["LANE_DEBUG_OVERLAY_DIR"] = "/Users/jolynng/Project/GPS/pi-lane-pipeline/debug_outputs"

# Make sure we can import from ./src
sys.path.insert(0, "src")


from src.inference import load_model, run_lane_inference  # noqa: E402


def main():
    # Enable debug overlay saving from inference.py
    # (inference.py checks this env var)
    os.environ["LANE_DEBUG_OVERLAY"] = "1"
    # Optional: choose where overlay images are saved
    os.environ["LANE_DEBUG_OVERLAY_DIR"] = os.path.join(os.getcwd(), "debug_outputs")

    print("Loading model...")
    load_model()

    # Test with a sample image
    test_img_path = "../dataset/tusimple/train_set/clips/0313-2/890/1.jpg"
    if not os.path.exists(test_img_path):
        print(f"⚠️  Test image not found: {test_img_path}")
        return

    print(f"Loading test image: {test_img_path}")
    frame = cv2.imread(test_img_path)

    if frame is None:
        print("❌ Could not load image (cv2.imread returned None)")
        return

    print("Running inference...")
    result = run_lane_inference(frame)

    print("\n✅ Results:")
    print(f"   Lane count: {result.get('lane_count')}")
    print(f"   Current lane index: {result.get('current_lane_index')}")
    print(f"   Lane lines (x): {result.get('lane_lines_x', [])}")
    print(f"   Lane centers: {result.get('lane_centers', [])}")
    print(f"   Number of centers: {len(result.get('lane_centers', []))}")
    print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"   Inference time: {result.get('inference_ms', 0.0):.1f}ms")

    # Overlay output path (only present when LANE_DEBUG_OVERLAY=1 and overlay save succeeds)
    overlay_path = result.get("debug_overlay_path")
    if overlay_path:
        print(f"   Debug overlay saved: {overlay_path}")
        # Optional auto-open on macOS (comment out if you don't want it)
        try:
            if sys.platform == "darwin":
                os.system(f"open '{overlay_path}'")
        except Exception:
            pass


if __name__ == "__main__":
    main()
