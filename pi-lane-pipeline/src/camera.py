import numpy as np
from typing import Optional
import time
from picamera2 import Picamera2

class CameraCapture:
    """
    Camera capture handler for Raspberry Pi Camera Module 3.
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 10):
        """
        Initialize the camera.
        
        Args:
            width: Frame width (default 640)
            height: Frame height (default 480)
            fps: Target FPS (default 10 for 5-10 Hz target)
        """
        self.width = width
        self.height = height
        self.target_fps = fps
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.camera = None
        
        try:
            # Initialize Picamera2
            self.camera = Picamera2()
            
            # Configure camera with optimized settings for lane detection
            config = self.camera.create_preview_configuration(
                main={
                    "size": (width, height),
                    "format": "RGB888"
                }
            )
            self.camera.configure(config)
            
            # Start camera
            self.camera.start()
            print(f"Camera initialized: {width}x{height} @ {fps} FPS")
            
            # Warm up camera (first few frames can be dark)
            time.sleep(2)
            for _ in range(5):
                self.camera.capture_array()
                
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            self.camera = None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Frame as numpy array (RGB format), or None if capture fails
        """
        if self.camera is None:
            return None
        
        try:
            # Capture frame as numpy array (RGB format)
            frame = self.camera.capture_array()
            
            # Update FPS calculation
            current_time = time.time()
            self.frame_count += 1
            elapsed = current_time - self.last_frame_time
            
            if elapsed >= 1.0:  # Update FPS every second
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_frame_time = current_time
            
            return frame
            
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None
    
    def get_fps(self) -> float:
        """Get current FPS estimate."""
        return self.fps
    
    def release(self):
        """Release camera resources."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
            except:
                pass
            self.camera = None
    
    def __del__(self):
        """Cleanup on destruction."""
        self.release()