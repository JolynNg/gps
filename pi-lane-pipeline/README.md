create virtual environment

run pip install -r requirements.txt

Pip Install:
Mandatory: server will not run without these:
pip install fastapi uvicorn

Required for CORS:
python-multipart is required internally by FastAPI middleware, especially when you later expand APIs.
pip install python-multipart

FastAPI WebSocket:
pip install websockets

OpenCV:
pip install opencv-python numpy

Raspberry Pi
pip install picamera2

TensorFlow / TFLite:
pip install tensorflow tensorflow-lite

Real AI Inference:
pip install tensorflow numpy
lightweight:
pip install tflite-runtime
