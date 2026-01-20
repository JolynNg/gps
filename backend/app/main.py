from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.database import engine, Base
from app.routes import telemetry, lane_hints

# Create tables (if not already created)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="GPS Navigation Backend")

# CORS middleware (allows mobile app to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(telemetry.router, prefix="/api/v1", tags=["telemetry"])
app.include_router(lane_hints.router, prefix="/api/v1", tags=["lane-hints"])
app.include_router(lane_hints.router, prefix="/api/v1", tags=["lane-hints"])

@app.get("/")
async def root():
    return {"message": "GPS Navigation Backend API"}

@app.get("/health")
async def health():
    return {"status": "ok", "database": "connected"}