from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from app.models.database import get_db
from app.models.models import TelemetryEvent
from datetime import datetime

router = APIRouter()

class TelemetryRequest(BaseModel):
    geohash: str
    heading_bucket: int
    speed_bucket: int
    lane_count_estimate: int
    recommended_lanes: List[int]
    confidence: float
    model_version: str = "lane_v0.1"

@router.post("/telemetry")
async def create_telemetry(
    data: TelemetryRequest,
    db: Session = Depends(get_db)
):
    """Save telemetry data from phone."""
    event = TelemetryEvent(
        geohash=data.geohash,
        heading_bucket=data.heading_bucket,
        speed_bucket=data.speed_bucket,
        lane_count_estimate=data.lane_count_estimate,
        recommended_lanes=data.recommended_lanes,
        confidence=data.confidence,
        model_version=data.model_version,
        timestamp=datetime.utcnow()
    )
    
    db.add(event)
    db.commit()
    db.refresh(event)
    
    return {"status": "saved", "id": event.id}