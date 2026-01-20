from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.database import get_db
from app.models.models import LaneHint
from app.services.aggregation import aggregate_telemetry_to_hints

router = APIRouter()

@router.post("/aggregate")
async def trigger_aggregation(db: Session = Depends(get_db)):
    """Manually trigger aggregation of telemetry to lane hints."""
    try:
        aggregate_telemetry_to_hints(db)
        return {"status": "success", "message": "Aggregation completed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/lane-hints")
async def get_lane_hints(
    geohash: str = Query(..., description="Location geohash"),
    heading: int = Query(None, description="Heading bucket (0-360)"),
    db: Session = Depends(get_db)
):
    """Get aggregated lane hints for a location."""
    query = db.query(LaneHint).filter(LaneHint.geohash == geohash)
    
    if heading is not None:
        query = query.filter(LaneHint.heading_bucket == heading)
    
    hints = query.order_by(LaneHint.sample_count.desc()).first()
    
    if not hints:
        return {"message": "No hints available for this location"}
    
    return {
        "geohash": hints.geohash,
        "heading_bucket": hints.heading_bucket,
        "recommended_lanes": hints.recommended_lanes,
        "confidence": hints.confidence,
        "sample_count": hints.sample_count,
        "last_updated": hints.last_updated.isoformat()
    }