from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.models import TelemetryEvent, LaneHint
from datetime import datetime

def aggregate_telemetry_to_hints(db: Session):
    """Process telemetry events and create/update lane hints."""
    # Group by geohash and heading_bucket
    results = db.query(
        TelemetryEvent.geohash,
        TelemetryEvent.heading_bucket,
        func.array_agg(TelemetryEvent.recommended_lanes).label('all_lanes'),
        func.avg(TelemetryEvent.confidence).label('avg_confidence'),
        func.count(TelemetryEvent.id).label('sample_count')
    ).group_by(
        TelemetryEvent.geohash,
        TelemetryEvent.heading_bucket
    ).having(func.count(TelemetryEvent.id) >= 10).all()  # Need at least 10 samples
    
    for result in results:
        # Find most common recommended_lanes
        # (simplified - you'd need more logic here)
        hint = db.query(LaneHint).filter(
            LaneHint.geohash == result.geohash,
            LaneHint.heading_bucket == result.heading_bucket
        ).first()
        
        if hint:
            # Update existing
            hint.recommended_lanes = result.all_lanes[0]  # Simplified
            hint.confidence = result.avg_confidence
            hint.sample_count = result.sample_count
            hint.last_updated = datetime.utcnow()
        else:
            # Create new
            hint = LaneHint(
                geohash=result.geohash,
                heading_bucket=result.heading_bucket,
                recommended_lanes=result.all_lanes[0],  # Simplified
                confidence=result.avg_confidence,
                sample_count=result.sample_count
            )
            db.add(hint)
    
    db.commit()