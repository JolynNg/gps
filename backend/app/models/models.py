#defines table structure (columns, types, constraints)
#maps python classes to postgreSQL tables
# used by code to query and insert data
from sqlalchemy import Column, Integer, String, Float, ARRAY, TIMESTAMP, Index
from datetime import datetime
from .database import Base

class TelemetryEvent(Base):
    """Raw telemetry data from phones."""
    __tablename__ = "telemetry_events"

    id = Column(Integer, primary_key=True, index=True)
    geohash = Column(String(10), index=True)
    heading_bucket = Column(Integer)
    speed_bucket = Column(Integer)
    lane_count_estimate = Column(Integer)
    recommended_lanes = Column(ARRAY(Integer))
    confidence = Column(Float)
    model_version = Column(String(50))
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)

    # Index for faster queries
    __table_args__ = (
        Index('idx_geohash_heading', 'geohash', 'heading_bucket'),
    )

class LaneHint(Base):
    """Aggregated lane hints computed from telemetry."""
    __tablename__ = "lane_hints"

    id = Column(Integer, primary_key=True, index=True)
    geohash = Column(String(10), index=True)
    heading_bucket = Column(Integer)
    recommended_lanes = Column(ARRAY(Integer))
    confidence = Column(Float)
    sample_count = Column(Integer)
    last_updated = Column(TIMESTAMP, default=datetime.utcnow)

    # Index for faster queries
    __table_args__ = (
        Index('idx_geohash_heading_hint', 'geohash', 'heading_bucket'),
    )