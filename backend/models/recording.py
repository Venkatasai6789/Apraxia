"""
Recording ORM Model
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    recording_id = Column(String(64), unique=True, index=True, nullable=False)
    assessment_id = Column(String(64), ForeignKey("assessments.assessment_id"), nullable=False, index=True)
    original_filename = Column(String(256), nullable=False)
    stored_filename = Column(String(256), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    sample_rate = Column(Integer, default=16000, nullable=False)
    channels = Column(Integer, default=1, nullable=False)

    validation_status = Column(String(32), default="valid", nullable=False)  # valid, warning, invalid
    processing_status = Column(String(32), default="pending", nullable=False)  # pending, processing, completed, failed
    prediction_score = Column(Float, nullable=True)
    classification = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)

    # Cached JSON payload containing downsampled waveform, 64-band spectrogram, pitch contour, acoustic features
    analysis_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    assessment = relationship("Assessment", back_populates="recordings")
