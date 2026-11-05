"""
Assessment ORM Model
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(String(64), unique=True, index=True, nullable=False)
    patient_id = Column(String(32), ForeignKey("patients.patient_id"), nullable=False, index=True)
    assessment_reference = Column(String(64), unique=True, index=True, nullable=False)
    clinical_notes = Column(Text, nullable=True)
    assessment_date = Column(String(32), nullable=False)
    status = Column(String(32), default="draft", nullable=False)  # draft, uploaded, processing, completed, partially_completed, failed

    # Model inference quantitative summary
    mean_score = Column(Float, nullable=True)
    median_score = Column(Float, nullable=True)
    min_score = Column(Float, nullable=True)
    max_score = Column(Float, nullable=True)
    token_inconsistency = Column(Float, nullable=True)
    risk_label = Column(String(64), nullable=True)
    action_text = Column(Text, nullable=True)
    utterances_evaluated = Column(Integer, default=0, nullable=False)
    acoustic_profile_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    patient = relationship("Patient", back_populates="assessments")
    recordings = relationship("Recording", back_populates="assessment", cascade="all, delete-orphan")
