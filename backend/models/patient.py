"""
Patient ORM Model
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from backend.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(32), unique=True, index=True, nullable=False)
    child_name = Column(String(128), nullable=True)
    age_years = Column(Integer, nullable=False)
    age_months = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    assessments = relationship("Assessment", back_populates="patient", cascade="all, delete-orphan")
