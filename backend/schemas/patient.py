"""
Patient Pydantic Schemas
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PatientBase(BaseModel):
    child_name: Optional[str] = Field(None, max_length=128)
    age_years: int = Field(..., ge=0, le=18)
    age_months: int = Field(..., ge=0, le=11)


class PatientCreate(PatientBase):
    patient_id: Optional[str] = None  # If provided, can link to existing patient


class PatientResponse(PatientBase):
    id: int
    patient_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
