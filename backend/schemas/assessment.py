"""
Assessment Pydantic Schemas
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from backend.schemas.patient import PatientResponse
from backend.schemas.recording import RecordingResponse


class AssessmentCreate(BaseModel):
    child_name: Optional[str] = Field(None, max_length=128)
    age_years: int = Field(..., ge=0, le=18)
    age_months: int = Field(..., ge=0, le=11)
    patient_id: Optional[str] = None  # Reuses existing patient or generates new
    assessment_reference: Optional[str] = None
    clinical_notes: Optional[str] = Field(None, max_length=500)


class AssessmentSummaryStats(BaseModel):
    total: int
    completed: int
    pending: int
    flagged: int
    total_patients: Optional[int] = None


class AssessmentItemResponse(BaseModel):
    id: str  # assessment_id
    patient_id: str
    child_name: Optional[str]
    age_years: int
    age_months: int
    assessment_reference: str
    assessment_date: str
    recording_count: int
    status: str
    clinical_notes: Optional[str]
    has_report: bool
    mean_score: Optional[float] = None
    risk_label: Optional[str] = None


class AssessmentDetailResponse(AssessmentItemResponse):
    patient: Optional[PatientResponse]
    recordings: List[RecordingResponse] = []
    token_inconsistency: Optional[float] = None
    action_text: Optional[str] = None
    completed_at: Optional[datetime] = None


class AssessmentListResponse(BaseModel):
    records: List[AssessmentItemResponse]
    total_matching: int
    page: int
    total_pages: int
    stats: AssessmentSummaryStats
