"""
Assessments API Router
"""
from typing import Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.assessment import Assessment
from backend.schemas.assessment import (
    AssessmentCreate,
    AssessmentDetailResponse,
    AssessmentListResponse,
)
from backend.services.assessment_service import AssessmentService

router = APIRouter(prefix="/api/v1/assessments", tags=["Assessments"])


@router.post("", response_model=AssessmentDetailResponse, status_code=201)
def create_assessment(payload: AssessmentCreate, db: Session = Depends(get_db)):
    service = AssessmentService(db)
    assessment = service.create_assessment(
        child_name=payload.child_name,
        age_years=payload.age_years,
        age_months=payload.age_months,
        patient_id=payload.patient_id,
        assessment_reference=payload.assessment_reference,
        clinical_notes=payload.clinical_notes,
    )

    return AssessmentDetailResponse(
        id=assessment.assessment_id,
        patient_id=assessment.patient_id,
        child_name=assessment.patient.child_name if assessment.patient else None,
        age_years=assessment.patient.age_years if assessment.patient else 0,
        age_months=assessment.patient.age_months if assessment.patient else 0,
        assessment_reference=assessment.assessment_reference,
        assessment_date=assessment.assessment_date,
        recording_count=0,
        status=assessment.status,
        clinical_notes=assessment.clinical_notes,
        has_report=False,
        patient=assessment.patient,
        recordings=[],
    )


@router.get("", response_model=AssessmentListResponse)
def list_assessments(
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    service = AssessmentService(db)
    records, total_matching, total_pages, stats = service.list_assessments(
        status=status,
        search_query=search,
        page=page,
        page_size=page_size
    )

    return AssessmentListResponse(
        records=records,
        total_matching=total_matching,
        page=page,
        total_pages=total_pages,
        stats=stats
    )


@router.get("/stats", response_model=Dict[str, int])
def get_assessment_stats(db: Session = Depends(get_db)):
    """Returns aggregated assessment summary statistics directly from SQLite."""
    service = AssessmentService(db)
    stats_data = service.get_dashboard_stats()
    return stats_data["stats"]


@router.get("/{assessment_id}", response_model=AssessmentDetailResponse)
def get_assessment(assessment_id: str, db: Session = Depends(get_db)):
    service = AssessmentService(db)
    assessment = service.get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail=f"Assessment '{assessment_id}' not found.")

    recordings = [
        {
            "recording_id": r.recording_id,
            "assessment_id": r.assessment_id,
            "original_filename": r.original_filename,
            "duration": r.duration,
            "file_size": r.file_size,
            "sample_rate": r.sample_rate,
            "channels": r.channels,
            "validation_status": r.validation_status,
            "processing_status": r.processing_status,
            "prediction_score": r.prediction_score,
            "classification": r.classification,
            "error_message": r.error_message,
            "created_at": r.created_at
        }
        for r in assessment.recordings
    ]

    return AssessmentDetailResponse(
        id=assessment.assessment_id,
        patient_id=assessment.patient_id,
        child_name=assessment.patient.child_name if assessment.patient else None,
        age_years=assessment.patient.age_years if assessment.patient else 0,
        age_months=assessment.patient.age_months if assessment.patient else 0,
        assessment_reference=assessment.assessment_reference,
        assessment_date=assessment.assessment_date,
        recording_count=len(assessment.recordings),
        status=assessment.status,
        clinical_notes=assessment.clinical_notes,
        has_report=assessment.status == "completed",
        mean_score=assessment.mean_score,
        risk_label=assessment.risk_label,
        action_text=assessment.action_text,
        token_inconsistency=assessment.token_inconsistency,
        completed_at=assessment.completed_at,
        patient=assessment.patient,
        recordings=recordings
    )


@router.get("/{assessment_id}/status")
def get_assessment_status(assessment_id: str, db: Session = Depends(get_db)):
    service = AssessmentService(db)
    assessment = service.get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail=f"Assessment '{assessment_id}' not found.")

    return {
        "assessment_id": assessment.assessment_id,
        "status": assessment.status,
        "utterances_evaluated": assessment.utterances_evaluated,
        "mean_score": assessment.mean_score,
        "risk_label": assessment.risk_label
    }


@router.delete("/{assessment_id}", status_code=200)
def delete_assessment(assessment_id: str, db: Session = Depends(get_db)):
    service = AssessmentService(db)
    assessment = service.get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail=f"Assessment '{assessment_id}' not found.")
    db.delete(assessment)
    db.commit()
    return {"message": f"Assessment '{assessment_id}' deleted successfully."}
