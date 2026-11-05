"""
Patients API Router
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.patient import Patient
from backend.schemas.patient import PatientCreate, PatientResponse
from backend.services.assessment_service import AssessmentService

router = APIRouter(prefix="/api/v1/patients", tags=["Patients"])


@router.get("/next-id", response_model=dict)
def get_next_patient_id(db: Session = Depends(get_db)):
    """Provides next collision-safe patient ID (e.g. APR-0001)."""
    service = AssessmentService(db)
    next_id = service.generate_patient_id()
    return {"next_patient_id": next_id}


@router.post("", response_model=PatientResponse, status_code=201)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    service = AssessmentService(db)
    patient = service.get_or_create_patient(
        child_name=payload.child_name,
        age_years=payload.age_years,
        age_months=payload.age_months,
        patient_id=payload.patient_id
    )
    return patient


@router.get("", response_model=List[PatientResponse])
def list_patients(
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    query = db.query(Patient)
    if search and search.strip():
        sq = f"%{search.strip().lower()}%"
        query = query.filter((Patient.patient_id.ilike(sq)) | (Patient.child_name.ilike(sq)))
    patients = query.order_by(Patient.created_at.desc()).limit(limit).all()
    return patients


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return patient
