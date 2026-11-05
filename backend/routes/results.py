"""
Results & Audio Analysis API Router
Connects ML inference execution, screening results, radar speech profiles, and waveform/spectrogram laboratory.
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.results import (
    AssessmentResultsResponse,
    AcousticProfileResponse,
)
from backend.schemas.recording import AudioAnalysisResponse
from backend.services.assessment_service import AssessmentService

router = APIRouter(prefix="/api/v1/assessments/{assessment_id}", tags=["Results & Analysis"])


@router.post("/process")
def process_assessment(assessment_id: str, db: Session = Depends(get_db)):
    """
    Executes real ML inference using openSMILE eGeMAPSv02 + HuBERT Layer 9 + Calibrated Classifier.
    """
    service = AssessmentService(db)
    success, msg, data = service.process_assessment_ml(assessment_id)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg, "data": data}


@router.get("/results", response_model=AssessmentResultsResponse)
def get_assessment_results(assessment_id: str, db: Session = Depends(get_db)):
    """Returns aggregated CAS prediction score, risk label, token inconsistency, and utterance table."""
    service = AssessmentService(db)
    payload = service.get_results_payload(assessment_id)
    if not payload:
        raise HTTPException(status_code=404, detail=f"Assessment '{assessment_id}' not found.")
    return payload


@router.get("/acoustic-profile", response_model=AcousticProfileResponse)
def get_acoustic_profile(assessment_id: str, db: Session = Depends(get_db)):
    """Returns absolute Z-score speech profile dimensions against Typically Developing baselines."""
    service = AssessmentService(db)
    payload = service.get_results_payload(assessment_id)
    if not payload:
        raise HTTPException(status_code=404, detail=f"Assessment '{assessment_id}' not found.")

    return AcousticProfileResponse(
        assessment_id=assessment_id,
        dimensions=payload["acoustic_profile"] or {},
        td_baselines=payload["td_baselines"],
        radial_max=3.0
    )


@router.get("/recordings/{recording_id}/audio-analysis", response_model=AudioAnalysisResponse)
def get_audio_analysis(assessment_id: str, recording_id: str, db: Session = Depends(get_db)):
    """
    Returns high-resolution waveform, 64-band log Mel-spectrogram, pitch contour (F0),
    and key acoustic features for the Audio Analysis Laboratory.
    """
    service = AssessmentService(db)
    analysis = service.get_recording_analysis(assessment_id, recording_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"Audio analysis for recording '{recording_id}' not found or could not be generated."
        )
    return analysis


@router.get("/export")
def export_csv_report(assessment_id: str, db: Session = Depends(get_db)):
    """Exports clinical CSV evaluation dossier matching exact app.py format."""
    service = AssessmentService(db)
    csv_text = service.export_csv(assessment_id)
    if not csv_text:
        raise HTTPException(status_code=404, detail=f"Assessment '{assessment_id}' not found.")

    filename = f"Aprixia_Screening_{assessment_id}.csv"
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
