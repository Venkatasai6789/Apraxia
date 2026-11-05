"""
Recordings API Router
Handles multi-file uploads, validation, and audio streaming.
"""
import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.recording import Recording
from backend.schemas.recording import RecordingResponse
from backend.services.assessment_service import AssessmentService

router = APIRouter(prefix="/api/v1/assessments/{assessment_id}/recordings", tags=["Recordings"])


@router.post("", response_model=List[RecordingResponse], status_code=201)
async def upload_recordings(
    assessment_id: str,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    service = AssessmentService(db)
    assessment = service.get_assessment(assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail=f"Assessment '{assessment_id}' not found.")

    uploaded_records = []
    for file in files:
        file_bytes = await file.read()
        is_valid, msg, rec = service.add_recording(
            assessment_id=assessment.assessment_id,
            file_bytes=file_bytes,
            filename=file.filename or "recording.wav"
        )
        if rec:
            uploaded_records.append(rec)

    return uploaded_records


@router.get("/{recording_id}", response_model=RecordingResponse)
def get_recording(assessment_id: str, recording_id: str, db: Session = Depends(get_db)):
    rec = db.query(Recording).filter(
        Recording.assessment_id == assessment_id,
        Recording.recording_id == recording_id
    ).first()
    if not rec:
        raise HTTPException(status_code=404, detail=f"Recording '{recording_id}' not found.")
    return rec


@router.get("/{recording_id}/audio")
def stream_recording_audio(assessment_id: str, recording_id: str, db: Session = Depends(get_db)):
    """Streams original uploaded audio directly to browser audio elements."""
    rec = db.query(Recording).filter(
        Recording.assessment_id == assessment_id,
        Recording.recording_id == recording_id
    ).first()
    if not rec or not Path(rec.file_path).exists():
        raise HTTPException(status_code=404, detail=f"Audio file for recording '{recording_id}' not found.")

    ext = Path(rec.file_path).suffix.lower()
    media_type = "audio/wav"
    if ext == ".mp3":
        media_type = "audio/mpeg"
    elif ext == ".m4a":
        media_type = "audio/mp4"
    elif ext == ".flac":
        media_type = "audio/flac"

    return FileResponse(
        path=rec.file_path,
        media_type=media_type,
        filename=rec.original_filename
    )


@router.delete("/{recording_id}", status_code=200)
def delete_recording(assessment_id: str, recording_id: str, db: Session = Depends(get_db)):
    rec = db.query(Recording).filter(
        Recording.assessment_id == assessment_id,
        Recording.recording_id == recording_id
    ).first()
    if not rec:
        raise HTTPException(status_code=404, detail=f"Recording '{recording_id}' not found.")

    if Path(rec.file_path).exists():
        try:
            os.remove(rec.file_path)
        except Exception:
            pass

    db.delete(rec)
    db.commit()
    return {"message": f"Recording '{recording_id}' removed."}
