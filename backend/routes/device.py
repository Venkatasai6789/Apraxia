"""
ESP32 Hardware Device API Router
Proxies hardware operations and routes device-recorded audio into Aprixia clinical assessments.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.device import (
    DeviceStatusResponse,
    DeviceRecordingItem,
    DeviceRecordingRenameRequest,
    DeviceRecordingAttachRequest,
    DeviceActionResponse,
)
from backend.schemas.recording import RecordingResponse
from backend.services.device_service import DeviceService
from backend.services.assessment_service import AssessmentService

router = APIRouter(prefix="/api/v1/device", tags=["Hardware Device"])


def get_device_service() -> DeviceService:
    return DeviceService()


@router.get("/status", response_model=DeviceStatusResponse)
async def get_device_status(service: DeviceService = Depends(get_device_service)):
    """Fetch real-time ESP32 board connection and storage status."""
    status_data = await service.get_status()
    return status_data


@router.get("/recordings", response_model=List[DeviceRecordingItem])
async def list_device_recordings(service: DeviceService = Depends(get_device_service)):
    """List all audio files stored in ESP32 LittleFS flash."""
    recordings = await service.list_recordings()
    return recordings


@router.get("/recordings/{filename}/download")
async def download_device_recording(
    filename: str,
    service: DeviceService = Depends(get_device_service)
):
    """Download raw WAV recording from ESP32."""
    wav_bytes = await service.download_recording(filename)
    if not wav_bytes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording '{filename}' not found on device or device unreachable."
        )

    clean_name = service.sanitize_filename(filename)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{clean_name}"',
            "Cache-Control": "no-cache",
        }
    )


@router.delete("/recordings/{filename}", response_model=DeviceActionResponse)
async def delete_device_recording(
    filename: str,
    service: DeviceService = Depends(get_device_service)
):
    """Delete a recording from ESP32 storage."""
    result = await service.delete_recording(filename)
    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to delete recording from device.")
        )
    return result


@router.post("/recordings/{filename}/rename", response_model=DeviceActionResponse)
async def rename_device_recording(
    filename: str,
    payload: DeviceRecordingRenameRequest,
    service: DeviceService = Depends(get_device_service)
):
    """Rename a recording on ESP32 storage."""
    result = await service.rename_recording(filename, payload.new_name)
    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to rename recording on device.")
        )
    return result


@router.post("/recordings/{filename}/attach", response_model=RecordingResponse, status_code=status.HTTP_201_CREATED)
async def attach_device_recording_to_assessment(
    filename: str,
    payload: DeviceRecordingAttachRequest,
    device_service: DeviceService = Depends(get_device_service),
    db: Session = Depends(get_db)
):
    """
    Downloads audio file from ESP32 and ingests it directly into an assessment.
    Reuses existing AssessmentService.add_recording() pipeline without altering ML evaluation.
    """
    assessment_service = AssessmentService(db)
    assessment = assessment_service.get_assessment(payload.assessment_id)
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target assessment '{payload.assessment_id}' not found."
        )

    # 1. Download audio binary from ESP32
    wav_bytes = await device_service.download_recording(filename)
    if not wav_bytes:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not retrieve audio '{filename}' from ESP32 device. Check Wi-Fi connection."
        )

    # 2. Determine target filename (tag with patient info if custom filename given)
    target_filename = payload.custom_filename if payload.custom_filename else filename
    target_filename = device_service.sanitize_filename(target_filename)

    # 3. Feed into existing AssessmentService pipeline
    is_valid, msg, rec = assessment_service.add_recording(
        assessment_id=assessment.assessment_id,
        file_bytes=wav_bytes,
        filename=target_filename
    )

    if not rec:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio ingestion failed: {msg}"
        )

    return rec


@router.post("/record/start", response_model=DeviceActionResponse)
async def start_device_recording(service: DeviceService = Depends(get_device_service)):
    """Remotely trigger recording start on ESP32 hardware."""
    result = await service.start_recording()
    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to start recording on device.")
        )
    return result


@router.post("/record/stop", response_model=DeviceActionResponse)
async def stop_device_recording(service: DeviceService = Depends(get_device_service)):
    """Remotely trigger recording stop on ESP32 hardware."""
    result = await service.stop_recording()
    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to stop recording on device.")
        )
    return result
