"""
Pydantic schemas for ESP32 Hardware Device Proxy.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class DeviceStatusResponse(BaseModel):
    connected: bool
    recording: bool = False
    current_file: Optional[str] = None
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    file_count: int = 0
    max_record_seconds: int = 30
    sample_rate: int = 16000
    uptime_ms: int = 0
    wifi_clients: int = 0
    error: Optional[str] = None


class DeviceRecordingItem(BaseModel):
    name: str
    size: int
    duration_sec: float = 0.0


class DeviceRecordingRenameRequest(BaseModel):
    new_name: str = Field(..., min_length=1, max_length=64, description="New filename for the recording")


class DeviceRecordingAttachRequest(BaseModel):
    assessment_id: str = Field(..., description="Target assessment ID to attach the recording to")
    custom_filename: Optional[str] = Field(None, max_length=100, description="Optional custom filename tagged with patient details")


class DeviceActionResponse(BaseModel):
    ok: bool
    message: Optional[str] = None
    file: Optional[str] = None
    bytes: Optional[int] = None
    error: Optional[str] = None
