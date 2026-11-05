"""
Recording Pydantic Schemas
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class RecordingBase(BaseModel):
    original_filename: str
    duration: float
    file_size: int
    sample_rate: int = 16000
    channels: int = 1


class RecordingResponse(RecordingBase):
    recording_id: str
    assessment_id: str
    validation_status: str
    processing_status: str
    prediction_score: Optional[float] = None
    classification: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class WaveformData(BaseModel):
    time: List[float]
    amplitude: List[float]
    total_samples: int


class SpectrogramData(BaseModel):
    times: List[float]
    frequencies: List[float]
    db_matrix: List[List[float]]
    n_mels: int = 64
    fmax: int = 8000


class PitchContourData(BaseModel):
    times: List[float]
    f0_hz: List[Optional[float]]


class AcousticFeaturesData(BaseModel):
    f0_mean_hz: Optional[float] = None
    f0_sd_hz: Optional[float] = None
    jitter_pct: Optional[float] = None
    shimmer_db: Optional[float] = None
    hnr_db: Optional[float] = None
    f1_mean_hz: Optional[float] = None
    f2_mean_hz: Optional[float] = None


class AudioAnalysisResponse(BaseModel):
    recording_id: str
    assessment_id: str
    original_filename: str
    duration: float
    sample_rate: int
    prediction_score: Optional[float] = None
    classification: Optional[str] = None
    waveform: WaveformData
    spectrogram: SpectrogramData
    pitch_contour: PitchContourData
    acoustic_features: AcousticFeaturesData
