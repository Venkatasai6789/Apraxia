"""
Assessment Results & Acoustic Profile Pydantic Schemas
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SubgroupProfile(BaseModel):
    name: str
    patient_z_score: float
    td_baseline_z_score: float


class DimensionProfile(BaseModel):
    dimension_name: str
    subgroups: Dict[str, float]
    td_baselines: Dict[str, float]


class AcousticProfileResponse(BaseModel):
    assessment_id: str
    dimensions: Dict[str, Dict[str, float]]
    td_baselines: Dict[str, Dict[str, float]]
    radial_max: float = 3.0


class UtteranceResultItem(BaseModel):
    recording_id: str
    audio_file: str
    duration: float
    prediction_score: Optional[float]
    classification: Optional[str]
    status: str


class AssessmentResultsResponse(BaseModel):
    assessment_id: str
    assessment_reference: str
    patient_id: str
    child_name: Optional[str]
    age_formatted: str
    assessment_date: str
    status: str

    mean_score: Optional[float]
    median_score: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    token_inconsistency: Optional[float]
    risk_label: Optional[str]
    action_text: Optional[str]
    utterances_evaluated: int

    acoustic_profile: Optional[Dict[str, Dict[str, float]]]
    td_baselines: Dict[str, Dict[str, float]]

    utterance_results: List[UtteranceResultItem]
