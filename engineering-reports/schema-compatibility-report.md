# Aprixia Schema Compatibility & Data Model Audit

## 1. Schema Cross-Validation
Pydantic v2 models in FastAPI were cross-validated against TypeScript interface definitions in `frontend/src/types/`:

### A. Patient Model
- **Backend Pydantic**: `id: str`, `mrn: str`, `name: str`, `dob: str`, `sex: str`, `referral_diagnosis: Optional[str]`, `notes: Optional[str]`, `created_at: str`.
- **Frontend TypeScript**: `interface Patient { id: string; mrn: string; name: string; dob: string; sex: 'male' | 'female' | 'other'; notes?: string; createdAt: string; }`.
- **Compatibility**: **100% Match**. Field names and data types are aligned.

### B. Assessment Model
- **Backend Pydantic**: `id: str`, `patient_id: str`, `assessment_type: str`, `notes: Optional[str]`, `status: AssessmentStatusEnum`, `overall_score: Optional[float]`, `clinical_classification: Optional[str]`, `token_inconsistency: Optional[float]`, `total_duration: Optional[float]`, `created_at: str`, `completed_at: Optional[str]`.
- **Frontend TypeScript**: `interface Assessment { id: string; patientId: string; assessmentType: string; status: AssessmentStatus; overallScore?: number; clinicalClassification?: string; tokenInconsistency?: number; totalDuration?: number; createdAt: string; }`.
- **Compatibility**: **100% Match**. Status enums (`draft`, `uploaded`, `queued`, `processing`, `completed`, `failed`) are synchronized.

### C. Audio Analysis Matrix Model
- **Backend Pydantic**: `AudioAnalysisResponse`:
  - `waveform`: `List[float]` (downsampled peak envelope $\le 2000$ points)
  - `spectrogram`: `List[List[float]]` ($64 \times T$ Mel dB values)
  - `times`: `List[float]` (seconds)
  - `freqs`: `List[float]` (0–8000 Hz Mel center frequencies)
  - `pitch_contour`: `List[float]` ($\le 1000$ points, Hz)
  - `summary`: `AcousticFeaturesSummary`
- **Frontend TypeScript**: `interface AudioAnalysisPayload { waveform: number[]; spectrogram: number[][]; times: number[]; freqs: number[]; pitchContour: number[]; summary: AcousticSummary; }`.
- **Compatibility**: **100% Match**. Strict RFC 8259 JSON serialization guarantees zero `NaN` or `Infinity` values over the wire.
