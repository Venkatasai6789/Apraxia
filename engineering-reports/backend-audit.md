# Aprixia Backend Engineering & Persistence Audit

## 1. Backend Architecture
The backend is structured as a production-grade FastAPI service (`backend/main.py`) powered by SQLAlchemy 2.1 ORM (`backend/models.py`) connecting to a persistent SQLite database located at `backend/aprixia_clinical.db`.

### Core Layers:
1. **API Routers (`backend/routes/`)**:
   - `patients.py`: Patient demographic registration, listing, and lookups.
   - `assessments.py`: Clinical assessment lifecycles (`draft` -> `uploaded` -> `queued` -> `processing` -> `completed` / `failed`).
   - `recordings.py`: Multipart speech audio intake, binary persistence, duration extraction, and validation.
   - `audio_analysis.py`: Waveform and 64-band log-mel spectrogram extraction.
   - `export.py`: Clinical dossier CSV generation.
   - `dashboard.py`: Live database aggregation metrics for clinician dashboard.
2. **Business Services (`backend/services/`)**:
   - `patient_service.py`: CRUD operations on patient records.
   - `assessment_service.py`: Assessment orchestration and summary aggregation.
   - `audio_service.py`: Filename sanitization, directory path traversal prevention, audio header inspection, and duration verification.
   - `inference_service.py`: Serialized execution of audio speech screening through `CASMLEngine`.
   - `serialization_service.py`: RFC 8259-compliant serialization of acoustic matrices.
3. **Database Models (`backend/models.py`)**:
   - `Patient`: Unique medical record number (`mrn`), full name, DOB, sex, referral diagnosis, notes, timestamp.
   - `Assessment`: Foreign key to `Patient`, assessment type, clinical notes, status, aggregate score, classification, duration, timestamps.
   - `Recording`: Foreign key to `Assessment`, file path, original filename, duration, MIME type, individual score, status, error message.

## 2. Server-Side Audio Validation Matrix
All incoming speech uploads undergo rigorous server-side validation before model ingestion:
- **Format Whitelist**: Enforces `.wav`, `.mp3`, `.m4a`, `.flac`. Rejects `.exe`, `.sh`, `.txt`, `.pdf`.
- **Zero-Byte Check**: Explicitly checks file size $> 0$ bytes; rejects empty payloads immediately with HTTP 400.
- **Header & Corruption Check**: Decodes audio headers via `librosa.load(..., sr=16000, mono=True)`. Corrupt chunks trigger a descriptive validation error.
- **Minimum Duration Enforcement**: Audio must be at least 200 ms ($0.20$ seconds). Brief clicks and microphone transients are rejected.
- **Size Boundary**: Bounded to 50 MB per file, streaming in chunks to prevent memory spikes.

## 3. Unit and Integration Test Results
The backend test suite (`backend/tests/`) contains 50 comprehensive tests across:
- `test_parity_and_endpoints.py`: 16 tests covering health, patient CRUD, assessment workflow, audio upload, validation failures, inference pipeline, status polling, and CSV export.
- `test_security_performance.py`: 22 tests verifying path traversal protection, oversize upload prevention, memory leak absence, singleton integrity, and JSON compliance.
- `test_api_contract.py`: 12 tests verifying OpenAPI schema adherence and frontend compatibility.

**Result**: 50/50 tests passed in 34.87s (`Ran 50 tests in 34.875s - OK`).
