# Aprixia — Childhood Apraxia of Speech (CAS) Screening & Acoustic Analysis Platform

> **Clinical Screening Disclaimer:** Aprixia is a machine learning–assisted clinical screening and acoustic research support system. It is **not** a standalone diagnostic medical device and must be used only to support—never replace—evaluation by a qualified Speech-Language Pathologist (SLP) using standardized clinical protocols (e.g., VMPAC, DEMSS).

Aprixia is a local-first, full-stack clinical speech screening platform for **Childhood Apraxia of Speech (CAS)**. It connects a **React 19 + TypeScript** clinical web interface to a **FastAPI + SQLite** persistence backend and a shared multimodal Python ML engine (**HuBERT Hidden Layer 9 + openSMILE eGeMAPSv02**) that preserves numerical parity with the research Streamlit prototype.

---

## Key Capabilities

- **Multimodal CAS Screening Engine (`CASMLEngine`)**: Combines **88** `openSMILE eGeMAPSv02` acoustic functionals with **768** mean-pooled `facebook/hubert-base-ls960` Hidden Layer 9 contextual embeddings into an **856-dimensional** feature space scored by a calibrated `LogisticRegression` classifier (`C=0.1`, balanced class weights, L2 penalty).
- **100% Local Audio Processing**: Speech audio validation, feature extraction, and transformer inference run entirely on the local machine—no patient audio or clinical metadata is transmitted to external cloud AI services.
- **Multi-Utterance Assessment & Token Inconsistency**: Evaluates single or batch speech recordings per assessment, aggregates mean, median, min, and max risk scores, and computes trial-to-trial **token inconsistency** ($\sigma$) across valid utterances while isolating corrupt or sub-200 ms recordings.
- **3-Domain Acoustic Radar Profile**: Computes absolute Z-score deviations from Typically Developing (TD) pediatric reference baselines (`td_baselines.joblib`) across **Pitch / Prosody**, **Voice Quality**, and **Vowel Formants**.
- **Synchronized Audio Analysis Laboratory**: Interactive Plotly inspection of time-synchronized **amplitude waveform** ($\le 2,000$ downsampled points), **64-band log-mel spectrogram** ($0\text{–}8,000\text{ Hz}$), and **$F_0$ pitch contour** (`librosa.pyin`), with synchronized zoom and reset across all subplots.
- **Persistent Clinical Dossiers & CSV Export**: Stores patients, assessment state transitions, uploaded audio metadata, and cached acoustic payloads in SQLite (`backend/aprixia_clinical.db`), with one-click per-assessment CSV export.
- **Streamlit Reference Coexistence**: Both the FastAPI backend and the standalone Streamlit interface (`ml-child-speech-disorder-classification/app.py`) share `ml_engine.py`, maintaining numerical divergence $< 10^{-10}$.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ React 19 + TypeScript Frontend (Vite :5173)                                 │
│  • Dashboard (/)  • Patients (/patients)  • Archive (/assessments)          │
│  • Intake (/assessments/new)  • Review (/assessments/review)                │
│  • Results Dossier (/assessments/:id)                                       │
│  • Synchronized Audio Lab (/assessments/:id/laboratory)                     │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ REST / Multipart Upload (Axios)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Backend (Uvicorn :8000)                                             │
│  • Routers: patients, assessments, recordings, results, dashboard           │
│  • Validation: .wav/.mp3/.m4a/.flac, <= 50 MB, >= 200 ms, path confinement  │
│  • Persistence: SQLAlchemy ORM -> SQLite (backend/aprixia_clinical.db)      │
│  • Audio Storage: UUID-prefixed files in backend/uploads/{assessment_id}/   │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ Thread-Safe Singleton (INFERENCE_LOCK)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Shared ML Engine (ml-child-speech-disorder-classification/ml_engine.py)     │
│  1. Audio Normalization -> 16,000 Hz mono                                   │
│  2. openSMILE eGeMAPSv02 Functionals (88 dims)                              │
│  3. HuBERT-Base-LS960 Hidden Layer 9 Mean Embedding (768 dims)              │
│  4. Alignment (cas_feature_cols) -> StandardScaler -> Calibrated LogReg     │
│  5. Acoustic Radar Z-Scores vs. TD Baselines + Waveform / Mel / F0 Payloads │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Clinical Risk Classification Thresholds

Implemented identically in [`calculate_aggregate_metrics`](ml-child-speech-disorder-classification/ml_engine.py):

| Mean Risk Score | Risk Classification | Recommended Clinical Action |
|---|---|---|
| **$\ge 65.0\%$** | `Elevated CAS Risk` | Comprehensive Motor Speech Evaluation (VMPAC / DEMSS) Recommended |
| **$40.0\% \text{ – } 64.99\%$** | `Borderline / Monitoring Needed` | Active Clinical Monitoring & Re-Screening Advised |
| **$< 40.0\%$** | `Low Risk / Typical Speech Sound Development` | Acoustic Parameters Consistent with Typical Pediatric Norms |

### Validation & Parity Highlights

- **Leave-One-Speaker-Out (LOSO) Balanced Accuracy**: **94.48%** (reported by `/api/v1/health` and the upgraded model evaluation suite).
- **Cross-Implementation Parity**: Tested on benchmark clinical samples (`sample_cas_child_03F.wav` scores **99.9996%**, `sample_non_cas_speech_delay_01M.wav` scores **0.0003%**, `sample_typically_developing_01M.wav` scores **0.0164%**) with $< 10^{-10}$ numerical divergence between FastAPI and Streamlit.
- **CPU Inference Latency**: Mean **417 ms** per utterance (warm `CASMLEngine` singleton).

---

## Repository Structure

```text
Apraxia/
├── backend/                                  # FastAPI application & SQLite persistence
│   ├── main.py                               # App entrypoint, CORS, lifespan ML warmup & seeding
│   ├── database.py                           # SQLAlchemy engine & session dependency
│   ├── requirements.txt                      # Core backend API dependencies
│   ├── models/                               # ORM models: Patient, Assessment, Recording
│   ├── routes/                               # REST routers (patients, assessments, recordings, results, dashboard)
│   ├── schemas/                              # Pydantic request/response contracts
│   ├── services/                             # Assessment orchestration, audio validation, RFC 8259 JSON serialization
│   └── tests/                                # 50 unittest tests (parity/endpoints, API contract, security/perf)
├── frontend/                                 # React 19 + TypeScript + Vite clinical UI
│   ├── src/
│   │   ├── app/                              # App shell and React Router definitions
│   │   ├── components/                       # Dashboard, intake, results radar, and synchronized Plotly lab
│   │   ├── hooks/                            # Filter and client-side audio metadata hooks
│   │   ├── pages/                            # Route views for dashboard, patients, assessments, and audio lab
│   │   ├── services/                         # Axios client (VITE_API_BASE_URL) and API service layer
│   │   └── styles/                           # Tailwind CSS v4 & clinical design tokens
│   ├── tests/clinical_browser_e2e.spec.ts    # Playwright end-to-end browser test suite (10 scenarios)
│   ├── playwright.config.ts                  # Playwright configuration (Chromium, backend/frontend webServers)
│   └── package.json                          # Frontend dependencies & scripts
├── ml-child-speech-disorder-classification/  # Shared ML engine, research notebooks & reference Streamlit UI
│   ├── ml_engine.py                          # Thread-safe CASMLEngine singleton shared with FastAPI
│   ├── app.py                                # Standalone Streamlit screening application
│   ├── test_ml_parity.py                     # 18 ML pipeline parity tests
│   ├── requirements.txt                      # Full ML/research Python environment (Torch, Transformers, openSMILE, etc.)
│   ├── *.joblib                              # Pre-trained classifier, scaler, column order & TD baseline artifacts
│   └── samples/                              # Benchmark reference audio samples (if present locally)
├── engineering-reports/                      # Architecture, data lineage, ML integrity, QA & security audit reports
├── .gitignore                                # Strict data-leakage & artifact exclusions
└── README.md
```

---

## Privacy, Security & Data-Leakage Safeguards

- **Local-Only Inference**: Once `facebook/hubert-base-ls960` weights are cached locally by Hugging Face `transformers`, inference runs completely offline with zero outbound audio or patient transmissions.
- **Upload Security Boundaries** (`backend/services/audio_service.py`):
  - Allowed extensions whitelist: `.wav`, `.mp3`, `.m4a`, `.flac`.
  - Maximum upload size: **50 MB**; zero-byte files rejected; minimum duration **$200\text{ ms}$** enforced via `librosa` decoding.
  - Filenames stripped of traversal sequences, drive letters, and control characters (`sanitize_filename`), prefixed with UUIDs, and confined strictly inside `backend/uploads/`.
- **Strict `.gitignore` Protections**:
  - **Clinical Persistence & Exports**: `backend/aprixia_clinical.db*`, `*.db`, `*.db-wal`, `*.db-shm`, `*.sqlite*`, `backend/uploads/`, `exports/`, `Aprixia_Screening_*.csv`, `cas_clinical_screening_report.csv`.
  - **Audio & Research Corpora**: `test_datasets/`, `core_upx/`, `core_uxssd/`, `core_uxtd/`, `*_processed_child_wav/`, and all audio files (`*.wav`, `*.mp3`, `*.m4a`, `*.flac`, `*.ogg`, `*.webm`, etc.).
  - **Tabular Data & Large Tensors**: `*.csv`, `*.tsv`, `*.parquet`, `*.xlsx`, `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.onnx`, `*.h5`, `*.pkl`.
  - **Secrets & Test Artifacts**: `.env`, `.env.*` (except `.env.example`), `*.pem`, `*.key`, `credentials.json`, `secrets.json`, `frontend/test-results/`, `frontend/playwright-report/`, `.venv/`, `node_modules/`, `.agents/`.

> **Note on Research Corpora:** Raw UltraSuite (`UPX`, `UXSSD`, `UXTD`), ENNI, and TORGO datasets are excluded from version control and must be obtained directly from their respective data providers under their licensing terms.

---

## Getting Started

### Prerequisites

- **Python 3.11+** (64-bit)
- **Node.js 18+** and **npm 9+**
- **Internet access on first ML startup** (only if `facebook/hubert-base-ls960` is not yet cached in your local Hugging Face cache; subsequent runs work offline)

### 1. Python Environment Setup (Backend + ML Engine)

From the repository root (`Apraxia/`):

```powershell
# Create and activate a virtual environment (or reuse ml-child-speech-disorder-classification\.venv if already present)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install full ML stack (PyTorch, Transformers, openSMILE, librosa, scikit-learn, Streamlit) and FastAPI backend deps
python -m pip install --upgrade pip
pip install -r ml-child-speech-disorder-classification/requirements.txt
pip install -r backend/requirements.txt
```

*(On macOS/Linux, activate with `source .venv/bin/activate`.)*

### 2. Start the FastAPI Backend

Run from the **repository root** so Python resolves both `backend.*` and `ml-child-speech-disorder-classification`:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- API base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`
- Health endpoint: `GET http://127.0.0.1:8000/api/v1/health`

> On first launch with an empty database, `backend/main.py` initializes `backend/aprixia_clinical.db` and seeds initial benchmark cases (`APR-0048`, `APR-0049`, `APR-0050`) if the reference WAV files exist in `ml-child-speech-disorder-classification/samples/`.

### 3. Start the React Frontend

In a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

- Frontend URL: `http://localhost:5173`
- Optional configuration: create `frontend/.env.local` (ignored by Git) to override the backend URL:
  ```env
  VITE_API_BASE_URL=http://127.0.0.1:8000
  ```

### 4. (Optional) Launch the Reference Streamlit Prototype

```powershell
cd ml-child-speech-disorder-classification
streamlit run app.py
```

---

## REST API Overview

| Domain | Method & Endpoint | Purpose |
|---|---|---|
| **Health** | `GET /api/v1/health` | Service readiness, ML warmup status, model architecture metadata |
| **Dashboard** | `GET /api/v1/dashboard/stats` | Aggregated clinical summary counts and recent activity |
| | `GET /api/v1/dashboard/overview` | Formatted dashboard overview KPIs and screening trend series |
| **Patients** | `GET /api/v1/patients/next-id` | Generate next collision-safe patient identifier (`APR-XXXX`) |
| | `POST /api/v1/patients` | Register or retrieve a pediatric patient record |
| | `GET /api/v1/patients` | List/search registered patients (`?search=&limit=`) |
| | `GET /api/v1/patients/{patient_id}` | Fetch single patient details |
| **Assessments** | `POST /api/v1/assessments` | Create a new screening assessment dossier |
| | `GET /api/v1/assessments` | Paginated archive (`?status=&search=&page=&page_size=`) |
| | `GET /api/v1/assessments/stats` | Assessment status breakdown counts |
| | `GET /api/v1/assessments/{assessment_id}` | Full assessment detail with linked patient and recordings |
| | `GET /api/v1/assessments/{assessment_id}/status` | Lightweight processing status polling endpoint |
| | `DELETE /api/v1/assessments/{assessment_id}` | Delete assessment and cascade recordings |
| **Recordings** | `POST /api/v1/assessments/{assessment_id}/recordings` | Multipart upload (`files`) with format, size, and duration validation |
| | `GET /api/v1/assessments/{assessment_id}/recordings/{recording_id}` | Recording validation and inference metadata |
| | `GET /api/v1/assessments/{assessment_id}/recordings/{recording_id}/audio` | Stream stored audio recording |
| | `DELETE /api/v1/assessments/{assessment_id}/recordings/{recording_id}` | Remove recording metadata and stored audio file |
| **ML & Results** | `POST /api/v1/assessments/{assessment_id}/process` | Execute `CASMLEngine` feature extraction, inference, and acoustic profiling |
| | `GET /api/v1/assessments/{assessment_id}/results` | Aggregate scores, risk label, token inconsistency, and utterance table |
| | `GET /api/v1/assessments/{assessment_id}/acoustic-profile` | 3-domain Z-score deviations vs. TD reference baselines |
| | `GET /api/v1/assessments/{assessment_id}/recordings/{recording_id}/audio-analysis` | Downsampled waveform, 64-band log-mel spectrogram, $F_0$ pitch, and eGeMAPS metrics |
| | `GET /api/v1/assessments/{assessment_id}/export` | Download clinical screening dossier as CSV |

---

## Testing & Quality Assurance

### Backend Unit, Contract & Security Suites (50 Tests)

Run from the repository root with the Python virtual environment activated:

```powershell
python -m unittest discover -s backend/tests -v
```

Covers:
- `backend/tests/test_parity_and_endpoints.py` (16 tests): ML feature dimensions, sample prediction parity, endpoint workflows, CSV export, RFC 8259 JSON handling.
- `backend/tests/test_api_contract.py` (12 tests): OpenAPI schema compliance, query validation, multipart upload contracts, and response schemas.
- `backend/tests/test_security_performance.py` (22 tests): POSIX/Windows path traversal defense, upload size/format/duration boundaries, local-only execution, memory stability, and inference latency.

### ML Pipeline Parity Suite (18 Tests)

```powershell
python ml-child-speech-disorder-classification/test_ml_parity.py
```

### Frontend Lint, Type-Check & Production Build

```powershell
cd frontend
npm run lint
npm run build
```

### End-to-End Browser Suite (Playwright — 10 Tests)

```powershell
cd frontend
npm run test:e2e
```

Playwright drives headless Chromium through live patient registration, assessment creation, real speech `.wav` upload, in-process ML execution, radar chart inspection, synchronized waveform/spectrogram zoom, CSV export, and negative audio validation.

---

## Engineering & Audit Documentation

Detailed engineering verification artifacts are available in [`engineering-reports/`](engineering-reports/):

- [`executive-summary.md`](engineering-reports/executive-summary.md) — High-level architecture and multi-track audit outcomes
- [`final-acceptance-report.md`](engineering-reports/final-acceptance-report.md) — Acceptance gate verification checklist
- [`ml-integrity-report.md`](engineering-reports/ml-integrity-report.md) & [`model-parity-report.md`](engineering-reports/model-parity-report.md) — Feature pipeline fidelity and numerical parity
- [`data-lineage-report.md`](engineering-reports/data-lineage-report.md) & [`hardcoded-data-audit.md`](engineering-reports/hardcoded-data-audit.md) — End-to-end data provenance and zero-mock verification
- [`security-audit.md`](engineering-reports/security-audit.md) & [`performance-report.md`](engineering-reports/performance-report.md) — Upload hardening, local processing, and latency profiling
- [`qa-test-report.md`](engineering-reports/qa-test-report.md) & [`e2e-test-report.md`](engineering-reports/e2e-test-report.md) — Automated unit, contract, and Playwright verification results
