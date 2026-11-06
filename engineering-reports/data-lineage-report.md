# Aprixia Data Lineage & Provenance Report

## 1. End-to-End Data Lifecycle

Every data point rendered on the Aprixia UI traces back through a verifiable chain of custody:

```
[1. User Intake]
Clinician enters Child Demographics (Name, MRN, DOB, Sex, Notes)
      │
      ▼ (POST /api/v1/patients)
Persisted in SQLite `patients` table with generated UUID and UTC timestamp.
      │
      ▼ (POST /api/v1/assessments)
Assessment session created in SQLite `assessments` table linked via `patient_id`.
      │
      ▼ (POST /api/v1/assessments/{id}/recordings)
Binary Audio Uploaded -> Saved to `backend/uploads/{assessment_id}/{recording_id}.wav`
Row created in `recordings` table with byte size, MIME type, and librosa-extracted duration.
      │
      ▼ (POST /api/v1/assessments/{id}/process)
Audio bytes read from disk -> Passed to `CASMLEngine.extract_audio_features_and_predict()`
      │
      ├─► openSMILE eGeMAPSv02: Extracts 88 functional features
      ├─► HuBERT Model: Extracts 768-dim Layer 9 mean embeddings
      ├─► Concatenation: 856-dim feature vector aligned to trained column order
      ├─► Scaler: Standardized with `cas_scaler.joblib`
      ├─► Classifier: `cas_classifier.joblib` computes probability: P(CAS) * 100.0
      ├─► Signal Analysis: Waveform downsampling (<= 2000 pts) + 64 Mel-band spectrogram
      └─► Radar Profile: Z-score calculated vs. `td_baselines.joblib`
      │
      ▼ (Database Update)
`recordings` updated with individual risk score, status='completed'
`assessments` updated with mean score, token inconsistency, classification, status='completed'
      │
      ▼ (GET /api/v1/assessments/{id}/results)
React frontend retrieves genuine computed payload and renders clinical summary.
      │
      ▼ (GET /api/v1/assessments/{id}/export)
CSV dossier compiled on-the-fly directly from persisted database and ML metrics.
```

## 2. Integrity Verification Proof
- **Database Schema**: Enforced with SQLite foreign keys (`PRAGMA foreign_keys = ON`). Deleting a patient cascades cleanly to related assessments and recordings.
- **Audio Filesystem Confinement**: All recordings reside within `backend/uploads/` with UUID-based filenames, isolated from directory traversal or system paths.
- **Audit Lineage**: Every assessment result stores the exact model version, timestamp, number of evaluated recordings, and excluded utterance count.
