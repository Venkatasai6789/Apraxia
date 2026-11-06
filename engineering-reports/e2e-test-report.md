# Aprixia Playwright Browser Automation & End-to-End Test Report

## 1. Playwright Test Suite Overview
The end-to-end browser test suite (`frontend/tests/clinical_browser_e2e.spec.ts`) automates a clinician user session inside a real headless Chromium browser ($1440 \times 900$ viewport). It drives the complete application through live HTTP requests against Vite (`:5173`) and FastAPI (`:8000`).

## 2. Test Execution Details & Results

```
Running 10 tests using 1 worker

  ✓ Test 1: Navigation & Live Database States (Zero Mock Data) (9.5s)
  ✓ Test 2: Patient Registration in Clinical Intake Workspace (3.0s)
  ✓ Test 3: New Assessment Dossier Configuration (255ms)
  ✓ Test 4: Real Speech Audio Upload and Client Validation (1.0s)
  ✓ Test 5: Clinician Sign-Off & In-Process ML Screening Execution (14.0s)
  ✓ Test 6: Verify Clinical Predictions, Score, and Acoustic Radar Domains (334ms)
  ✓ Test 7: Audio Laboratory Acoustic Inspection (Waveform, Spectrogram, Pitch) (719ms)
  ✓ Test 8: Waveform and Spectrogram Time-Synchronized Zoom and Reset (2.2s)
  ✓ Test 9: Clinical Assessment CSV Export & Data Integrity (3.0s)
  ✓ Test 10: Negative Audio Validation (Corrupt / Empty Audio Graceful Handling) (3.1s)

  10 passed (1.4m)
```

## 3. Step-by-Step Scenario Analysis

### Test 1: Navigation & Live Database States
Navigated to Dashboard, Patients, and Assessments. Verified that zero mock patient cards or fake charts rendered. Verified genuine live empty/populated states.

### Test 2: Patient Intake Registration
Filled in clinical intake form: Child Name (`E2E Test Subject`), MRN (`E2E-MRN-9092`), DOB (`2019-04-12`), Sex (`male`), Referral Notes (`Clinical E2E Test`). Verified patient record appeared immediately in the patient directory.

### Test 3: New Assessment Creation
Selected newly created patient. Configured assessment protocol and notes. Verified redirect to the recording intake stage.

### Test 4: Audio Upload & Client Validation
Uploaded real clinical audio sample (`sample_cas_child_03F.wav`). Verified audio duration was correctly extracted ($3.56$s) and visual confirmation badge displayed.

### Test 5: In-Process Machine Learning Execution
Clinician signed off and launched assessment. Real-time polling observed `queued` -> `processing` -> `completed`. Verified openSMILE + HuBERT Layer 9 feature extraction finished without timeout.

### Test 6: Clinical Results & Acoustic Radar Inspection
Verified clinical outcome displayed:
- Risk Percentage: **99.9996%** (Elevated CAS Risk).
- Severity: High.
- Radar Chart: Plotted 3 clinical domains (Pitch / Prosody, Voice Quality, Formants) with normative TD overlay.
- Utterance Evaluation Table: Confirmed recording filename, duration, and score match backend.

### Test 7: Audio Laboratory Visualization
Opened the Audio Analysis Laboratory for the recording. Verified that:
- Waveform rendered real amplitude samples.
- 64-band Log-Mel Spectrogram rendered real dB frequency energy.
- F0 Pitch contour rendered actual fundamental frequency curve.

### Test 8: Time-Synchronized Zoom & Reset
Executed drag-zoom over a 1.0-second time interval ($1.0\text{s} - 2.0\text{s}$) on the waveform. Verified that the spectrogram and pitch subplots automatically zoomed to the exact same interval. Clicked "Reset Zoom" and confirmed all three plots returned to full duration ($0.0\text{s} - 3.56\text{s}$).

### Test 9: CSV Dossier Download
Triggered CSV export. Verified file download with non-empty payload containing genuine patient MRN, assessment ID, overall risk score, and recording breakdowns.

### Test 10: Negative Audio Validation
Attempted upload of an invalid/empty file. Verified that the UI gracefully displayed a clinical validation error message without crashing.
