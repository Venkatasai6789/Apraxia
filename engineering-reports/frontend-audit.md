# Aprixia Frontend Clinical UI/UX & Architecture Audit

## 1. Frontend Architecture & Technology Stack
- **Framework**: React 19 with TypeScript.
- **Build Tool**: Vite 8.3 (production bundle generated in 4.23s).
- **Styling**: Tailwind CSS with clinical design system tokens (restrained slate/indigo/purple palette, high clinical readability).
- **Visualizations**: Plotly.js (`react-plotly.js`) for multi-subplot synchronized acoustics; Recharts / Lucide icons for clinical telemetry.

## 2. Screen-by-Screen Clinical Review

### 1. Dashboard (`/`)
- Replaced hardcoded patient cards and mock charts with live calls to `/api/v1/dashboard/stats` and `/api/v1/assessments`.
- Displays real clinical intake metrics: Total Patients, Total Screenings, Elevated Risk Flags, and Recent Screening Activity.
- Displays truthful empty states when no patients or assessments are registered yet.

### 2. Patients (`/patients`)
- Full clinical patient list loaded from SQLite backend via `GET /api/v1/patients`.
- Patient registration modal persists real demographic data (`name`, `mrn`, `dob`, `sex`, `notes`).
- Direct action button to initiate an assessment for any registered patient.

### 3. Assessments Archive (`/assessments`)
- Table view displaying all recorded assessments with status badges (`draft`, `processing`, `completed`, `failed`).
- Filters by risk tier and date.
- Real CSV dossier export triggered directly from table rows.

### 4. New Assessment (`/assessments/new`)
- Multi-step structured workflow:
  1. Patient selection from live database.
  2. Clinical intake notes & assessment protocol selection.
  3. Audio upload workspace supporting drag-and-drop of multiple speech files.
  4. Real-time file validation (rejection of invalid formats or $< 200$ ms files).
  5. Clinician review & sign-off prior to triggering inference.

### 5. Processing State
- Real-time polling of `GET /api/v1/assessments/{id}/status`.
- Progress tracking reflecting genuine backend inference stages.

### 6. Assessment Results (`/assessments/{id}/results`)
- Prominent clinical risk banner (Score %, Classification, Severity badge).
- Three-domain acoustic radar chart comparing child speech vs. Typically Developing (TD) normative reference.
- Utterance-level evaluation table detailing individual recording duration, risk score, and quality indicators.
- Direct link to launch the Audio Analysis Laboratory.

### 7. Audio Analysis Laboratory (`/assessments/{id}/recordings/{rec_id}/audio-lab`)
- Three stacked, synchronized subplots:
  1. Amplitude Waveform (downsampled envelope, actual time in seconds).
  2. 64-Band Log-Mel Spectrogram (0–8 kHz, dB intensity colormap).
  3. F0 Pitch Contour (fundamental frequency track in Hz).
- Interactive Zoom Synchronization: Selecting a time range on any subplot synchronizes the x-axis range across all three subplots. "Reset Zoom" button restores full duration.

## 3. Linting and Build Verification
- **Linter**: `oxlint` executed across 79 files with 116 rules: **0 warnings, 0 errors** (99 ms).
- **TypeScript**: `tsc -b` compiled with **0 type errors**.
- **Vite Build**: Built production bundle in 4.23s (`dist/index.html`, `dist/assets/`).
