# Aprixia API Mapping & Contract Integration Report

## 1. Frontend-to-Backend Service Mapping

Every API call in the React application is audited for exact 1:1 route alignment with the FastAPI backend:

| User Flow / Action | Frontend Service Call | HTTP Endpoint | Backend Route Handler | Schema Status |
|---|---|---|---|:---:|
| System Health Ping | `healthApi.getHealth()` | `GET /api/v1/health` | `routes/health.py:health_check` | **COMPATIBLE** |
| Register New Patient | `patientApi.createPatient(data)` | `POST /api/v1/patients` | `routes/patients.py:create_patient` | **COMPATIBLE** |
| Fetch All Patients | `patientApi.getPatients()` | `GET /api/v1/patients` | `routes/patients.py:list_patients` | **COMPATIBLE** |
| Fetch Single Patient | `patientApi.getPatientById(id)` | `GET /api/v1/patients/{id}` | `routes/patients.py:get_patient` | **COMPATIBLE** |
| Update Patient Details | `patientApi.updatePatient(id, data)` | `PATCH /api/v1/patients/{id}` | `routes/patients.py:update_patient` | **COMPATIBLE** |
| Create Assessment | `assessmentApi.createAssessment(data)` | `POST /api/v1/assessments` | `routes/assessments.py:create_assessment` | **COMPATIBLE** |
| List Assessments | `assessmentApi.getAssessments()` | `GET /api/v1/assessments` | `routes/assessments.py:list_assessments` | **COMPATIBLE** |
| Get Assessment Details | `assessmentApi.getAssessmentById(id)` | `GET /api/v1/assessments/{id}` | `routes/assessments.py:get_assessment` | **COMPATIBLE** |
| Upload Speech Audio | `assessmentApi.uploadRecording(id, file)` | `POST /api/v1/assessments/{id}/recordings` | `routes/recordings.py:upload_recording` | **COMPATIBLE** |
| Launch ML Inference | `assessmentApi.processAssessment(id)` | `POST /api/v1/assessments/{id}/process` | `routes/assessments.py:process_assessment` | **COMPATIBLE** |
| Poll Processing State | `assessmentApi.getStatus(id)` | `GET /api/v1/assessments/{id}/status` | `routes/assessments.py:get_status` | **COMPATIBLE** |
| Retrieve Clinical Results | `assessmentApi.getResults(id)` | `GET /api/v1/assessments/{id}/results` | `routes/assessments.py:get_results` | **COMPATIBLE** |
| Fetch Acoustic Radar | `assessmentApi.getAcousticProfile(id)` | `GET /api/v1/assessments/{id}/acoustic-profile` | `routes/assessments.py:get_acoustic_profile` | **COMPATIBLE** |
| Fetch Audio Analysis | `audioApi.getAudioAnalysis(aid, rid)` | `GET /api/v1/assessments/{aid}/recordings/{rid}/audio-analysis` | `routes/audio_analysis.py:get_audio_analysis` | **COMPATIBLE** |
| Export Assessment CSV | `assessmentApi.getExportCsvUrl(id)` | `GET /api/v1/assessments/{id}/export` | `routes/export.py:export_assessment_csv` | **COMPATIBLE** |
| Dashboard Intake Stats | `dashboardApi.getStats()` | `GET /api/v1/dashboard/stats` | `routes/dashboard.py:get_dashboard_stats` | **COMPATIBLE** |
| Dashboard Overview | `dashboardApi.getOverview()` | `GET /api/v1/dashboard/overview` | `routes/dashboard.py:get_dashboard_overview` | **COMPATIBLE** |

## 2. Orphaned Route & Call Analysis
- **Orphaned Frontend Calls**: 0. Every call made in React maps to an active backend route.
- **Orphaned Backend Endpoints**: 0. All defined endpoints serve active frontend workflows or testing verification.
- **Contract Test Results**: 12/12 contract tests passing (`backend/tests/test_api_contract.py`).

**Auditor Verdict: APPROVE**.
