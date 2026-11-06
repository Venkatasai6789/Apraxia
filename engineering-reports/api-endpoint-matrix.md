# Aprixia API Endpoint Matrix

| # | Endpoint | Method | Implemented | Request Schema | Response Schema | Status Code | Tested & Verified |
|---|---|---|:---:|---|---|:---:|:---:|
| 1 | `/api/v1/health` | GET | Yes | None | `{"status": "ok", "app": "Aprixia Clinical CAS API"}` | 200 | Yes (`test_01_health`) |
| 2 | `/api/v1/patients` | POST | Yes | `PatientCreate` (`name`, `dob`, `sex`, `mrn`, `notes`) | `PatientResponse` | 201 | Yes (`test_02_create_patient`) |
| 3 | `/api/v1/patients` | GET | Yes | None | `List[PatientResponse]` | 200 | Yes (`test_03_list_patients`) |
| 4 | `/api/v1/patients/{patient_id}` | GET | Yes | Path param (`patient_id`) | `PatientResponse` | 200 | Yes (`test_04_get_patient_by_id`) |
| 5 | `/api/v1/patients/{patient_id}` | PATCH | Yes | `PatientUpdate` | `PatientResponse` | 200 | Yes (Unit test verified) |
| 6 | `/api/v1/assessments` | POST | Yes | `AssessmentCreate` (`patient_id`, `type`, `notes`) | `AssessmentResponse` | 201 | Yes (`test_05_create_assessment`) |
| 7 | `/api/v1/assessments` | GET | Yes | Query params (`patient_id`, `status`) | `List[AssessmentResponse]` | 200 | Yes (`test_06_list_assessments`) |
| 8 | `/api/v1/assessments/{assessment_id}` | GET | Yes | Path param (`assessment_id`) | `AssessmentDetailResponse` | 200 | Yes (`test_07_get_assessment_by_id`) |
| 9 | `/api/v1/assessments/{assessment_id}` | PATCH | Yes | `AssessmentUpdate` | `AssessmentResponse` | 200 | Yes (Unit test verified) |
| 10 | `/api/v1/assessments/{assessment_id}/recordings` | POST | Yes | `multipart/form-data` (`file: UploadFile`) | `RecordingResponse` | 201 | Yes (`test_08_upload_recording_valid`) |
| 11 | `/api/v1/assessments/{assessment_id}/process` | POST | Yes | Path param (`assessment_id`) | `AssessmentStatusResponse` | 200 / 202 | Yes (`test_11_process_assessment`) |
| 12 | `/api/v1/assessments/{assessment_id}/status` | GET | Yes | Path param (`assessment_id`) | `AssessmentStatusResponse` | 200 | Yes (`test_12_get_assessment_status`) |
| 13 | `/api/v1/assessments/{assessment_id}/results` | GET | Yes | Path param (`assessment_id`) | `AssessmentResultResponse` | 200 | Yes (`test_13_get_assessment_results`) |
| 14 | `/api/v1/assessments/{assessment_id}/acoustic-profile` | GET | Yes | Path param (`assessment_id`) | `AcousticProfileResponse` | 200 | Yes (`test_14_get_acoustic_profile`) |
| 15 | `/api/v1/assessments/{assessment_id}/recordings/{recording_id}` | GET | Yes | Path params (`assessment_id`, `recording_id`) | `RecordingResponse` | 200 | Yes (Unit test verified) |
| 16 | `/api/v1/assessments/{assessment_id}/recordings/{recording_id}/audio-analysis` | GET | Yes | Path params (`assessment_id`, `recording_id`) | `AudioAnalysisResponse` | 200 | Yes (`test_15_get_audio_analysis`) |
| 17 | `/api/v1/assessments/{assessment_id}/export` | GET | Yes | Path param (`assessment_id`) | `text/csv` attachment | 200 | Yes (`test_16_export_assessment_csv`) |
| 18 | `/api/v1/dashboard/stats` | GET | Yes | None | `DashboardStatsResponse` | 200 | Yes (E2E & integration tested) |
| 19 | `/api/v1/dashboard/overview` | GET | Yes | None | `DashboardOverviewResponse` | 200 | Yes (E2E & integration tested) |
