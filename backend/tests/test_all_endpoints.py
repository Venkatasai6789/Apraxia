"""
Comprehensive All-Endpoints Integration Test Suite
Exercises all 31 registered API endpoints across all routers in Aprixia:
- Health Router
- Patients Router
- Assessments Router
- Recordings Router
- Results & Analysis Router
- Dashboard Router
- Device Proxy Router

Ensures zero HTTP errors, strict schema compliance, and end-to-end clinical workflow execution.
"""
import io
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.device_service import DeviceService

ML_DIR = Path(__file__).parent.parent.parent / "ml-child-speech-disorder-classification"


class TestAllAPIEndpoints(unittest.TestCase):
    test_assessment_id = None
    test_recording_id = None
    test_attached_rec_id = None

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_path = ML_DIR / "samples" / "sample_cas_child_03F.wav"
        if cls.sample_path.exists():
            with open(cls.sample_path, "rb") as f:
                cls.valid_wav_bytes = f.read()
        else:
            # Fallback 1-second synthetic 16kHz 16-bit mono PCM wav header + data
            import wave
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 16000)
            cls.valid_wav_bytes = buf.getvalue()

    # =========================================================================
    # 1. HEALTH ENDPOINT
    # =========================================================================
    def test_01_health_check(self):
        """GET /api/v1/health -> healthy status & model metadata"""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        self.assertIn("model_ready", data)
        self.assertTrue(data["database_ready"])

    # =========================================================================
    # 2. PATIENTS ENDPOINTS
    # =========================================================================
    def test_02_patients_next_id(self):
        """GET /api/v1/patients/next-id -> dynamic collision-safe ID"""
        response = self.client.get("/api/v1/patients/next-id")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("next_patient_id", data)
        self.assertTrue(data["next_patient_id"].startswith("APR-"))

    def test_03_create_patient(self):
        """POST /api/v1/patients -> creates patient record"""
        payload = {
            "child_name": "Integration Test Child",
            "age_years": 5,
            "age_months": 6,
            "patient_id": "APR-TEST-01"
        }
        response = self.client.post("/api/v1/patients", json=payload)
        self.assertIn(response.status_code, [200, 201])
        data = response.json()
        self.assertEqual(data["patient_id"], "APR-TEST-01")
        self.assertEqual(data["child_name"], "Integration Test Child")

    def test_04_list_patients(self):
        """GET /api/v1/patients -> list of patients with search and limit"""
        response = self.client.get("/api/v1/patients?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)

        # Search filter
        search_res = self.client.get("/api/v1/patients?search=Integration")
        self.assertEqual(search_res.status_code, 200)
        s_data = search_res.json()
        self.assertGreaterEqual(len(s_data), 1)

    def test_05_get_patient_by_id(self):
        """GET /api/v1/patients/{patient_id} -> patient details or 404"""
        response = self.client.get("/api/v1/patients/APR-TEST-01")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["patient_id"], "APR-TEST-01")

        # 404 test
        not_found = self.client.get("/api/v1/patients/NON-EXISTENT-ID")
        self.assertEqual(not_found.status_code, 404)

    # =========================================================================
    # 3. ASSESSMENTS ENDPOINTS (LIFECYCLE)
    # =========================================================================
    def test_06_create_assessment(self):
        """POST /api/v1/assessments -> creates assessment dossier"""
        payload = {
            "child_name": "E2E Assessment Patient",
            "age_years": 6,
            "age_months": 2,
            "patient_id": "APR-E2E-001",
            "assessment_reference": "AS-E2E-999",
            "clinical_notes": "Automated end-to-end integration testing dossier."
        }
        response = self.client.post("/api/v1/assessments", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["patient_id"], "APR-E2E-001")
        self.assertEqual(data["child_name"], "E2E Assessment Patient")
        self.assertEqual(data["status"], "draft")

        # Save assessment ID for downstream steps
        TestAllAPIEndpoints.test_assessment_id = data["id"]

    def test_07_list_assessments_and_stats(self):
        """GET /api/v1/assessments & GET /api/v1/assessments/stats"""
        res_list = self.client.get("/api/v1/assessments?page=1&page_size=10")
        self.assertEqual(res_list.status_code, 200)
        list_data = res_list.json()
        self.assertIn("records", list_data)
        self.assertIn("total_matching", list_data)
        self.assertIn("stats", list_data)

        # Aggregated stats
        res_stats = self.client.get("/api/v1/assessments/stats")
        self.assertEqual(res_stats.status_code, 200)
        stats_data = res_stats.json()
        self.assertIn("total", stats_data)
        self.assertIn("completed", stats_data)
        self.assertIn("pending", stats_data)

    def test_08_get_assessment_details_and_status(self):
        """GET /api/v1/assessments/{id} & GET /api/v1/assessments/{id}/status"""
        aid = TestAllAPIEndpoints.test_assessment_id
        self.assertIsNotNone(aid)

        res_detail = self.client.get(f"/api/v1/assessments/{aid}")
        self.assertEqual(res_detail.status_code, 200)
        d_data = res_detail.json()
        self.assertEqual(d_data["id"], aid)

        res_status = self.client.get(f"/api/v1/assessments/{aid}/status")
        self.assertEqual(res_status.status_code, 200)
        s_data = res_status.json()
        self.assertEqual(s_data["assessment_id"], aid)

    # =========================================================================
    # 4. RECORDINGS ENDPOINTS (UPLOAD, STREAM, METADATA)
    # =========================================================================
    def test_09_upload_recording_multipart(self):
        """POST /api/v1/assessments/{id}/recordings -> multipart audio upload"""
        aid = TestAllAPIEndpoints.test_assessment_id
        self.assertIsNotNone(aid)
        files = [
            ("files", ("test_utterance_01.wav", self.valid_wav_bytes, "audio/wav"))
        ]
        response = self.client.post(f"/api/v1/assessments/{aid}/recordings", files=files)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        rec = data[0]
        self.assertEqual(rec["original_filename"], "test_utterance_01.wav")
        self.assertEqual(rec["validation_status"], "valid")

        # Save recording ID
        TestAllAPIEndpoints.test_recording_id = rec["recording_id"]

    def test_10_get_recording_and_stream_audio(self):
        """GET /api/v1/assessments/{id}/recordings/{rid} & .../audio"""
        aid = TestAllAPIEndpoints.test_assessment_id
        rid = TestAllAPIEndpoints.test_recording_id
        self.assertIsNotNone(aid)
        self.assertIsNotNone(rid)

        # 1. Recording metadata
        res_rec = self.client.get(f"/api/v1/assessments/{aid}/recordings/{rid}")
        self.assertEqual(res_rec.status_code, 200)
        r_data = res_rec.json()
        self.assertEqual(r_data["recording_id"], rid)
        self.assertEqual(r_data["assessment_id"], aid)

        # 2. Audio streaming
        res_audio = self.client.get(f"/api/v1/assessments/{aid}/recordings/{rid}/audio")
        self.assertEqual(res_audio.status_code, 200)
        self.assertIn("audio/", res_audio.headers.get("content-type", ""))
        self.assertGreater(len(res_audio.content), 0)

    # =========================================================================
    # 5. ML INFERENCE, RESULTS, ACOUSTIC PROFILE & AUDIO ANALYSIS
    # =========================================================================
    def test_11_ml_process_assessment(self):
        """POST /api/v1/assessments/{id}/process -> runs full 856-dim ML pipeline"""
        aid = TestAllAPIEndpoints.test_assessment_id
        self.assertIsNotNone(aid)
        response = self.client.post(f"/api/v1/assessments/{aid}/process")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("data", data)
        self.assertIn("risk_label", data["data"])
        self.assertIn("mean_score", data["data"])

    def test_12_get_results_and_acoustic_profile(self):
        """GET /api/v1/assessments/{id}/results & .../acoustic-profile"""
        aid = TestAllAPIEndpoints.test_assessment_id
        self.assertIsNotNone(aid)

        # Results
        res_results = self.client.get(f"/api/v1/assessments/{aid}/results")
        self.assertEqual(res_results.status_code, 200)
        r_data = res_results.json()
        self.assertIn("mean_score", r_data)
        self.assertIn("risk_label", r_data)
        self.assertIn("utterance_results", r_data)
        self.assertGreaterEqual(len(r_data["utterance_results"]), 1)

        # Acoustic Profile
        res_profile = self.client.get(f"/api/v1/assessments/{aid}/acoustic-profile")
        self.assertEqual(res_profile.status_code, 200)
        p_data = res_profile.json()
        self.assertIn("dimensions", p_data)
        self.assertIn("td_baselines", p_data)

    def test_13_get_recording_audio_analysis_laboratory(self):
        """GET /api/v1/assessments/{id}/recordings/{rid}/audio-analysis"""
        aid = TestAllAPIEndpoints.test_assessment_id
        rid = TestAllAPIEndpoints.test_recording_id
        self.assertIsNotNone(aid)
        self.assertIsNotNone(rid)

        response = self.client.get(f"/api/v1/assessments/{aid}/recordings/{rid}/audio-analysis")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["recording_id"], rid)
        self.assertIn("waveform", data)
        self.assertIn("spectrogram", data)
        self.assertIn("pitch_contour", data)
        self.assertIn("acoustic_features", data)
        self.assertLessEqual(len(data["waveform"]["time"]), 2000)
        self.assertEqual(data["spectrogram"]["n_mels"], 64)
        self.assertEqual(len(data["spectrogram"]["frequencies"]), 64)
        self.assertEqual(len(data["spectrogram"]["db_matrix"]), 64)

    def test_14_export_assessment_csv(self):
        """GET /api/v1/assessments/{id}/export -> downloads clinical CSV report"""
        aid = TestAllAPIEndpoints.test_assessment_id
        self.assertIsNotNone(aid)
        response = self.client.get(f"/api/v1/assessments/{aid}/export")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers.get("content-type", ""))
        self.assertIn(f"attachment; filename=\"Aprixia_Screening_{aid}.csv\"", response.headers.get("content-disposition", ""))
        self.assertIn("Audio File", response.text)

    # =========================================================================
    # 6. DASHBOARD ENDPOINTS
    # =========================================================================
    def test_15_dashboard_stats_and_overview(self):
        """GET /api/v1/dashboard/stats & /api/v1/dashboard/overview & /api/dashboard/overview"""
        # Stats
        res_stats = self.client.get("/api/v1/dashboard/stats")
        self.assertEqual(res_stats.status_code, 200)
        s_data = res_stats.json()
        self.assertIn("stats", s_data)
        self.assertIn("recent_assessments", s_data)

        # Overview v1
        res_ov1 = self.client.get("/api/v1/dashboard/overview")
        self.assertEqual(res_ov1.status_code, 200)
        ov1 = res_ov1.json()
        self.assertIn("stats", ov1)
        self.assertIn("recentAssessments", ov1)

        # Overview alias
        res_ov2 = self.client.get("/api/dashboard/overview")
        self.assertEqual(res_ov2.status_code, 200)
        ov2 = res_ov2.json()
        self.assertIn("stats", ov2)
        self.assertIn("recentAssessments", ov2)

    # =========================================================================
    # 7. ESP32 DEVICE PROXY ENDPOINTS (ALL 8 HARDWARE ROUTES)
    # =========================================================================
    def test_16_device_status_online_and_offline(self):
        """GET /api/v1/device/status -> online telemetry and offline fallback"""
        with patch.object(DeviceService, "get_status", new_callable=AsyncMock) as mock_status:
            # Online mock
            mock_status.return_value = {
                "connected": True,
                "recording": False,
                "current_file": None,
                "total_bytes": 1441792,
                "used_bytes": 286720,
                "free_bytes": 1155072,
                "file_count": 2,
                "max_record_seconds": 30,
                "sample_rate": 16000,
                "uptime_ms": 125000,
                "wifi_clients": 1,
                "error": None
            }
            res_online = self.client.get("/api/v1/device/status")
            self.assertEqual(res_online.status_code, 200)
            self.assertTrue(res_online.json()["connected"])
            self.assertEqual(res_online.json()["sample_rate"], 16000)

            # Offline mock
            mock_status.return_value = {
                "connected": False,
                "recording": False,
                "current_file": None,
                "total_bytes": 0,
                "used_bytes": 0,
                "free_bytes": 0,
                "file_count": 0,
                "max_record_seconds": 30,
                "sample_rate": 16000,
                "uptime_ms": 0,
                "wifi_clients": 0,
                "error": "ESP32 unreachable at http://192.168.4.1"
            }
            res_offline = self.client.get("/api/v1/device/status")
            self.assertEqual(res_offline.status_code, 200)
            self.assertFalse(res_offline.json()["connected"])

    def test_17_device_list_recordings(self):
        """GET /api/v1/device/recordings -> returns acoustic assets stored on flash"""
        with patch.object(DeviceService, "list_recordings", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [
                {
                    "name": "REC0001.wav",
                    "size": 160044,
                    "duration_sec": 5.0
                },
                {
                    "name": "REC0002.wav",
                    "size": 96044,
                    "duration_sec": 3.0
                }
            ]
            response = self.client.get("/api/v1/device/recordings")
            self.assertEqual(response.status_code, 200)
            items = response.json()
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["name"], "REC0001.wav")

    def test_18_device_download_recording(self):
        """GET /api/v1/device/recordings/{filename}/download -> streams WAV file"""
        with patch.object(DeviceService, "download_recording", new_callable=AsyncMock) as mock_download:
            mock_download.return_value = self.valid_wav_bytes
            response = self.client.get("/api/v1/device/recordings/REC0001.wav/download")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "audio/wav")
            self.assertIn("attachment; filename=\"REC0001.wav\"", response.headers["content-disposition"])
            self.assertEqual(response.content, self.valid_wav_bytes)

            # 404 test
            mock_download.return_value = None
            res_404 = self.client.get("/api/v1/device/recordings/NONEXIST.wav/download")
            self.assertEqual(res_404.status_code, 404)

    def test_19_device_rename_recording(self):
        """POST /api/v1/device/recordings/{filename}/rename -> renames file on LittleFS"""
        with patch.object(DeviceService, "rename_recording", new_callable=AsyncMock) as mock_rename:
            mock_rename.return_value = {
                "ok": True,
                "message": "File renamed successfully",
                "file": "REC0001_word_pataka.wav"
            }
            payload = {"new_name": "REC0001_word_pataka.wav"}
            response = self.client.post("/api/v1/device/recordings/REC0001.wav/rename", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ok"])

    def test_20_device_delete_recording(self):
        """DELETE /api/v1/device/recordings/{filename} -> deletes file from LittleFS"""
        with patch.object(DeviceService, "delete_recording", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = {
                "ok": True,
                "message": "File REC0002.wav deleted successfully"
            }
            response = self.client.delete("/api/v1/device/recordings/REC0002.wav")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ok"])

    def test_21_device_record_start_and_stop(self):
        """POST /api/v1/device/record/start & /api/v1/device/record/stop"""
        with patch.object(DeviceService, "start_recording", new_callable=AsyncMock) as mock_start:
            mock_start.return_value = {"ok": True, "message": "Recording initiated"}
            res_start = self.client.post("/api/v1/device/record/start")
            self.assertEqual(res_start.status_code, 200)
            self.assertTrue(res_start.json()["ok"])

        with patch.object(DeviceService, "stop_recording", new_callable=AsyncMock) as mock_stop:
            mock_stop.return_value = {"ok": True, "message": "Recording stopped and saved"}
            res_stop = self.client.post("/api/v1/device/record/stop")
            self.assertEqual(res_stop.status_code, 200)
            self.assertTrue(res_stop.json()["ok"])

    def test_22_device_attach_recording_to_assessment(self):
        """POST /api/v1/device/recordings/{filename}/attach -> downloads from ESP32 & ingests"""
        aid = TestAllAPIEndpoints.test_assessment_id
        self.assertIsNotNone(aid)
        with patch.object(DeviceService, "download_recording", new_callable=AsyncMock) as mock_download:
            mock_download.return_value = self.valid_wav_bytes
            payload = {
                "assessment_id": aid,
                "custom_filename": "ESP32_Maya_Pataka_Test.wav"
            }
            response = self.client.post("/api/v1/device/recordings/REC0003.wav/attach", json=payload)
            self.assertEqual(response.status_code, 201)
            data = response.json()
            self.assertEqual(data["assessment_id"], aid)
            self.assertEqual(data["original_filename"], "ESP32_Maya_Pataka_Test.wav")
            self.assertEqual(data["validation_status"], "valid")

            # Save attached recording ID for cleanup
            TestAllAPIEndpoints.test_attached_rec_id = data["recording_id"]

    # =========================================================================
    # 8. CLEANUP & DELETION ENDPOINTS
    # =========================================================================
    def test_23_delete_recording_from_assessment(self):
        """DELETE /api/v1/assessments/{id}/recordings/{rid}"""
        aid = TestAllAPIEndpoints.test_assessment_id
        rid = TestAllAPIEndpoints.test_attached_rec_id
        if aid and rid:
            response = self.client.delete(f"/api/v1/assessments/{aid}/recordings/{rid}")
            self.assertEqual(response.status_code, 200)
            self.assertIn("removed", response.json()["message"])

    def test_24_delete_assessment(self):
        """DELETE /api/v1/assessments/{id} -> cascades and deletes assessment"""
        aid = TestAllAPIEndpoints.test_assessment_id
        if aid:
            response = self.client.delete(f"/api/v1/assessments/{aid}")
            self.assertEqual(response.status_code, 200)
            self.assertIn("deleted successfully", response.json()["message"])

            # Verify subsequent 404
            check_404 = self.client.get(f"/api/v1/assessments/{aid}")
            self.assertEqual(check_404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
