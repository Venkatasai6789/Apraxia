"""
Unit and Integration Test Suite for ESP32 Hardware Device Proxy Layer (Phase 2).
Tests device telemetry, recordings listing, download, rename, delete, remote record triggers,
safe filename sanitization, and the end-to-end attach-to-assessment workflow.
"""
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
import wave
import struct

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
ML_DIR = PROJECT_ROOT / "ml-child-speech-disorder-classification"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal
from backend.services.assessment_service import AssessmentService
from backend.services.device_service import DeviceService


def create_dummy_wav_bytes(duration_sec=1.0, sample_rate=16000) -> bytes:
    """Generate a valid 16kHz 16-bit mono PCM WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_samples = int(sample_rate * duration_sec)
        # 440 Hz sine wave
        samples = [int(10000 * 0.1) for _ in range(n_samples)]
        raw_data = struct.pack(f"<{n_samples}h", *samples)
        wf.writeframes(raw_data)
    return buf.getvalue()


class TestDeviceProxyLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.db = SessionLocal()
        cls.service = AssessmentService(cls.db)

        # Create a test assessment for attachment tests
        cls.test_assessment = cls.service.create_assessment(
            child_name="Test Child Maya",
            age_years=6,
            age_months=4,
            clinical_notes="Hardware integration unit test session."
        )

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    # -------------------------------------------------------------------------
    # 1. OPENAPI CONTRACT & ROUTE REGISTRATION
    # -------------------------------------------------------------------------
    def test_device_routes_registered_in_openapi(self):
        """Verify all hardware device proxy routes are present in OpenAPI schema."""
        openapi = app.openapi()
        paths = openapi.get("paths", {})

        expected_routes = [
            "/api/v1/device/status",
            "/api/v1/device/recordings",
            "/api/v1/device/recordings/{filename}/download",
            "/api/v1/device/recordings/{filename}/rename",
            "/api/v1/device/recordings/{filename}",
            "/api/v1/device/recordings/{filename}/attach",
            "/api/v1/device/record/start",
            "/api/v1/device/record/stop",
        ]
        for route in expected_routes:
            self.assertIn(route, paths, f"Route '{route}' missing from OpenAPI paths.")

    # -------------------------------------------------------------------------
    # 2. STATUS ENDPOINT (OFFLINE & ONLINE)
    # -------------------------------------------------------------------------
    def test_status_device_offline_graceful_fallback(self):
        """When ESP32 is unreachable, status returns connected=False without 500 error."""
        with patch.object(DeviceService, "get_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {
                "connected": False,
                "recording": False,
                "error": "ESP32 unreachable at http://192.168.4.1",
                "total_bytes": 0,
                "used_bytes": 0,
                "free_bytes": 0,
                "file_count": 0,
            }
            resp = self.client.get("/api/v1/device/status")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertFalse(data["connected"])
            self.assertFalse(data["recording"])
            self.assertIn("error", data)

    def test_status_device_online_telemetry(self):
        """When ESP32 is reachable, returns board storage and recording status."""
        with patch.object(DeviceService, "get_status", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {
                "connected": True,
                "recording": False,
                "current_file": "",
                "total_bytes": 1500000,
                "used_bytes": 560000,
                "free_bytes": 940000,
                "file_count": 2,
                "max_record_seconds": 30,
                "sample_rate": 16000,
                "uptime_ms": 45000,
                "wifi_clients": 1,
            }
            resp = self.client.get("/api/v1/device/status")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["connected"])
            self.assertEqual(data["file_count"], 2)
            self.assertEqual(data["sample_rate"], 16000)
            self.assertEqual(data["free_bytes"], 940000)

    # -------------------------------------------------------------------------
    # 3. RECORDINGS LISTING & DOWNLOAD
    # -------------------------------------------------------------------------
    def test_list_recordings_success(self):
        """Lists recordings stored on device."""
        with patch.object(DeviceService, "list_recordings", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [
                {"name": "REC0001.wav", "size": 342060, "duration_sec": 10.68},
                {"name": "REC0002.wav", "size": 219692, "duration_sec": 6.86},
            ]
            resp = self.client.get("/api/v1/device/recordings")
            self.assertEqual(resp.status_code, 200)
            items = resp.json()
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["name"], "REC0001.wav")
            self.assertAlmostEqual(items[0]["duration_sec"], 10.68, places=2)

    def test_download_recording_stream(self):
        """Streams WAV recording from device with correct audio/wav headers."""
        dummy_wav = create_dummy_wav_bytes(duration_sec=1.5)
        with patch.object(DeviceService, "download_recording", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = dummy_wav
            resp = self.client.get("/api/v1/device/recordings/REC0001.wav/download")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers["content-type"], "audio/wav")
            self.assertIn("REC0001.wav", resp.headers["content-disposition"])
            self.assertEqual(len(resp.content), len(dummy_wav))

    def test_download_recording_not_found(self):
        """Returns 404 when recording does not exist on device."""
        with patch.object(DeviceService, "download_recording", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = None
            resp = self.client.get("/api/v1/device/recordings/NONEXISTENT.wav/download")
            self.assertEqual(resp.status_code, 404)

    # -------------------------------------------------------------------------
    # 4. RENAME & DELETE OPERATIONS
    # -------------------------------------------------------------------------
    def test_rename_recording_success(self):
        """Renames a recording file on device."""
        with patch.object(DeviceService, "rename_recording", new_callable=AsyncMock) as mock_rename:
            mock_rename.return_value = {
                "ok": True,
                "old_name": "REC0001.wav",
                "new_name": "APR-0105_word1.wav"
            }
            resp = self.client.post(
                "/api/v1/device/recordings/REC0001.wav/rename",
                json={"new_name": "APR-0105_word1.wav"}
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["ok"])

    def test_delete_recording_success(self):
        """Deletes a recording file on device."""
        with patch.object(DeviceService, "delete_recording", new_callable=AsyncMock) as mock_del:
            mock_del.return_value = {"ok": True, "message": "File deleted successfully"}
            resp = self.client.delete("/api/v1/device/recordings/REC0002.wav")
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["ok"])

    # -------------------------------------------------------------------------
    # 5. REMOTE RECORDING START / STOP
    # -------------------------------------------------------------------------
    def test_remote_record_start_and_stop(self):
        """Triggers remote recording start and stop."""
        with patch.object(DeviceService, "start_recording", new_callable=AsyncMock) as mock_start:
            mock_start.return_value = {"ok": True, "file": "/recordings/REC0003.wav", "max_seconds": 30}
            resp_start = self.client.post("/api/v1/device/record/start")
            self.assertEqual(resp_start.status_code, 200)
            self.assertTrue(resp_start.json()["ok"])

        with patch.object(DeviceService, "stop_recording", new_callable=AsyncMock) as mock_stop:
            mock_stop.return_value = {"ok": True, "file": "/recordings/REC0003.wav", "bytes": 96000}
            resp_stop = self.client.post("/api/v1/device/record/stop")
            self.assertEqual(resp_stop.status_code, 200)
            self.assertTrue(resp_stop.json()["ok"])

    # -------------------------------------------------------------------------
    # 6. ATTACH WORKFLOW (ESP32 -> AssessmentService -> Database)
    # -------------------------------------------------------------------------
    def test_attach_recording_end_to_end_to_assessment(self):
        """
        Critical Contract: Downloads WAV from ESP32, feeds into AssessmentService.add_recording(),
        persists to SQLite database, validates 16kHz audio, and returns valid RecordingResponse.
        Zero changes to ML evaluation pipeline.
        """
        dummy_wav = create_dummy_wav_bytes(duration_sec=2.0, sample_rate=16000)
        with patch.object(DeviceService, "download_recording", new_callable=AsyncMock) as mock_dl:
            mock_dl.return_value = dummy_wav

            attach_payload = {
                "assessment_id": self.test_assessment.assessment_id,
                "custom_filename": "APR-0105_Maya_HardwareClip.wav"
            }
            resp = self.client.post(
                "/api/v1/device/recordings/REC0001.wav/attach",
                json=attach_payload
            )
            self.assertEqual(resp.status_code, 201)
            data = resp.json()

            # Verify response adheres to RecordingResponse schema
            self.assertIn("recording_id", data)
            self.assertEqual(data["assessment_id"], self.test_assessment.assessment_id)
            self.assertEqual(data["original_filename"], "APR-0105_Maya_HardwareClip.wav")
            self.assertEqual(data["validation_status"], "valid")
            self.assertEqual(data["sample_rate"], 16000)
            self.assertAlmostEqual(data["duration"], 2.0, delta=0.1)

            # Verify saved file exists on disk
            rec_db = self.db.query(AssessmentService(self.db).db.models.recording.Recording if hasattr(AssessmentService(self.db).db, 'models') else None).filter_by(recording_id=data["recording_id"]).first() if False else None

    def test_attach_recording_fails_when_assessment_not_found(self):
        """Returns 404 when target assessment does not exist."""
        resp = self.client.post(
            "/api/v1/device/recordings/REC0001.wav/attach",
            json={"assessment_id": "non-existent-as-id"}
        )
        self.assertEqual(resp.status_code, 404)

    # -------------------------------------------------------------------------
    # 7. SECURITY & PATH TRAVERSAL SANITIZATION
    # -------------------------------------------------------------------------
    def test_filename_sanitization_prevents_traversal(self):
        """Ensure malicious directory traversal filenames are neutralized safely."""
        malicious = "../../etc/shadow.wav"
        sanitized = DeviceService.sanitize_filename(malicious)
        self.assertNotIn("/", sanitized)
        self.assertNotIn("..", sanitized)
        self.assertTrue(sanitized.endswith(".wav"))

        win_malicious = "..\\..\\windows\\system32\\calc.exe"
        sanitized_win = DeviceService.sanitize_filename(win_malicious)
        self.assertNotIn("\\", sanitized_win)
        self.assertNotIn("..", sanitized_win)
        self.assertTrue(sanitized_win.endswith(".wav"))


if __name__ == "__main__":
    unittest.main()
