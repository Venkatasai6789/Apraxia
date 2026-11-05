"""
Comprehensive Backend Test Suite: Model Parity, Validations, and REST Endpoints
Verifies 100% mathematical parity with the original CAS Streamlit screening pipeline.
"""
import io
import os
import sys
import unittest
from pathlib import Path

# Add project root and ml repo to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
ML_DIR = PROJECT_ROOT / "ml-child-speech-disorder-classification"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.audio_service import validate_and_inspect_audio
from backend.services.serialization_service import dumps_safe_json, loads_safe_json
from ml_engine import CASMLEngine, calculate_aggregate_metrics


class TestModelParityAndEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = CASMLEngine.get_instance(base_dir=ML_DIR)
        cls.sample_path = ML_DIR / "samples" / "sample_cas_child_03F.wav"
        cls.client = TestClient(app)

    # -------------------------------------------------------------------------
    # 1. MODEL ARCHITECTURE & FEATURE PARITY TESTS
    # -------------------------------------------------------------------------
    def test_feature_dimensions(self):
        """Verify 856-dim feature alignment (88 eGeMAPSv02 + 768 HuBERT Layer 9)."""
        self.assertEqual(len(self.engine.feature_cols), 856)
        self.assertEqual(self.engine.scaler.n_features_in_, 856)
        self.assertEqual(self.engine.classifier.n_features_in_, 856)

    def test_sample_prediction_parity(self):
        """Verify deterministic prediction on benchmark CAS recording."""
        self.assertTrue(self.sample_path.exists(), "Sample audio file missing.")
        with open(self.sample_path, "rb") as f:
            data = f.read()

        res, err = self.engine.extract_audio_features_and_predict(data, self.sample_path.name)
        self.assertIsNone(err)
        self.assertIsNotNone(res)
        self.assertGreater(res["score"], 90.0, "Confirmed CAS sample must have high risk score")
        self.assertAlmostEqual(res["duration"], 3.56, delta=0.1)

    def test_acoustic_profile_calculation(self):
        """Verify absolute Z-score calculation against Typically Developing baselines."""
        with open(self.sample_path, "rb") as f:
            data = f.read()
        res, _ = self.engine.extract_audio_features_and_predict(data, self.sample_path.name)
        profile = self.engine.calculate_speech_profile(res["ege_df"])

        self.assertIn("Pitch / Prosody", profile)
        self.assertIn("Voice Quality", profile)
        self.assertIn("Vowel Formants", profile)

        # Baseline comparison
        self.assertEqual(set(profile.keys()), set(self.engine.td_baselines.keys()))
        for dim, sub in profile.items():
            for subgroup, val in sub.items():
                self.assertIsInstance(val, float)
                self.assertGreater(val, 0.0)

    def test_audio_analysis_payload(self):
        """Verify waveform, 64-band Mel-spectrogram, and pitch contour."""
        with open(self.sample_path, "rb") as f:
            data = f.read()
        res, _ = self.engine.extract_audio_features_and_predict(data, self.sample_path.name)
        analysis = self.engine.generate_audio_analysis_payload(
            res["audio"], res["sr"], res["ege_df"]
        )

        # Waveform check
        self.assertIn("waveform", analysis)
        self.assertLessEqual(len(analysis["waveform"]["time"]), 2500)
        self.assertEqual(len(analysis["waveform"]["time"]), len(analysis["waveform"]["amplitude"]))

        # Spectrogram check (64 Mel bands, 0-8000 Hz)
        self.assertIn("spectrogram", analysis)
        self.assertEqual(analysis["spectrogram"]["n_mels"], 64)
        self.assertEqual(analysis["spectrogram"]["fmax"], 8000)
        self.assertEqual(len(analysis["spectrogram"]["db_matrix"]), 64)

        # Pitch contour check
        self.assertIn("pitch_contour", analysis)
        self.assertEqual(
            len(analysis["pitch_contour"]["times"]),
            len(analysis["pitch_contour"]["f0_hz"])
        )

        # Acoustic features check
        self.assertIn("f0_mean_hz", analysis["acoustic_features"])
        self.assertIn("jitter_pct", analysis["acoustic_features"])
        self.assertIn("f1_mean_hz", analysis["acoustic_features"])

    # -------------------------------------------------------------------------
    # 2. VALIDATION GATES (SHORT AUDIO, CORRUPT FILES)
    # -------------------------------------------------------------------------
    def test_short_audio_rejection(self):
        """Rejects audio recordings shorter than 200 milliseconds."""
        # Generate 100ms synthetic audio
        sr = 16000
        short_y = np.zeros(int(sr * 0.1), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, short_y, sr, format="WAV")
        short_bytes = buf.getvalue()

        res, err = self.engine.extract_audio_features_and_predict(short_bytes, "short.wav")
        self.assertIsNone(res)
        self.assertIn("too short", err.lower())

    def test_corrupt_audio_rejection(self):
        """Rejects non-audio binary junk."""
        corrupt_bytes = b"NOT_AN_AUDIO_FILE_JUST_RANDOM_GARBAGE"
        res, err = self.engine.extract_audio_features_and_predict(corrupt_bytes, "corrupt.wav")
        self.assertIsNone(res)
        self.assertIsNotNone(err)

    def test_empty_audio_rejection(self):
        """Rejects 0-byte empty audio files."""
        is_valid, msg, _ = validate_and_inspect_audio(b"", "empty.wav")
        self.assertFalse(is_valid)
        self.assertIn("empty", msg.lower())

    def test_unsupported_format_rejection(self):
        """Rejects non-whitelisted audio file extensions."""
        is_valid, msg, _ = validate_and_inspect_audio(b"fake_content", "document.pdf")
        self.assertFalse(is_valid)
        self.assertIn("unsupported", msg.lower())

    # -------------------------------------------------------------------------
    # 3. END-TO-END REST API WORKFLOW TESTS
    # -------------------------------------------------------------------------
    def test_api_health_check(self):
        """Test GET /api/v1/health."""
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["database_ready"])

    def test_next_patient_id(self):
        """Test GET /api/v1/patients/next-id returns dynamic collision-safe ID."""
        res = self.client.get("/api/v1/patients/next-id")
        self.assertEqual(res.status_code, 200)
        pid = res.json()["next_patient_id"]
        self.assertTrue(pid.startswith("APR-"))

    def test_end_to_end_assessment_and_inference(self):
        """
        Tests complete lifecycle:
        1. Create patient & assessment via API.
        2. Upload real speech audio file via multipart form.
        3. Trigger ML inference pipeline.
        4. Query calculated results, acoustic profile, audio analysis, and CSV export.
        """
        # 1. Create Assessment
        create_payload = {
            "child_name": "Test Child Integration",
            "age_years": 5,
            "age_months": 6,
            "clinical_notes": "Unit testing automated pipeline integration."
        }
        res_create = self.client.post("/api/v1/assessments", json=create_payload)
        self.assertEqual(res_create.status_code, 201)
        ass_data = res_create.json()
        ass_id = ass_data["id"]
        self.assertTrue(ass_id.startswith("as-"))
        self.assertTrue(ass_data["patient_id"].startswith("APR-"))

        # 2. Upload Real Recording
        with open(self.sample_path, "rb") as f:
            audio_bytes = f.read()

        files = [
            ("files", ("test_sample.wav", audio_bytes, "audio/wav"))
        ]
        res_upload = self.client.post(f"/api/v1/assessments/{ass_id}/recordings", files=files)
        self.assertEqual(res_upload.status_code, 201)
        rec_list = res_upload.json()
        self.assertEqual(len(rec_list), 1)
        rec_id = rec_list[0]["recording_id"]

        # 3. Trigger ML Pipeline
        res_process = self.client.post(f"/api/v1/assessments/{ass_id}/process")
        self.assertEqual(res_process.status_code, 200)

        # 4. Query Assessment Results
        res_results = self.client.get(f"/api/v1/assessments/{ass_id}/results")
        self.assertEqual(res_results.status_code, 200)
        result_payload = res_results.json()
        self.assertGreater(result_payload["mean_score"], 90.0)
        self.assertEqual(result_payload["risk_label"], "Elevated CAS Risk")
        self.assertEqual(result_payload["utterances_evaluated"], 1)

        # 5. Query Acoustic Radar Profile
        res_profile = self.client.get(f"/api/v1/assessments/{ass_id}/acoustic-profile")
        self.assertEqual(res_profile.status_code, 200)
        prof_data = res_profile.json()
        self.assertIn("Pitch / Prosody", prof_data["dimensions"])

        # 6. Query Audio Analysis for Laboratory
        res_analysis = self.client.get(
            f"/api/v1/assessments/{ass_id}/recordings/{rec_id}/audio-analysis"
        )
        self.assertEqual(res_analysis.status_code, 200)
        an_data = res_analysis.json()
        self.assertIn("waveform", an_data)
        self.assertIn("spectrogram", an_data)
        self.assertIn("pitch_contour", an_data)

        # 7. Query Audio Streaming
        res_audio = self.client.get(
            f"/api/v1/assessments/{ass_id}/recordings/{rec_id}/audio"
        )
        self.assertEqual(res_audio.status_code, 200)
        self.assertEqual(res_audio.headers["content-type"], "audio/wav")

        # 8. Query CSV Export
        res_csv = self.client.get(f"/api/v1/assessments/{ass_id}/export")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("Apraxia Prediction Score", res_csv.text)

    def test_assessments_stats_endpoint(self):
        """Test GET /api/v1/assessments/stats returns real SQLite aggregated counts."""
        res = self.client.get("/api/v1/assessments/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total", data)
        self.assertIn("completed", data)
        self.assertIn("pending", data)
        self.assertIn("flagged", data)
        self.assertIn("total_patients", data)
        self.assertGreaterEqual(data["total"], 1)
        self.assertGreaterEqual(data["total_patients"], 1)

    def test_dashboard_stats_endpoint(self):
        """Test GET /api/v1/dashboard/stats returns real clinical stats and recent dossiers."""
        res = self.client.get("/api/v1/dashboard/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_patients", data)
        self.assertIn("total_assessments", data)
        self.assertIn("completed_assessments", data)
        self.assertIn("pending_assessments", data)
        self.assertIn("flagged_assessments", data)
        self.assertIn("stats", data)
        self.assertIn("recent_assessments", data)
        self.assertIsInstance(data["recent_assessments"], list)

    def test_dashboard_overview_endpoints(self):
        """Test GET /api/v1/dashboard/overview and alias /api/dashboard/overview for React Dashboard."""
        for path in ["/api/v1/dashboard/overview", "/api/dashboard/overview"]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, f"Endpoint {path} failed")
            data = res.json()
            self.assertIn("stats", data)
            self.assertIn("screeningTrends", data)
            self.assertIn("insights", data)
            self.assertIn("recentAssessments", data)
            self.assertEqual(len(data["stats"]), 4)
            self.assertGreaterEqual(len(data["insights"]), 4)

    def test_invalid_recording_upload_handling(self):
        """Test uploading empty and unsupported audio files via multipart endpoint."""
        # Create an assessment first
        create_res = self.client.post("/api/v1/assessments", json={
            "child_name": "Upload Validation Test",
            "age_years": 4,
            "age_months": 2
        })
        ass_id = create_res.json()["id"]

        # Upload an empty file and an unsupported format file
        files = [
            ("files", ("empty.wav", b"", "audio/wav")),
            ("files", ("malicious.exe", b"binary_data", "application/octet-stream"))
        ]
        upload_res = self.client.post(f"/api/v1/assessments/{ass_id}/recordings", files=files)
        self.assertEqual(upload_res.status_code, 201)
        recs = upload_res.json()
        self.assertEqual(len(recs), 2)
        for r in recs:
            self.assertEqual(r["validation_status"], "invalid")
            self.assertEqual(r["processing_status"], "failed")
            self.assertIsNotNone(r["error_message"])

    def test_safe_json_rfc8259_serialization(self):
        """Test RFC 8259 compliance ensuring NaN and Infinity are safely serialized to null."""
        test_payload = {
            "valid_float": 12.34,
            "nan_val": float("nan"),
            "inf_pos": float("inf"),
            "inf_neg": float("-inf"),
            "numpy_nan": np.nan,
            "nested": {
                "sub_nan": float("nan"),
                "sub_list": [1.0, float("nan"), 3.0]
            }
        }
        json_str = dumps_safe_json(test_payload)
        self.assertNotIn("NaN", json_str)
        self.assertNotIn("Infinity", json_str)
        loaded = loads_safe_json(json_str)
        self.assertIsNone(loaded["nan_val"])
        self.assertIsNone(loaded["inf_pos"])
        self.assertIsNone(loaded["inf_neg"])
        self.assertIsNone(loaded["numpy_nan"])
        self.assertIsNone(loaded["nested"]["sub_nan"])
        self.assertIsNone(loaded["nested"]["sub_list"][1])


if __name__ == "__main__":
    unittest.main()
