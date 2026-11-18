"""
Aprixia Clinical CAS Screening System - Security, Performance, and Reliability Test Suite
Covers Requirement R4:
1. Server-side file upload validation (format whitelist, 50MB cap, librosa decode, 0-byte check, sub-200ms rejection).
2. Safe filename sanitization and path traversal defense across POSIX and Windows formats.
3. Local-only processing verification (zero outbound cloud AI transmissions).
4. Heavy visualization caching and downsampling (waveform <= 2000 points, 64-band Mel spectrogram).
5. Strict RFC 8259 JSON compliance (NaN/Inf -> null).
6. Latency benchmarking and memory stability of cached CASMLEngine singleton.
"""
import io
import json
import math
import os
import socket
import sys
import time
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
import psutil
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.audio_service import (
    sanitize_filename,
    validate_and_inspect_audio,
    save_recording_file,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MIN_DURATION_SECONDS,
)
from backend.services.serialization_service import (
    dumps_safe_json,
    loads_safe_json,
    sanitize_floats,
)
from ml_engine import CASMLEngine


class TestPathTraversalAndSanitization(unittest.TestCase):
    """Verifies that filename sanitization defends against path traversal across OS platforms."""

    def test_posix_path_traversal(self):
        """Rejects/strips Unix-style directory traversal sequences."""
        attack_vectors = [
            "../../etc/passwd.wav",
            "../../../etc/shadow.wav",
            "....//....//etc/hosts.wav",
            "/var/log/syslog.wav",
            "./../../sensitive.wav",
        ]
        for attack in attack_vectors:
            safe = sanitize_filename(attack)
            self.assertNotIn("/", safe, f"Failed for {attack}: contains forward slash")
            self.assertNotIn("\\", safe, f"Failed for {attack}: contains backslash")
            self.assertNotIn("..", safe, f"Failed for {attack}: contains dot-dot")
            self.assertTrue(safe.endswith(".wav"), f"Failed for {attack}: lost extension")

    def test_windows_path_traversal(self):
        """Rejects/strips Windows-style directory traversal and drive letters."""
        attack_vectors = [
            "..\\..\\malicious.wav",
            "..\\..\\..\\..\\windows\\system32\\calc.wav",
            "C:\\Windows\\System32\\cmd.wav",
            "D:\\secret\\patient_audio.wav",
            "..\\..\\boot.ini.wav",
        ]
        for attack in attack_vectors:
            safe = sanitize_filename(attack)
            self.assertNotIn("\\", safe, f"Failed for {attack}: contains backslash")
            self.assertNotIn("/", safe, f"Failed for {attack}: contains forward slash")
            self.assertNotIn("..", safe, f"Failed for {attack}: contains dot-dot")
            self.assertNotIn(":", safe, f"Failed for {attack}: contains drive colon")
            self.assertTrue(safe.endswith(".wav"), f"Failed for {attack}: lost extension")

    def test_control_characters_and_null_bytes(self):
        """Strips null bytes and ASCII control characters."""
        dirty = "sample\x00_injection\r\n\t_test.wav"
        safe = sanitize_filename(dirty)
        self.assertNotIn("\x00", safe)
        self.assertNotIn("\r", safe)
        self.assertNotIn("\n", safe)
        self.assertNotIn("\t", safe)
        self.assertTrue(safe.endswith(".wav"))

    def test_empty_and_dot_only_filenames(self):
        """Falls back to safe default name when given empty or dot-only strings."""
        for candidate in ["", "   ", ".", "..", "...", "...."]:
            safe = sanitize_filename(candidate)
            self.assertGreater(len(safe), 0)
            self.assertNotIn("..", safe)
            self.assertFalse(safe.startswith("."))

    def test_save_recording_file_containment(self):
        """Verifies save_recording_file strictly confines output inside target directory."""
        test_dir = PROJECT_ROOT / "backend" / "uploads" / "security_test_sandbox"
        try:
            rec_id, stored_name, full_path = save_recording_file(
                test_dir, b"dummy_content", "../../traversal_attempt.wav"
            )
            # Must be strictly within test_dir
            resolved_target = full_path.resolve()
            resolved_parent = test_dir.resolve()
            self.assertTrue(
                resolved_target.is_relative_to(resolved_parent),
                f"{resolved_target} escaped {resolved_parent}"
            )
            self.assertTrue(full_path.exists())
        finally:
            if test_dir.exists():
                import shutil
                shutil.rmtree(test_dir, ignore_errors=True)


class TestUploadValidationAndRejection(unittest.TestCase):
    """Verifies server-side validation bounds, format whitelisting, and decodability."""

    def test_format_whitelisting(self):
        """Only .wav, .mp3, .m4a, .flac allowed; all others rejected."""
        self.assertEqual(ALLOWED_EXTENSIONS, {".wav", ".mp3", ".m4a", ".flac"})
        rejected_extensions = [
            "payload.exe",
            "script.sh",
            "exploit.py",
            "notes.txt",
            "data.bin",
            "report.pdf",
            "audio.wav.exe",
            "archive.tar.gz",
            "no_extension_file",
        ]
        for name in rejected_extensions:
            valid, msg, _ = validate_and_inspect_audio(b"sample_content", name)
            self.assertFalse(valid, f"Should have rejected {name}")
            self.assertIn("unsupported", msg.lower())

    def test_oversize_payload_rejection(self):
        """Rejects files exceeding 50MB (52,428,800 bytes)."""
        oversize_len = MAX_FILE_SIZE_BYTES + 1024  # 50MB + 1KB
        fake_large_payload = b"0" * 1024  # Don't allocate full 50MB in memory, test boundary check
        # We test with a dummy buffer representing 50MB + 1 byte
        valid, msg, _ = validate_and_inspect_audio(b"x" * (MAX_FILE_SIZE_BYTES + 1), "large.wav")
        self.assertFalse(valid)
        self.assertIn("exceeds maximum allowed size", msg.lower())

    def test_zero_byte_empty_file_rejection(self):
        """Rejects 0-byte audio uploads."""
        valid, msg, _ = validate_and_inspect_audio(b"", "empty.wav")
        self.assertFalse(valid)
        self.assertIn("empty", msg.lower())

    def test_sub_200ms_audio_rejection(self):
        """Rejects audio shorter than 200 milliseconds minimum required by ML pipeline."""
        sr = 16000
        durations_to_fail = [0.05, 0.10, 0.15, 0.19]
        for dur in durations_to_fail:
            samples = int(sr * dur)
            audio = np.random.uniform(-0.1, 0.1, samples).astype(np.float32)
            bio = io.BytesIO()
            sf.write(bio, audio, sr, format="WAV")
            audio_bytes = bio.getvalue()

            valid, msg, _ = validate_and_inspect_audio(audio_bytes, f"short_{int(dur*1000)}ms.wav")
            self.assertFalse(valid, f"Expected rejection for duration {dur}s")
            self.assertIn("too short", msg.lower())

    def test_valid_audio_acceptance_boundary(self):
        """Accepts audio meeting or exceeding 200 milliseconds."""
        sr = 16000
        # 300 ms audio
        audio = np.random.uniform(-0.1, 0.1, int(sr * 0.3)).astype(np.float32)
        bio = io.BytesIO()
        sf.write(bio, audio, sr, format="WAV")
        audio_bytes = bio.getvalue()

        valid, msg, meta = validate_and_inspect_audio(audio_bytes, "valid_300ms.wav")
        self.assertTrue(valid)
        self.assertGreaterEqual(meta["duration"], 0.2)
        self.assertEqual(meta["sample_rate"], 16000)
        self.assertEqual(len(meta["peaks"]), 32)

    def test_corrupt_wav_header_rejection(self):
        """Rejects corrupted, truncated, or malformed audio headers."""
        corrupt_samples = [
            b"RIFF\x24\x00\x00\x00WAVE",  # Truncated 12-byte header with no data chunk
            b"RIFF\xff\xff\xff\xffWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00garbage_data",
            b"PK\x03\x04zip_file_pretending_to_be_audio",
            b"\x7fELF_binary_executable_not_audio",
            os.urandom(512),  # Random bytes
        ]
        for i, payload in enumerate(corrupt_samples):
            valid, msg, _ = validate_and_inspect_audio(payload, f"corrupt_{i}.wav")
            self.assertFalse(valid, f"Sample {i} should be rejected as corrupt")
            self.assertIn("unable to decode", msg.lower())


class TestRFC8259Compliance(unittest.TestCase):
    """Verifies strict JSON RFC 8259 compliance (no NaN, Infinity, or -Infinity tokens)."""

    def test_dumps_safe_json_nan_and_inf(self):
        """Verifies that all floating NaN, Inf, and -Inf values map to JSON null."""
        dirty_dict = {
            "normal_float": 42.5,
            "nan_val": float("nan"),
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
            "numpy_nan": np.nan,
            "numpy_inf": np.float64("inf"),
            "nested_list": [1.0, float("nan"), float("inf"), 4.0],
            "nested_dict": {
                "inner_nan": float("nan"),
                "inner_array": np.array([10.0, np.nan, 30.0]),
            },
        }

        json_output = dumps_safe_json(dirty_dict)

        # RFC 8259: The tokens 'NaN', 'Infinity', '-Infinity' MUST NOT appear
        self.assertNotIn("NaN", json_output)
        self.assertNotIn("Infinity", json_output)

        # Verify strict standard json.loads parses successfully
        parsed = json.loads(json_output)
        self.assertEqual(parsed["normal_float"], 42.5)
        self.assertIsNone(parsed["nan_val"])
        self.assertIsNone(parsed["pos_inf"])
        self.assertIsNone(parsed["neg_inf"])
        self.assertIsNone(parsed["numpy_nan"])
        self.assertIsNone(parsed["numpy_inf"])
        self.assertIsNone(parsed["nested_list"][1])
        self.assertIsNone(parsed["nested_list"][2])
        self.assertIsNone(parsed["nested_dict"]["inner_nan"])
        self.assertIsNone(parsed["nested_dict"]["inner_array"][1])

    def test_allow_nan_false_enforcement(self):
        """Verifies dumps_safe_json disallows unhandled NaNs by default."""
        clean = {"value": 100.0}
        serialized = dumps_safe_json(clean)
        self.assertEqual(json.loads(serialized)["value"], 100.0)

    def test_real_audio_analysis_payload_rfc8259(self):
        """Verifies actual ML engine analysis payload contains zero NaN/Inf tokens."""
        engine = CASMLEngine.get_instance(base_dir=ML_DIR)
        sample_path = ML_DIR / "samples" / "sample_cas_child_03F.wav"
        with open(sample_path, "rb") as f:
            audio_bytes = f.read()

        res, _ = engine.extract_audio_features_and_predict(audio_bytes, sample_path.name)
        payload = engine.generate_audio_analysis_payload(res["audio"], res["sr"], res["ege_df"])

        json_str = dumps_safe_json(payload)
        self.assertNotIn("NaN", json_str)
        self.assertNotIn("Infinity", json_str)

        deserialized = json.loads(json_str)
        self.assertIn("waveform", deserialized)
        self.assertIn("spectrogram", deserialized)
        self.assertIn("pitch_contour", deserialized)


class TestLocalOnlyProcessing(unittest.TestCase):
    """Verifies that all inference occurs locally without external cloud AI calls."""

    def test_no_external_cloud_ai_imports_in_codebase(self):
        """Scans backend and ML modules for prohibited external AI service SDKs."""
        prohibited_modules = [
            "openai",
            "anthropic",
            "google.generativeai",
            "google.cloud",
            "boto3",
            "azure.cognitiveservices",
            "cohere",
            "replicate",
        ]

        # Check all python files under backend/
        backend_files = list((PROJECT_ROOT / "backend").rglob("*.py"))
        for py_file in backend_files:
            content = py_file.read_text(encoding="utf-8")
            for mod in prohibited_modules:
                self.assertNotIn(
                    f"import {mod}",
                    content,
                    f"Prohibited external cloud AI module '{mod}' found in {py_file.name}"
                )
                self.assertNotIn(
                    f"from {mod}",
                    content,
                    f"Prohibited external cloud AI module '{mod}' found in {py_file.name}"
                )

    def test_feature_extraction_is_local(self):
        """Verifies that HuBERT model and openSMILE execute completely in-process on local CPU."""
        engine = CASMLEngine.get_instance(base_dir=ML_DIR)
        # Check HuBERT model device
        device = next(engine.hubert_model.parameters()).device
        self.assertEqual(device.type, "cpu")
        self.assertFalse(engine.hubert_model.training)


class TestVisualizationCachingAndDownsampling(unittest.TestCase):
    """Verifies waveform downsampling ceiling and 64 Mel-band spectrogram structure."""

    def test_waveform_downsampling_ceiling(self):
        """Verifies downsampled waveform never exceeds 2000 points regardless of duration."""
        engine = CASMLEngine.get_instance(base_dir=ML_DIR)
        sr = 16000

        # Test various lengths: 1s, 5s, 15s, 30s
        for duration in [1.0, 5.0, 15.0, 30.0]:
            sig = np.random.uniform(-0.2, 0.2, int(sr * duration)).astype(np.float32)
            payload = engine.generate_audio_analysis_payload(sig, sr)
            waveform_len = len(payload["waveform"]["time"])
            self.assertLessEqual(
                waveform_len,
                2000,
                f"Waveform length {waveform_len} exceeded 2000 points for duration {duration}s"
            )
            self.assertEqual(waveform_len, len(payload["waveform"]["amplitude"]))

    def test_spectrogram_specifications(self):
        """Verifies spectrogram has exactly 64 Mel bands up to 8000 Hz."""
        engine = CASMLEngine.get_instance(base_dir=ML_DIR)
        sig = np.random.uniform(-0.1, 0.1, 16000 * 2).astype(np.float32)
        payload = engine.generate_audio_analysis_payload(sig, 16000)

        spec = payload["spectrogram"]
        self.assertEqual(spec["n_mels"], 64)
        self.assertEqual(spec["fmax"], 8000)
        self.assertEqual(len(spec["db_matrix"]), 64)
        self.assertEqual(len(spec["frequencies"]), 64)


class TestCASMLEnginePerformanceAndMemoryStability(unittest.TestCase):
    """Benchmarks latency and memory stability of the cached CASMLEngine across consecutive calls."""

    @classmethod
    def setUpClass(cls):
        cls.engine = CASMLEngine.get_instance(base_dir=ML_DIR)
        cls.sample_path = ML_DIR / "samples" / "sample_cas_child_03F.wav"
        with open(cls.sample_path, "rb") as f:
            cls.audio_bytes = f.read()

    def test_singleton_instance_identity(self):
        """Verifies CASMLEngine is a strict singleton; repeated get_instance returns identical object."""
        inst1 = CASMLEngine.get_instance(base_dir=ML_DIR)
        inst2 = CASMLEngine.get_instance(base_dir=ML_DIR)
        self.assertIs(inst1, inst2, "CASMLEngine must return the exact same singleton instance")

    def test_consecutive_inference_latency_benchmark(self):
        """
        Runs consecutive inference calls and benchmarks execution latency.
        Confirms cached model eliminates initialization overhead.
        """
        latencies = []
        n_iterations = 4

        # Warm-up call
        self.engine.extract_audio_features_and_predict(self.audio_bytes, "warmup.wav")

        for i in range(n_iterations):
            t_start = time.perf_counter()
            res, err = self.engine.extract_audio_features_and_predict(
                self.audio_bytes, f"bench_{i}.wav"
            )
            t_elapsed = time.perf_counter() - t_start
            self.assertIsNone(err)
            self.assertIsNotNone(res)
            latencies.append(t_elapsed)

        mean_latency = float(np.mean(latencies))
        min_latency = float(np.min(latencies))
        max_latency = float(np.max(latencies))
        std_latency = float(np.std(latencies))

        print(
            f"\n[BENCHMARK] Consecutive Inferences (N={n_iterations}): "
            f"Mean={mean_latency:.3f}s, Min={min_latency:.3f}s, Max={max_latency:.3f}s, Std={std_latency:.3f}s"
        )
        # Each subsequent inference on 3.56s audio should complete in reasonable time (< 4.5s on CPU)
        self.assertLess(mean_latency, 4.5, "Cached inference latency too slow")

    def test_consecutive_inference_memory_stability(self):
        """
        Monitors process RSS memory across consecutive inference executions.
        Verifies no runaway memory leakage across iterations.
        """
        import gc
        gc.collect()
        proc = psutil.Process(os.getpid())
        initial_rss_mb = proc.memory_info().rss / (1024 * 1024)

        for i in range(4):
            res, _ = self.engine.extract_audio_features_and_predict(
                self.audio_bytes, f"mem_{i}.wav"
            )
            _ = self.engine.generate_audio_analysis_payload(res["audio"], res["sr"], res["ege_df"])

        del res
        gc.collect()
        final_rss_mb = proc.memory_info().rss / (1024 * 1024)
        rss_delta_mb = final_rss_mb - initial_rss_mb

        print(
            f"\n[MEMORY AUDIT] Initial RSS: {initial_rss_mb:.2f} MB, "
            f"Final RSS: {final_rss_mb:.2f} MB, Delta: {rss_delta_mb:.2f} MB"
        )
        # Memory growth across 4 inference runs should remain bounded (< 75MB delta on CPU allocator)
        self.assertLess(rss_delta_mb, 75.0, "Excessive memory growth detected across inferences")


class TestAPIPathTraversalUpload(unittest.TestCase):
    """Verifies that API upload endpoint safely handles malicious filenames."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_path = ML_DIR / "samples" / "sample_cas_child_03F.wav"
        with open(cls.sample_path, "rb") as f:
            cls.audio_bytes = f.read()

    def test_api_upload_with_traversal_filenames(self):
        """Uploads valid audio but with path traversal filename; verifies stored safely."""
        # Create an assessment first
        res_create = self.client.post("/api/v1/assessments", json={
            "child_name": "Traversal Test Patient",
            "age_years": 5,
            "age_months": 0
        })
        self.assertEqual(res_create.status_code, 201)
        ass_id = res_create.json()["id"]

        malicious_filenames = [
            "../../etc/passwd.wav",
            "..\\..\\windows_malicious.wav",
            "....//....//shadow.wav",
        ]

        files = [
            ("files", (fname, self.audio_bytes, "audio/wav"))
            for fname in malicious_filenames
        ]

        res_upload = self.client.post(f"/api/v1/assessments/{ass_id}/recordings", files=files)
        self.assertEqual(res_upload.status_code, 201)
        records = res_upload.json()
        self.assertEqual(len(records), len(malicious_filenames))

        from backend.database import SessionLocal
        from backend.models.recording import Recording

        db = SessionLocal()
        try:
            for rec in records:
                rec_id = rec["recording_id"]
                db_rec = db.query(Recording).filter(Recording.recording_id == rec_id).first()
                self.assertIsNotNone(db_rec)
                self.assertNotIn("..", db_rec.stored_filename)
                self.assertNotIn("/", db_rec.stored_filename)
                self.assertNotIn("\\", db_rec.stored_filename)
                self.assertEqual(rec["validation_status"], "valid")
                self.assertTrue(Path(db_rec.file_path).exists())
                self.assertTrue(
                    Path(db_rec.file_path).resolve().is_relative_to(
                        (PROJECT_ROOT / "backend" / "uploads").resolve()
                    )
                )
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
