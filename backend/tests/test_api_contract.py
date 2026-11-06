"""
Automated API Contract Verification Test Suite
Validates 100% contract fidelity between FastAPI backend and React frontend.
Audits OpenAPI schema, HTTP methods, path/query parameters, multipart uploads,
and request/response payload schemas.
"""
import io
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
ML_DIR = PROJECT_ROOT / "ml-child-speech-disorder-classification"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from fastapi.testclient import TestClient
from backend.main import app


class TestOpenAPISchemaContract(unittest.TestCase):
    """Verifies that the generated OpenAPI 3.x schema matches all frontend interface definitions."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.openapi = app.openapi()
        cls.paths = cls.openapi.get("paths", {})
        cls.schemas = cls.openapi.get("components", {}).get("schemas", {})

    def test_openapi_schema_loaded(self):
        """Ensure OpenAPI schema generated successfully."""
        self.assertIsNotNone(self.openapi)
        self.assertIn("paths", self.openapi)
        self.assertGreater(len(self.paths), 10)

    def test_all_frontend_endpoints_exist_in_openapi(self):
        """Verify zero orphaned frontend endpoints: every frontend path exists in OpenAPI."""
        frontend_endpoints = [
            ("GET", "/api/v1/patients/next-id"),
            ("GET", "/api/v1/patients"),
            ("POST", "/api/v1/assessments"),
            ("GET", "/api/v1/assessments"),
            ("GET", "/api/v1/assessments/{assessment_id}"),
            ("DELETE", "/api/v1/assessments/{assessment_id}"),
            ("POST", "/api/v1/assessments/{assessment_id}/recordings"),
            ("GET", "/api/v1/assessments/{assessment_id}/recordings/{recording_id}/audio"),
            ("POST", "/api/v1/assessments/{assessment_id}/process"),
            ("GET", "/api/v1/assessments/{assessment_id}/results"),
            ("GET", "/api/v1/assessments/{assessment_id}/recordings/{recording_id}/audio-analysis"),
            ("GET", "/api/v1/assessments/{assessment_id}/export"),
        ]

        for method, path in frontend_endpoints:
            self.assertIn(
                path,
                self.paths,
                f"Missing route in backend OpenAPI schema: {path}"
            )
            route_methods = [m.upper() for m in self.paths[path].keys()]
            self.assertIn(
                method,
                route_methods,
                f"HTTP method {method} not supported on {path}. Available: {route_methods}"
            )

    def test_assessments_query_parameters_schema(self):
        """Verify query parameters on GET /api/v1/assessments match frontend expectations."""
        route_info = self.paths.get("/api/v1/assessments", {}).get("get", {})
        parameters = route_info.get("parameters", [])
        param_names = {p["name"]: p for p in parameters if p.get("in") == "query"}

        # Frontend sends: status, search, page, page_size
        self.assertIn("status", param_names, "Missing 'status' query parameter")
        self.assertIn("search", param_names, "Missing 'search' query parameter")
        self.assertIn("page", param_names, "Missing 'page' query parameter")
        self.assertIn("page_size", param_names, "Missing 'page_size' query parameter")

        # Verify defaults and constraints
        self.assertFalse(param_names["status"].get("required", False))
        self.assertFalse(param_names["search"].get("required", False))
        self.assertFalse(param_names["page"].get("required", False))
        self.assertFalse(param_names["page_size"].get("required", False))

    def test_multipart_upload_schema(self):
        """Verify multipart upload contract on POST /api/v1/assessments/{assessment_id}/recordings."""
        route_info = self.paths.get("/api/v1/assessments/{assessment_id}/recordings", {}).get("post", {})
        request_body = route_info.get("requestBody", {})
        content = request_body.get("content", {})
        self.assertIn(
            "multipart/form-data",
            content,
            "Upload route must accept multipart/form-data content type"
        )

        form_schema = content["multipart/form-data"].get("schema", {})
        if "$ref" in form_schema:
            ref_name = form_schema["$ref"].split("/")[-1]
            form_schema = self.schemas.get(ref_name, {})

        properties = form_schema.get("properties", {})
        self.assertIn(
            "files",
            properties,
            "Multipart form data must have form field name 'files'"
        )

    def test_pydantic_schema_definitions(self):
        """Verify key Pydantic schemas exist in OpenAPI components."""
        expected_schemas = [
            "AssessmentCreate",
            "AssessmentDetailResponse",
            "AssessmentListResponse",
            "AssessmentResultsResponse",
            "AudioAnalysisResponse",
            "RecordingResponse",
            "PatientResponse",
        ]
        for s in expected_schemas:
            self.assertIn(s, self.schemas, f"Pydantic schema '{s}' missing from OpenAPI components")


class TestRuntimeAPIContractExecution(unittest.TestCase):
    """Executes live requests using TestClient to verify request/response payloads against TypeScript contracts."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_path = ML_DIR / "samples" / "sample_cas_child_03F.wav"

    def test_contract_next_patient_id(self):
        """Contract: GET /api/v1/patients/next-id -> { next_patient_id: string }"""
        res = self.client.get("/api/v1/patients/next-id")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, dict)
        self.assertIn("next_patient_id", data)
        self.assertIsInstance(data["next_patient_id"], str)
        self.assertTrue(data["next_patient_id"].startswith("APR-"))

    def test_contract_patients_list(self):
        """Contract: GET /api/v1/patients -> PatientItem[]"""
        res = self.client.get("/api/v1/patients")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, list)
        if len(data) > 0:
            item = data[0]
            self.assertIn("id", item)
            self.assertIn("patient_id", item)
            self.assertIn("age_years", item)
            self.assertIn("age_months", item)
            self.assertIn("created_at", item)
            self.assertIsInstance(item["id"], int)
            self.assertIsInstance(item["patient_id"], str)
            self.assertIsInstance(item["age_years"], int)
            self.assertIsInstance(item["age_months"], int)

    def test_contract_assessment_create_and_lifecycle(self):
        """
        Contract:
        1. POST /api/v1/assessments with AssessmentCreatePayload -> AssessmentDetailResponse
        2. GET /api/v1/assessments?page=1&page_size=10 -> AssessmentListResponse
        3. POST /api/v1/assessments/{id}/recordings with multipart files -> RecordingResponse[]
        4. POST /api/v1/assessments/{id}/process -> { message, data }
        5. GET /api/v1/assessments/{id}/results -> AssessmentResultsResponse
        6. GET /api/v1/assessments/{id}/recordings/{rec_id}/audio-analysis -> AudioAnalysisResponse
        7. GET /api/v1/assessments/{id}/recordings/{rec_id}/audio -> audio/wav stream
        8. GET /api/v1/assessments/{id}/export -> text/csv attachment
        9. DELETE /api/v1/assessments/{id} -> 200 OK
        """
        # Step 1: Create Assessment
        create_payload = {
            "child_name": "Contract Test Child",
            "age_years": 6,
            "age_months": 4,
            "clinical_notes": "Contract test verification audit record."
        }
        res_create = self.client.post("/api/v1/assessments", json=create_payload)
        self.assertEqual(res_create.status_code, 201)
        created = res_create.json()

        # Verify AssessmentDetailResponse contract fields
        expected_detail_fields = [
            "id", "patient_id", "child_name", "age_years", "age_months",
            "assessment_reference", "assessment_date", "recording_count",
            "status", "clinical_notes", "has_report", "recordings"
        ]
        for f in expected_detail_fields:
            self.assertIn(f, created, f"Field '{f}' missing from AssessmentDetailResponse")

        ass_id = created["id"]
        self.assertTrue(ass_id.startswith("as-"))
        self.assertEqual(created["age_years"], 6)
        self.assertEqual(created["age_months"], 4)

        # Step 2: List Assessments (Pagination & Stats Contract)
        res_list = self.client.get("/api/v1/assessments", params={"page": 1, "page_size": 10})
        self.assertEqual(res_list.status_code, 200)
        list_data = res_list.json()

        self.assertIn("records", list_data)
        self.assertIn("total_matching", list_data)
        self.assertIn("page", list_data)
        self.assertIn("total_pages", list_data)
        self.assertIn("stats", list_data)
        self.assertIsInstance(list_data["records"], list)
        self.assertIsInstance(list_data["total_matching"], int)
        self.assertIsInstance(list_data["page"], int)
        self.assertIsInstance(list_data["total_pages"], int)

        # Stats contract: total, completed, pending, flagged
        stats = list_data["stats"]
        self.assertIn("total", stats)
        self.assertIn("completed", stats)
        self.assertIn("pending", stats)
        self.assertIn("flagged", stats)

        # Step 3: Multipart Audio Upload
        with open(self.sample_path, "rb") as f:
            audio_bytes = f.read()

        upload_files = [
            ("files", ("sample_cas_child_03F.wav", audio_bytes, "audio/wav"))
        ]
        res_upload = self.client.post(f"/api/v1/assessments/{ass_id}/recordings", files=upload_files)
        self.assertEqual(res_upload.status_code, 201)
        recs = res_upload.json()
        self.assertIsInstance(recs, list)
        self.assertEqual(len(recs), 1)

        rec = recs[0]
        expected_rec_fields = [
            "recording_id", "assessment_id", "original_filename",
            "duration", "file_size", "sample_rate", "channels",
            "validation_status", "processing_status", "created_at"
        ]
        for f in expected_rec_fields:
            self.assertIn(f, rec, f"Field '{f}' missing from RecordingResponse")
        self.assertEqual(rec["validation_status"], "valid")
        rec_id = rec["recording_id"]

        # Step 4: ML Process Trigger
        res_proc = self.client.post(f"/api/v1/assessments/{ass_id}/process")
        self.assertEqual(res_proc.status_code, 200)
        proc_data = res_proc.json()
        self.assertIn("message", proc_data)
        self.assertIn("data", proc_data)

        # Step 5: Query Assessment Results
        res_results = self.client.get(f"/api/v1/assessments/{ass_id}/results")
        self.assertEqual(res_results.status_code, 200)
        results = res_results.json()

        expected_results_fields = [
            "assessment_id", "assessment_reference", "patient_id",
            "child_name", "age_formatted", "assessment_date", "status",
            "mean_score", "median_score", "min_score", "max_score",
            "token_inconsistency", "risk_label", "action_text",
            "utterances_evaluated", "acoustic_profile", "td_baselines",
            "utterance_results"
        ]
        for f in expected_results_fields:
            self.assertIn(f, results, f"Field '{f}' missing from AssessmentResultsResponse")

        self.assertIsInstance(results["mean_score"], (int, float))
        self.assertIsInstance(results["risk_label"], str)
        self.assertIsInstance(results["acoustic_profile"], dict)
        self.assertIsInstance(results["td_baselines"], dict)
        self.assertIsInstance(results["utterance_results"], list)
        self.assertGreater(len(results["utterance_results"]), 0)

        # Step 6: Audio Analysis Laboratory Payload
        res_analysis = self.client.get(
            f"/api/v1/assessments/{ass_id}/recordings/{rec_id}/audio-analysis"
        )
        self.assertEqual(res_analysis.status_code, 200)
        analysis = res_analysis.json()

        expected_analysis_fields = [
            "recording_id", "assessment_id", "original_filename",
            "duration", "sample_rate", "waveform", "spectrogram",
            "pitch_contour", "acoustic_features"
        ]
        for f in expected_analysis_fields:
            self.assertIn(f, analysis, f"Field '{f}' missing from AudioAnalysisResponse")

        # Waveform check
        wf = analysis["waveform"]
        self.assertIn("time", wf)
        self.assertIn("amplitude", wf)
        self.assertIn("total_samples", wf)
        self.assertEqual(len(wf["time"]), len(wf["amplitude"]))

        # Spectrogram check (64 Mel-bands)
        spec = analysis["spectrogram"]
        self.assertIn("times", spec)
        self.assertIn("frequencies", spec)
        self.assertIn("db_matrix", spec)
        self.assertEqual(spec.get("n_mels", 64), 64)
        self.assertEqual(len(spec["db_matrix"]), 64)

        # Pitch contour check
        pc = analysis["pitch_contour"]
        self.assertIn("times", pc)
        self.assertIn("f0_hz", pc)
        self.assertEqual(len(pc["times"]), len(pc["f0_hz"]))

        # Step 7: Audio Stream Endpoint
        res_audio = self.client.get(
            f"/api/v1/assessments/{ass_id}/recordings/{rec_id}/audio"
        )
        self.assertEqual(res_audio.status_code, 200)
        self.assertEqual(res_audio.headers.get("content-type"), "audio/wav")

        # Step 8: CSV Export Endpoint
        res_csv = self.client.get(f"/api/v1/assessments/{ass_id}/export")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("text/csv", res_csv.headers.get("content-type", ""))
        self.assertIn("attachment", res_csv.headers.get("content-disposition", ""))
        self.assertIn("Apraxia Prediction Score", res_csv.text)

        # Step 9: Delete Assessment
        res_del = self.client.delete(f"/api/v1/assessments/{ass_id}")
        self.assertEqual(res_del.status_code, 200)

        # Verify 404 after deletion
        res_check = self.client.get(f"/api/v1/assessments/{ass_id}")
        self.assertEqual(res_check.status_code, 404)


class TestAdversarialContractStress(unittest.TestCase):
    """Stress tests boundary conditions, invalid inputs, and error payload structures."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_invalid_query_page_less_than_1(self):
        """Query parameter validation: page < 1 must return 422 Unprocessable Entity."""
        res = self.client.get("/api/v1/assessments", params={"page": 0})
        self.assertEqual(res.status_code, 422)
        err = res.json()
        self.assertIn("detail", err)

    def test_invalid_query_page_size_exceeds_max(self):
        """Query parameter validation: page_size > 100 must return 422."""
        res = self.client.get("/api/v1/assessments", params={"page_size": 200})
        self.assertEqual(res.status_code, 422)

    def test_nonexistent_assessment_returns_404(self):
        """Verify 404 behavior on non-existent assessment ID."""
        res = self.client.get("/api/v1/assessments/as-nonexistent-999")
        self.assertEqual(res.status_code, 404)
        err = res.json()
        self.assertIn("detail", err)
        self.assertIn("not found", err["detail"].lower())

    def test_nonexistent_recording_analysis_returns_404(self):
        """Verify 404 on audio analysis with non-existent recording."""
        res = self.client.get("/api/v1/assessments/as-fake/recordings/rec-fake/audio-analysis")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
