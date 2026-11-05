"""
Aprixia Assessment Service: Core Business Logic & ML Pipeline Orchestration
Connects database persistence, audio validation, model inference, and results generation.
"""
import io
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func

# Ensure ML package is resolvable on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
ML_DIR = PROJECT_ROOT / "ml-child-speech-disorder-classification"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from ml_engine import CASMLEngine, calculate_aggregate_metrics
from backend.database import SessionLocal
from backend.models.patient import Patient
from backend.models.assessment import Assessment
from backend.models.recording import Recording
from backend.services.serialization_service import dumps_safe_json, loads_safe_json
from backend.services.audio_service import (
    validate_and_inspect_audio,
    save_recording_file,
)

UPLOADS_BASE = PROJECT_ROOT / "backend" / "uploads"
UPLOADS_BASE.mkdir(parents=True, exist_ok=True)


class AssessmentService:
    def __init__(self, db: Session):
        self.db = db
        self.ml_engine = CASMLEngine.get_instance(base_dir=ML_DIR)

    # -------------------------------------------------------------------------
    # IDENTIFIER GENERATION (COLLISION SAFE & DYNAMIC)
    # -------------------------------------------------------------------------
    def generate_patient_id(self) -> str:
        """
        Generates next consecutive APR-XXXX identifier based on highest database record.
        Zero hardcoded APR-0151.
        """
        patients = self.db.query(Patient.patient_id).all()
        max_num = 0
        pattern = re.compile(r"^APR-(\d+)$", re.IGNORECASE)
        for (pid,) in patients:
            match = pattern.match(pid)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        next_num = max_num + 1
        return f"APR-{next_num:04d}"

    def generate_assessment_reference(self) -> str:
        """
        Generates consecutive AS-YYYY-XXX reference if clinician leaves it blank.
        """
        current_year = datetime.utcnow().year
        assessments = self.db.query(Assessment.assessment_reference).all()
        prefix = f"AS-{current_year}-"
        max_num = 0
        for (ref,) in assessments:
            if ref and ref.startswith(prefix):
                try:
                    num = int(ref.split("-")[-1])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass
        next_num = max_num + 1
        return f"AS-{current_year}-{next_num:03d}"

    # -------------------------------------------------------------------------
    # PATIENT CREATION / LOOKUP
    # -------------------------------------------------------------------------
    def get_or_create_patient(
        self,
        child_name: Optional[str],
        age_years: int,
        age_months: int,
        patient_id: Optional[str] = None
    ) -> Patient:
        """
        Reuses existing patient if patient_id is supplied and found; otherwise creates new.
        """
        if patient_id:
            patient = self.db.query(Patient).filter(Patient.patient_id == patient_id).first()
            if patient:
                # Update attributes if provided
                if child_name is not None:
                    patient.child_name = child_name
                patient.age_years = age_years
                patient.age_months = age_months
                patient.updated_at = datetime.utcnow()
                self.db.commit()
                self.db.refresh(patient)
                return patient

        new_pid = patient_id if (patient_id and patient_id.startswith("APR-")) else self.generate_patient_id()
        patient = Patient(
            patient_id=new_pid,
            child_name=child_name,
            age_years=age_years,
            age_months=age_months,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)
        return patient

    # -------------------------------------------------------------------------
    # ASSESSMENT CREATION & LISTING
    # -------------------------------------------------------------------------
    def create_assessment(
        self,
        child_name: Optional[str],
        age_years: int,
        age_months: int,
        patient_id: Optional[str] = None,
        assessment_reference: Optional[str] = None,
        clinical_notes: Optional[str] = None
    ) -> Assessment:
        patient = self.get_or_create_patient(
            child_name=child_name,
            age_years=age_years,
            age_months=age_months,
            patient_id=patient_id
        )

        ref = assessment_reference.strip() if assessment_reference and assessment_reference.strip() else self.generate_assessment_reference()

        # Ensure uniqueness of reference
        existing_ref = self.db.query(Assessment).filter(Assessment.assessment_reference == ref).first()
        if existing_ref:
            ref = self.generate_assessment_reference()

        assessment_id = f"as-{uuid.uuid4().hex[:12]}"
        now_str = datetime.utcnow().strftime("%Y-%m-%d")

        assessment = Assessment(
            assessment_id=assessment_id,
            patient_id=patient.patient_id,
            assessment_reference=ref,
            clinical_notes=clinical_notes,
            assessment_date=now_str,
            status="draft",
            created_at=datetime.utcnow()
        )
        self.db.add(assessment)
        self.db.commit()
        self.db.refresh(assessment)
        return assessment

    def get_assessment(self, assessment_id: str) -> Optional[Assessment]:
        return self.db.query(Assessment).filter(
            (Assessment.assessment_id == assessment_id) |
            (Assessment.assessment_reference == assessment_id)
        ).first()

    def list_assessments(
        self,
        status: Optional[str] = None,
        search_query: Optional[str] = None,
        page: int = 1,
        page_size: int = 10
    ) -> Tuple[List[Dict[str, Any]], int, int, Dict[str, int]]:
        """
        Returns filtered, paginated assessments and real dashboard stats.
        """
        query = self.db.query(Assessment).join(Patient, Assessment.patient_id == Patient.patient_id)

        if status and status != "all":
            query = query.filter(Assessment.status == status)

        if search_query and search_query.strip():
            sq = f"%{search_query.strip().lower()}%"
            query = query.filter(
                (Assessment.assessment_reference.ilike(sq)) |
                (Assessment.patient_id.ilike(sq)) |
                (Patient.child_name.ilike(sq))
            )

        total_matching = query.count()
        total_pages = max(1, (total_matching + page_size - 1) // page_size)
        offset = (max(1, page) - 1) * page_size

        assessments = query.order_by(Assessment.created_at.desc()).offset(offset).limit(page_size).all()

        # Real summary metrics
        total_count = self.db.query(Assessment).count()
        completed_count = self.db.query(Assessment).filter(Assessment.status == "completed").count()
        pending_count = self.db.query(Assessment).filter(Assessment.status.in_(["draft", "uploaded", "processing"])).count()
        flagged_count = self.db.query(Assessment).filter(Assessment.status == "flagged").count()

        stats = {
            "total": total_count,
            "completed": completed_count,
            "pending": pending_count,
            "flagged": flagged_count
        }

        results = []
        for a in assessments:
            results.append({
                "id": a.assessment_id,
                "patient_id": a.patient_id,
                "child_name": a.patient.child_name if a.patient else None,
                "age_years": a.patient.age_years if a.patient else 0,
                "age_months": a.patient.age_months if a.patient else 0,
                "assessment_reference": a.assessment_reference,
                "assessment_date": a.assessment_date,
                "recording_count": len(a.recordings),
                "status": a.status,
                "clinical_notes": a.clinical_notes,
                "has_report": a.status == "completed",
                "mean_score": a.mean_score,
                "risk_label": a.risk_label
            })

        return results, total_matching, total_pages, stats

    # -------------------------------------------------------------------------
    # RECORDINGS INGESTION
    # -------------------------------------------------------------------------
    def add_recording(
        self,
        assessment_id: str,
        file_bytes: bytes,
        filename: str
    ) -> Tuple[bool, str, Optional[Recording]]:
        assessment = self.get_assessment(assessment_id)
        if not assessment:
            return False, f"Assessment '{assessment_id}' not found.", None

        is_valid, msg, meta = validate_and_inspect_audio(file_bytes, filename)
        if not is_valid:
            # Create invalid/rejected record for audit integrity
            rec_id, stored_name, target_path = save_recording_file(
                UPLOADS_BASE / assessment.assessment_id, file_bytes, filename
            )
            recording = Recording(
                recording_id=rec_id,
                assessment_id=assessment.assessment_id,
                original_filename=filename,
                stored_filename=stored_name,
                file_path=str(target_path),
                file_size=len(file_bytes),
                duration=0.0,
                validation_status="invalid",
                processing_status="failed",
                error_message=msg,
                created_at=datetime.utcnow()
            )
            self.db.add(recording)
            self.db.commit()
            self.db.refresh(recording)
            return False, msg, recording

        rec_id, stored_name, target_path = save_recording_file(
            UPLOADS_BASE / assessment.assessment_id, file_bytes, filename
        )

        recording = Recording(
            recording_id=rec_id,
            assessment_id=assessment.assessment_id,
            original_filename=filename,
            stored_filename=stored_name,
            file_path=str(target_path),
            file_size=meta["file_size"],
            duration=meta["duration"],
            sample_rate=meta["sample_rate"],
            channels=meta["channels"],
            validation_status="valid",
            processing_status="pending",
            created_at=datetime.utcnow()
        )
        self.db.add(recording)

        # Update assessment status to uploaded if in draft
        if assessment.status == "draft":
            assessment.status = "uploaded"

        self.db.commit()
        self.db.refresh(recording)
        return True, "Recording ingested successfully.", recording

    # -------------------------------------------------------------------------
    # ML PIPELINE EXECUTION (100% EMPIRICAL INFERENCE)
    # -------------------------------------------------------------------------
    def process_assessment_ml(self, assessment_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Executes real ML inference across all valid recordings for an assessment:
        1. 88 eGeMAPSv02 functionals + 768 HuBERT Layer 9 contextual representations.
        2. Scaler + Calibrated Classifier inference.
        3. Statistical aggregation across utterances (mean, median, range, token inconsistency).
        4. Absolute Z-score speech profile calculation against Typically Developing baselines.
        5. Waveform, Log-Mel spectrogram, and Pitch contour generation for Audio Laboratory.
        """
        assessment = self.get_assessment(assessment_id)
        if not assessment:
            return False, f"Assessment '{assessment_id}' not found.", {}

        valid_recordings = [r for r in assessment.recordings if r.validation_status == "valid"]
        if not valid_recordings:
            return False, "No valid recordings available for machine learning evaluation.", {}

        assessment.status = "processing"
        self.db.commit()

        results_list = []
        errors_list = []

        for rec in valid_recordings:
            rec.processing_status = "processing"
            self.db.commit()

            try:
                with open(rec.file_path, "rb") as f:
                    file_bytes = f.read()

                # Call shared ML engine
                res, err = self.ml_engine.extract_audio_features_and_predict(file_bytes, rec.original_filename)

                if res is not None:
                    sc = float(res["score"])
                    rec.prediction_score = sc
                    rec.duration = float(res["duration"])
                    rec.processing_status = "completed"

                    # Individual recording classification
                    if sc >= 65.0:
                        rec.classification = "Elevated CAS Risk"
                    elif sc >= 40.0:
                        rec.classification = "Borderline"
                    else:
                        rec.classification = "Typical / Low Risk"

                    # Generate Audio Analysis Laboratory payload (Waveform, Spectrogram, Pitch, Metrics)
                    analysis_data = self.ml_engine.generate_audio_analysis_payload(
                        audio_sig=res["audio"],
                        sr=res["sr"],
                        ege_df=res["ege_df"]
                    )
                    rec.analysis_json = dumps_safe_json(analysis_data)
                    results_list.append(res)
                else:
                    rec.processing_status = "failed"
                    rec.error_message = err
                    errors_list.append((rec.original_filename, err))

            except Exception as e:
                rec.processing_status = "failed"
                rec.error_message = str(e)
                errors_list.append((rec.original_filename, str(e)))

            self.db.commit()

        if not results_list:
            assessment.status = "failed"
            self.db.commit()
            return False, "All recordings failed during acoustic feature extraction.", {}

        # ---------------------------------------------------------------------
        # COMPUTE AGGREGATE SCREENING RESULTS (EXACT APP.PY PARITY)
        # ---------------------------------------------------------------------
        scores = [float(r["score"]) for r in results_list]
        agg = calculate_aggregate_metrics(scores)

        assessment.mean_score = agg["mean_score"]
        assessment.median_score = agg["median_score"]
        assessment.min_score = agg["min_score"]
        assessment.max_score = agg["max_score"]
        assessment.token_inconsistency = agg["token_inconsistency"]
        assessment.utterances_evaluated = agg["n_utterances"]
        assessment.risk_label = agg["risk_label"]
        assessment.action_text = agg["action_text"]
        assessment.completed_at = datetime.utcnow()

        # ---------------------------------------------------------------------
        # COMPUTE ACOUSTIC SPEECH PROFILE (EXACT APP.PY PARITY)
        # ---------------------------------------------------------------------
        combined_ege = pd.concat([r["ege_df"] for r in results_list], ignore_index=True)
        mean_child_ege = combined_ege.mean(axis=0).to_frame().T
        speech_profile = self.ml_engine.calculate_speech_profile(mean_child_ege)
        assessment.acoustic_profile_json = dumps_safe_json(speech_profile)

        # Flagged if high risk or high inconsistency
        if agg["mean_score"] >= 65.0:
            assessment.status = "flagged"
        elif errors_list:
            assessment.status = "partially_completed"
        else:
            assessment.status = "completed"

        self.db.commit()
        self.db.refresh(assessment)

        return True, "Assessment evaluated successfully.", {
            "assessment_id": assessment.assessment_id,
            "status": assessment.status,
            "mean_score": assessment.mean_score,
            "risk_label": assessment.risk_label,
            "token_inconsistency": assessment.token_inconsistency,
            "utterances_evaluated": assessment.utterances_evaluated
        }

    # -------------------------------------------------------------------------
    # RESULTS & AUDIO ANALYSIS QUERYING
    # -------------------------------------------------------------------------
    def get_results_payload(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        assessment = self.get_assessment(assessment_id)
        if not assessment:
            return None

        patient = assessment.patient
        age_formatted = f"{patient.age_years}y {patient.age_months}m" if patient else "Age Unspecified"

        utterance_items = []
        for r in assessment.recordings:
            utterance_items.append({
                "recording_id": r.recording_id,
                "audio_file": r.original_filename,
                "duration": round(r.duration, 2),
                "prediction_score": round(r.prediction_score, 2) if r.prediction_score is not None else None,
                "classification": r.classification,
                "status": r.processing_status
            })

        profile = loads_safe_json(assessment.acoustic_profile_json)

        return {
            "assessment_id": assessment.assessment_id,
            "assessment_reference": assessment.assessment_reference,
            "patient_id": assessment.patient_id,
            "child_name": patient.child_name if patient else None,
            "age_formatted": age_formatted,
            "assessment_date": assessment.assessment_date,
            "status": assessment.status,
            "mean_score": assessment.mean_score,
            "median_score": assessment.median_score,
            "min_score": assessment.min_score,
            "max_score": assessment.max_score,
            "token_inconsistency": assessment.token_inconsistency,
            "risk_label": assessment.risk_label,
            "action_text": assessment.action_text,
            "utterances_evaluated": assessment.utterances_evaluated,
            "acoustic_profile": profile,
            "td_baselines": self.ml_engine.td_baselines,
            "utterance_results": utterance_items
        }

    def get_recording_analysis(self, assessment_id: str, recording_id: str) -> Optional[Dict[str, Any]]:
        recording = self.db.query(Recording).filter(
            Recording.assessment_id == assessment_id,
            Recording.recording_id == recording_id
        ).first()

        if not recording:
            return None

        analysis = loads_safe_json(recording.analysis_json)
        if not analysis:
            # If not yet generated, compute from file on the fly
            if Path(recording.file_path).exists():
                audio_sig, sr = librosa.load(recording.file_path, sr=16000, mono=True)
                analysis = self.ml_engine.generate_audio_analysis_payload(audio_sig, sr)
                recording.analysis_json = dumps_safe_json(analysis)
                self.db.commit()

        if not analysis:
            return None

        return {
            "recording_id": recording.recording_id,
            "assessment_id": recording.assessment_id,
            "original_filename": recording.original_filename,
            "duration": recording.duration,
            "sample_rate": recording.sample_rate,
            "prediction_score": recording.prediction_score,
            "classification": recording.classification,
            "waveform": analysis["waveform"],
            "spectrogram": analysis["spectrogram"],
            "pitch_contour": analysis["pitch_contour"],
            "acoustic_features": analysis["acoustic_features"]
        }

    def export_csv(self, assessment_id: str) -> Optional[str]:
        """
        Generates clinical screening CSV report with exact app.py columns.
        """
        assessment = self.get_assessment(assessment_id)
        if not assessment:
            return None

        rows = []
        for r in assessment.recordings:
            rows.append({
                "Audio File": r.original_filename,
                "Duration (s)": round(r.duration, 2),
                "Apraxia Prediction Score (%)": round(r.prediction_score, 2) if r.prediction_score is not None else "N/A",
                "Clinical Classification": r.classification or "Pending"
            })

        df = pd.DataFrame(rows)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue()
