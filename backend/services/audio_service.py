"""
Aprixia Audio Service: Secure audio ingestion, validation, and storage
"""
import io
import os
import re
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, List

import librosa
import numpy as np
import soundfile as sf

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MIN_DURATION_SECONDS = 0.2  # 200 milliseconds minimum as required by ML pipeline


def sanitize_filename(filename: str) -> str:
    """Removes path traversals and invalid characters."""
    base = os.path.basename(filename)
    safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
    return safe[:128]


def validate_and_inspect_audio(
    file_bytes: bytes, filename: str
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates audio decodability, duration, channels, and sample rate.
    Returns: (is_valid, message, metadata_dict)
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported audio format '{ext}'. Allowed: .wav, .mp3, .m4a, .flac", {}

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, f"Audio file exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES // (1024*1024)}MB)", {}

    try:
        # Load audio using librosa directly from memory buffer
        bio = io.BytesIO(file_bytes)
        audio, sr = librosa.load(bio, sr=16000, mono=True)
        duration = float(len(audio) / sr)

        if duration < MIN_DURATION_SECONDS:
            return False, f"Audio is too short ({duration:.2f}s). Minimum required duration is {MIN_DURATION_SECONDS*1000:.0f}ms.", {}

        # Compute preview peaks (32 normalized amplitude bins) for immediate React waveform display
        step = max(1, len(audio) // 32)
        peaks = [float(abs(val)) for val in audio[::step][:32]]
        max_p = max(peaks) if peaks and max(peaks) > 0 else 1.0
        norm_peaks = [round(p / max_p, 3) for p in peaks]

        metadata = {
            "duration": round(duration, 3),
            "sample_rate": sr,
            "channels": 1,
            "file_size": len(file_bytes),
            "peaks": norm_peaks,
            "audio_sig": audio
        }
        return True, "Audio format validated successfully.", metadata

    except Exception as e:
        return False, f"Unable to decode audio recording: {str(e)}", {}


def save_recording_file(
    storage_dir: Path, file_bytes: bytes, filename: str
) -> Tuple[str, str, Path]:
    """
    Saves audio file securely to disk.
    Returns: (recording_id, stored_filename, full_file_path)
    """
    storage_dir.mkdir(parents=True, exist_ok=True)
    rec_id = f"rec-{uuid.uuid4().hex[:10]}"
    safe_name = sanitize_filename(filename)
    stored_name = f"{rec_id}_{safe_name}"
    target_path = storage_dir / stored_name

    with open(target_path, "wb") as f:
        f.write(file_bytes)

    return rec_id, stored_name, target_path
