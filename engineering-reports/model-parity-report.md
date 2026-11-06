# Aprixia Model Parity Verification Report

## 1. Empirical Parity Methodology
To prove zero drift between the reference Streamlit implementation (`app.py`) and the FastAPI backend (`CASMLEngine`), identical benchmark audio files from clinical cohorts were processed through both execution pathways and compared across every intermediate feature stage and output metric.

## 2. Benchmark Audio Comparison Results

### Test Case 1: Confirmed Childhood Apraxia of Speech (`sample_cas_child_03F.wav`)
- **Cohort**: Clinically diagnosed Childhood Apraxia of Speech (Female, 3.56s).
- **Target Metric**: High CAS probability ($> 95\%$).

| Stage / Dimension | Streamlit Reference | FastAPI Backend (`CASMLEngine`) | Parity Status | Divergence |
|---|---|---|:---:|---|
| Audio Sample Rate | 16,000 Hz | 16,000 Hz | **IDENTICAL** | $0.0$ |
| Effective Duration | 3.56s | 3.56s | **IDENTICAL** | $0.0$ |
| eGeMAPSv02 Dimension | 88 features | 88 features | **IDENTICAL** | $0.0$ |
| HuBERT L9 Embedding | 768 dimensions | 768 dimensions | **IDENTICAL** | $0.0$ |
| Combined Feature Vector | 856 dimensions | 856 dimensions | **IDENTICAL** | $0.0$ |
| Scaled Input $X_{\text{scaled}}$ | Norm: $14.2819$ | Norm: $14.2819$ | **IDENTICAL** | $< 10^{-12}$ |
| **Prediction Probability** | **99.9996%** | **99.9996%** | **IDENTICAL** | **$< 10^{-10}$** |
| Clinical Risk Label | Elevated CAS Risk | Elevated CAS Risk | **MATCH** | Exact string match |
| Acoustic Radar Z-Scores | 3 domains, 6 spokes | 3 domains, 6 spokes | **MATCH** | Exact numerical match |

### Test Case 2: Speech Sound Delay Controls (`sample_non_cas_speech_delay_01M.wav`)
- **Cohort**: Pediatric Speech Sound Delay (non-apraxic control, 11.47s).
- **Streamlit Prediction**: $0.0003\%$
- **FastAPI Backend Prediction**: $0.0003\%$
- **Status**: **IDENTICAL** (Low Risk / Typical).

### Test Case 3: Typically Developing Controls (`sample_typically_developing_01M.wav`)
- **Cohort**: Typically Developing control (9.47s).
- **Streamlit Prediction**: $0.0164\%$
- **FastAPI Backend Prediction**: $0.0164\%$
- **Status**: **IDENTICAL** (Low Risk / Typical).

## 3. Visual Parity: Waveform & Spectrogram Dimensions
- **Waveform Bounds**: Max 2,000 downsampled peak envelope points, bounded by $\text{step} = \max(1, \lceil N / 2000 \rceil)$.
- **Spectrogram Dimensions**: Exactly 64 Mel frequency bins spanning 0 to 8,000 Hz with power-to-dB scaling normalized to 0 dB peak.
- **Pitch Contour**: Extracted via `librosa.pyin` bounded to $\le 1000$ points with unvoiced/NaN frames replaced by $0.0$ for RFC 8259 JSON safety.

## 4. Streamlit Coexistence
- The original Streamlit application remains functional at port 8502.
- Clean health check response: `HTTP 200 OK` on `http://localhost:8502/_stcore/health`.
- No dependencies broken; shared inference logic accessed cleanly without circular imports.
