# Aprixia — Clinical Childhood Apraxia of Speech (CAS) Screening & Acoustic Analysis Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.3%20%2B%20TypeScript-61DAFB.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![openSMILE](https://img.shields.io/badge/openSMILE-eGeMAPSv02-orange.svg)](https://audeering.github.io/opensmile-python/)
[![ESP32 Hardware](https://img.shields.io/badge/Hardware-ESP32%20%2B%20INMP441%20I2S-red.svg)](#hardware-voice-acquisition-unit)
[![Backend Tests](https://img.shields.io/badge/Backend%20Tests-86%2F86%20Passing-brightgreen.svg)](#backend-unit-integration-contract--device-suites-86-tests)
[![ML Parity Tests](https://img.shields.io/badge/ML%20Parity-18%2F18%20Passing-brightgreen.svg)](#ml-pipeline-parity-suite-18-tests)
[![Inference Privacy](https://img.shields.io/badge/Inference-100%25%20Local%20%2F%20Air--Gapped-success.svg)](#privacy-security--data-leakage-safeguards)

> [!IMPORTANT]
> **Clinical Screening Disclaimer:** Aprixia is a machine learning–assisted clinical screening, acoustic phonetic analysis, and research support system. It is **not** a standalone diagnostic medical device and must be used exclusively to support—never replace—comprehensive evaluation by a certified Speech-Language Pathologist (SLP) using standardized clinical diagnostic protocols (e.g., VMPAC, DEMSS).

---

## Executive Overview

**Childhood Apraxia of Speech (CAS)** is a neurological pediatric speech sound disorder in which the precision and consistency of movements underlying speech are impaired in the absence of neuromuscular deficits (e.g., abnormal reflexes, abnormal tone). Differential diagnosis between CAS and general Speech Sound Delay (SSD) or Dysarthria remains one of the most challenging areas in speech-language pathology due to high perceptual subjectivity and assessment variability.

**Aprixia** delivers an objective, end-to-end, multi-tier clinical screening platform that bridges hardware-level acoustic acquisition with state-of-the-art transformer representation learning:

1. **Hardware Tier (IoT Voice Acquisition)**: A dedicated **ESP32 + INMP441 I2S MEMS** acoustic unit records calibrated $16\text{ kHz}$ $16\text{-bit}$ mono PCM WAV speech with onboard LittleFS flash storage and a live OLED countdown, bypassing the non-linear gain control, compression, and acoustic artifacts of PC microphones.
2. **Web Frontend Tier**: A modern **React 19 + TypeScript + Tailwind CSS** clinical application featuring a dedicated **Hardware Voice Acquisition Studio**, real-time device telemetry capsules, multi-utterance assessment intake, 3-domain radar baselines, and a time-synchronized multi-tier **Audio Laboratory** (waveform, 64-band log-mel spectrogram, $F_0$ pitch contour).
3. **Backend & Device Proxy Tier**: A **FastAPI + SQLite** REST gateway providing asynchronous hardware device proxying, direct-to-assessment audio ingestion, input validation, and RFC 8259-compliant serialization.
4. **Multimodal ML Engine Tier**: An empirical **856-dimensional** feature architecture combining $88$ `openSMILE eGeMAPSv02` acoustic functionals with $768$ mean-pooled `facebook/hubert-base-ls960` Hidden Layer 9 contextual embeddings, scored by a calibrated `LogisticRegression` classifier ($94.48\%$ Leave-One-Speaker-Out balanced accuracy).

---

## Key Capabilities

- **Integrated Hardware Voice Acquisition Studio**: Seamlessly connects to the ESP32 acoustic recording unit over local Wi-Fi SoftAP (`ESP32_Voice_Recorder`). Clinicians can remotely trigger recordings, monitor live countdowns on the device's OLED screen, inspect LittleFS flash storage allocation, preview audio in-browser, and attach recordings directly into clinical assessment dossiers with one click.
- **Multimodal CAS Screening Engine (`CASMLEngine`)**: Blends $88$ acoustic phonetic functionals (`openSMILE eGeMAPSv02`) capturing prosody, voice quality, and formant dynamics with $768$ deep self-supervised speech representations (`facebook/hubert-base-ls960` Layer 9) into an aligned $856\text{-dimensional}$ space scored by a calibrated `LogisticRegression` classifier ($C=0.1$, balanced class weights, L2 penalty).
- **100% Local & Air-Gapped Privacy**: Audio decoding, acoustic functional extraction, transformer forward passes, and classifier inference execute entirely on the local CPU/host. Zero patient audio, spectrograms, or clinical metadata are transmitted over external networks or cloud AI services.
- **Multi-Utterance Assessment & Token Inconsistency ($\sigma$)**: Evaluates single or repeated speech trials per assessment, calculates aggregate mean, median, minimum, and maximum risk scores, and computes trial-to-trial motor speech variability (token inconsistency $\sigma$), isolating corrupted or sub-$200\text{ ms}$ recordings.
- **3-Domain Acoustic Radar Profile**: Projects acoustic deviations against verified pediatric Typically Developing (TD) reference norms (`td_baselines.joblib`) across **Pitch & Prosody**, **Voice Quality**, and **Vowel Formants**, surfacing clinically interpretable Z-score radar plots.
- **Synchronized Audio Analysis Laboratory**: Interactive Plotly-based acoustic inspection linking time-aligned **amplitude waveform** ($\le 2,000$ downsampled points), **64-band log-mel spectrogram** ($0\text{–}8,000\text{ Hz}$), and **$F_0$ pitch contour** (`librosa.pyin`) with synchronized time-scrubbing, zooming, and pan across all three subplots.
- **Direct-to-Dossier Audio Ingestion Pipeline**: Downloads audio binaries directly from ESP32 LittleFS flash into the backend assessment storage (`backend/uploads/{assessment_id}/`), applies MIME validation and format normalization, creates SQLite database records, and triggers automated ML evaluation without manual file transfer.
- **Dual-Interface Architecture**: Provides both the production **React 19 clinical web application** and the standalone **Streamlit research prototype** (`ml-child-speech-disorder-classification/app.py`), maintaining strict numerical divergence $< 10^{-10}$ via shared `ml_engine.py`.

---

## End-to-End System Architecture

```text
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               TIER 1: HARDWARE VOICE ACQUISITION UNIT                             │
│                                                                                                   │
│   ┌──────────────────────────┐         I2S Bus (GPIO 25, 26, 33)        ┌─────────────────────┐   │
│   │  INMP441 MEMS Microphone │ ───────────────────────────────────────> │  ESP32 Dev Module   │   │
│   │  (Omnidirectional, 24-bit)│                                          │  (Xtensa Dual-Core) │   │
│   └──────────────────────────┘                                          └──────────┬──────────┘   │
│   ┌──────────────────────────┐         I2C Bus (GPIO 21, 22; 0x3C)                 │              │
│   │  SSD1306 128x64 OLED     │ <───────────────────────────────────────────────────┤              │
│   │  (Ready / Live Countdown)│                                                     │              │
│   └──────────────────────────┘                                                     ▼              │
│                                                                   ┌───────────────────────────┐   │
│                                                                   │ LittleFS Flash Partition  │   │
│                                                                   │ /recordings/RECxxxx.wav   │   │
│                                                                   │ 16kHz 16-bit Mono PCM     │   │
│                                                                   └─────────────┬─────────────┘   │
│                                                                                 │                 │
│                                           Embedded HTTP Server (Port 80) / SoftAP Wi-Fi           │
│                                           SSID: ESP32_Voice_Recorder  IP: 192.168.4.1             │
└─────────────────────────────────────────────────────────┬─────────────────────────────────────────┘
                                                          │ REST API & Direct WAV Streaming
                                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           TIER 2: CLINICAL WEB APPLICATION (REACT 19 + VITE)                      │
│                                                                                                   │
│  • Hardware Status Capsule (Global Topbar Telemetry Indicator & Popover)                          │
│  • Hardware Voice Acquisition Studio (/recorder)                                                  │
│    ├── Live Telemetry Metric Strip (Connection, Uptime, Sample Rate, Recording Count)             │
│    ├── FlowForge LittleFS Storage Allocation Gauge (Used vs Free Flash Bytes)                     │
│    ├── Remote Recording Console (Remote Start/Stop Triggers & Live Timer Sync)                    │
│    └── Saved Recordings Vault (Audio Player, Ingestion Modal, Rename, Delete, Download)           │
│  • Clinical Workflow: Dashboard (/) | Patients (/patients) | Assessments (/assessments)           │
│  • Multi-Utterance Intake (/assessments/new) | Assessment Review & Attestation                    │
│  • Synchronized Audio Laboratory (/assessments/:id/laboratory) — Waveform / Mel / F0 Plotly Lab   │
└─────────────────────────────────────────────────────────┬─────────────────────────────────────────┘
                                                          │ Axios HTTP / REST (VITE_API_BASE_URL)
                                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               TIER 3: BACKEND & DEVICE PROXY (FASTAPI)                            │
│                                                                                                   │
│  • Device Proxy Router (/api/v1/device/*):                                                        │
│    ├── GET /status (Telemetry)       ├── GET /recordings (List)     ├── POST /record/start        │
│    ├── GET /recordings/{file}/down   ├── DELETE /recordings/{file}  ├── POST /record/stop         │
│    └── POST /recordings/{file}/attach -> Direct ingestion into AssessmentService pipeline         │
│  • Core Clinical Routers: /patients, /assessments, /recordings, /results, /dashboard              │
│  • Input Validation: Whitelist (.wav, .mp3, .m4a, .flac), <= 50 MB, >= 200 ms, Path Confinement   │
│  • Persistence: SQLAlchemy ORM -> SQLite (backend/aprixia_clinical.db)                            │
│  • Storage: UUID-prefixed quarantined files in backend/uploads/{assessment_id}/                   │
└─────────────────────────────────────────────────────────┬─────────────────────────────────────────┘
                                                          │ Thread-Safe Singleton (INFERENCE_LOCK)
                                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          TIER 4: MULTIMODAL CAS ML INFERENCE ENGINE                               │
│                                                                                                   │
│  1. Audio Ingestion & Resampling -> 16,000 Hz 16-bit Mono (Matches Hardware Native Sampling)       │
│  2. Acoustic Phonetic Extraction: openSMILE eGeMAPSv02 (88 summary functionals)                   │
│  3. Transformer Representations: HuBERT-Base-LS960 Hidden Layer 9 Mean-Pooled Embeddings (768 d)  │
│  4. Feature Alignment: 856-dim Concatenation -> StandardScaler -> Calibrated Logistic Regression  │
│  5. Clinical Risk Classification: Calibrated Probability -> Elevated / Borderline / Low Risk      │
│  6. Acoustic Radar Profiling: Absolute Z-Scores vs Typically Developing Pediatric Norms           │
│  7. Synchronized Visualization Generation: Downsampled Waveform + 64-band Log-Mel + pYIN F0       │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Voice Acquisition Unit

### Clinical Rationale

Standard consumer microphones (laptops, mobile phones, consumer webcams) introduce severe acoustic variability:
- **Automatic Gain Control (AGC)** and non-linear dynamic range compression artificially distort intensity, energy functionals, and harmonic amplitudes.
- **Inconsistent frequency responses** suppress high-frequency formant transitions ($F_2$, $F_3$, $F_4$) critical for vowel distinctiveness.
- **Ambient room noise and browser clipping** contaminate jitter, shimmer, and spectral flux metrics.

To eliminate acoustic drift, Aprixia integrates a dedicated **hardware voice capture device** engineered specifically for pediatric speech assessment:
- **Native $16\text{ kHz}$ $16\text{-bit}$ Linear PCM capture** ensures zero-resampling alignment with the HuBERT and openSMILE feature extractors.
- **Studio-grade MEMS digital conversion** converts acoustics directly to digital $I^2S$ signals at the microphone head, eliminating analog wire noise.
- **Local flash persistence (LittleFS)** guarantees zero audio packet loss even under unstable network environments.
- **Standalone Wi-Fi SoftAP** allows air-gapped clinical deployments in school or hospital environments without requiring hospital Wi-Fi access.

### Hardware Specifications & Bill of Materials (BOM)

| Component | Part / Model | Specification / Role |
|---|---|---|
| **Microcontroller** | ESP32-WROOM-32 / ESP32 DevKit v1 | 32-bit Xtensa dual-core ($240\text{ MHz}$), $520\text{ KB}$ SRAM, $4\text{ MB}$ SPI Flash, Wi-Fi $802.11\text{ b/g/n}$ |
| **Microphone** | INMP441 MEMS Microphone Module | Omnidirectional, $I^2S$ digital output, $61\text{ dBA}$ SNR, flat $60\text{ Hz – }15\text{ kHz}$ frequency response |
| **Display** | $0.96\text{''}$ I2C OLED (SSD1306) | $128 \times 64$ monochrome OLED, $I^2C$ interface ($0\text{x3C}$ address), live timer and status feedback |
| **Storage** | LittleFS Flash Partition | Onboard SPI Flash allocated for circular audio recording directory (`/recordings/`) |
| **Power** | Micro-USB / 5V DC or 3.7V LiPo | Standard $5\text{V}$ USB power or portable single-cell lithium polymer battery with $3.3\text{V}$ LDO regulator |

### Pinout & Wiring Connections

```text
  INMP441 I2S Microphone                 ESP32 DevKit v1                SSD1306 0.96" OLED
┌────────────────────────┐            ┌───────────────────┐            ┌──────────────────┐
│  VDD (Power)           │ ─────────> │ 3V3               │ <───────── │ VCC (Power)      │
│  GND (Ground)          │ ─────────> │ GND               │ <───────── │ GND (Ground)     │
│  SD  (Serial Data In)  │ ─────────> │ GPIO 33 (I2S_DIN) │            │                  │
│  WS  (Word Select)     │ ─────────> │ GPIO 25 (I2S_WS)  │            │                  │
│  SCK (Bit Clock)       │ ─────────> │ GPIO 26 (I2S_BCLK)│            │                  │
│  L/R (Channel Select)  │ ─────────> │ GND (Left Channel)│            │                  │
│                        │            │ GPIO 21 (SDA)     │ ─────────> │ SDA (Data)       │
│                        │            │ GPIO 22 (SCL)     │ ─────────> │ SCL (Clock)      │
└────────────────────────┘            └───────────────────┘            └──────────────────┘
```

#### Detailed Pin Connection Table

| Peripheral | Peripheral Pin | ESP32 GPIO | Description / Notes |
|---|---|---|---|
| **INMP441** | `VDD` | `3V3` | $3.3\text{V}$ regulated power supply (do **not** connect to $5\text{V}$) |
| **INMP441** | `GND` | `GND` | Common ground reference |
| **INMP441** | `SD` (Serial Data) | `GPIO 33` | $I^2S$ Serial Data Input (`I2S_DIN`) |
| **INMP441** | `WS` (Word Select) | `GPIO 25` | $I^2S$ Word Select / L-R Clock (`I2S_WS`) |
| **INMP441** | `SCK` (Serial Clock) | `GPIO 26` | $I^2S$ Bit Clock (`I2S_BCLK`) |
| **INMP441** | `L/R` (Channel) | `GND` | Hardwired to `GND` to select Left audio channel mono recording |
| **SSD1306** | `VCC` | `3V3` | $3.3\text{V}$ DC power supply |
| **SSD1306** | `GND` | `GND` | Common ground reference |
| **SSD1306** | `SDA` | `GPIO 21` | $I^2C$ Data line (`Wire` default SDA) |
| **SSD1306** | `SCL` | `GPIO 22` | $I^2C$ Clock line (`Wire` default SCL, address `0x3C`) |

### Firmware Architecture (`hardware/esp32_voice_recorder/esp32_voice_recorder.ino`)

The firmware implements a fully autonomous, real-time embedded acquisition system:

1. **Direct $I^2S$ DMA Double-Buffering**:
   - Audio is streamed through an internal ESP32 DMA buffer ring (`dma_buf_count=8`, `dma_buf_len=256`).
   - The INMP441 transmits $24\text{-bit}$ samples left-aligned in a $32\text{-bit}$ slot. The firmware converts samples in-flight to $16\text{-bit}$ linear PCM (`pcm = (int16_t)(raw >> 16)`).
   - Audio is captured at exactly $16,000\text{ Hz}$ mono, yielding an exact byte rate of $32,000\text{ bytes/second}$.
2. **Standard 44-byte WAV RIFF Header Synthesis**:
   - The recording engine reserves a $44\text{-byte}$ header upon file creation.
   - Upon termination (via remote stop, serial stop, or reaching the $30\text{-second}$ ceiling), the exact data chunk size and RIFF size are populated in standard little-endian PCM format before finalizing the file.
3. **LittleFS Internal Flash Storage**:
   - Audio is committed to internal SPI flash under `/recordings/REC%04u.wav`.
   - Before each recording, flash memory is inspected to verify that at least $964\text{ KB}$ free space exists ($30\text{ s} \times 16,000\text{ samples/s} \times 2\text{ bytes} + 4\text{ KB}$ overhead).
4. **Autonomous OLED Feedback**:
   - Displays real-time operational status: `INITIALIZING`, `READY`, `RECORDING...` with dynamic countdown timer (`00:30` down to `00:00`), `RECORDING SAVED`, and `STORAGE FULL` alerts.
5. **SoftAP Web Server & Cross-Origin REST API**:
   - Operates a local Wi-Fi Access Point (`ESP32_Voice_Recorder` / `Voice@1234`) with IP `192.168.4.1`.
   - Exposes both a mobile-friendly HTML web portal (`/`) and a JSON REST API with complete CORS headers (`Access-Control-Allow-Origin: *`).
6. **Serial UART Control (115200 baud)**:
   - Supports interactive terminal commands: `START`, `STOP`, `STATUS`, `LIST`.

### Hardware Flashing & Setup Guide

#### Prerequisites
- [Arduino IDE 2.x](https://www.arduino.cc/en/software) or [PlatformIO](https://platformio.org/)
- Silicon Labs CP210x or CH340 USB-to-UART bridge drivers installed

#### Step 1: Configure Arduino IDE
1. Open **Preferences** in Arduino IDE and add the ESP32 board manager URL:
   ```text
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
2. Navigate to **Tools > Board > Boards Manager**, search for `esp32` by Espressif, and install version `2.0.14` or higher (tested on `2.0.x` and `3.x`).
3. Install required libraries via **Tools > Manage Libraries**:
   - `Adafruit SSD1306` (v2.5.9+)
   - `Adafruit GFX Library` (v1.11.9+)
   - `LittleFS_esp32` (built into ESP32 core)

#### Step 2: Board Settings
Under the **Tools** menu, select the following parameters:
- **Board**: `ESP32 Dev Module`
- **Upload Speed**: `921600` (or `115200` if upload fails)
- **CPU Frequency**: `240MHz (WiFi/BT)`
- **Flash Frequency**: `80MHz`
- **Flash Mode**: `QIO`
- **Flash Size**: `4MB (32Mb)`
- **Partition Scheme**: `Default 4MB with spiffs (1.2MB APP/1.5MB SPIFFS)` or `No OTA (2MB APP/2MB SPIFFS)`
- **Port**: Select the active COM port (e.g., `COM3`, `COM5`)

#### Step 3: Flash the Firmware
1. Open [`hardware/esp32_voice_recorder/esp32_voice_recorder.ino`](hardware/esp32_voice_recorder/esp32_voice_recorder.ino).
2. Connect your ESP32 board via USB.
3. Click **Upload**. (If the board does not auto-reset into bootloader mode, press and hold the `BOOT` button on the ESP32 until compilation finishes and flashing begins).
4. Open the Serial Monitor at **115200 baud** to verify initial boot:
   ```text
   LittleFS mounted successfully!
   Wi-Fi access point started.
   Wi-Fi name: ESP32_Voice_Recorder
   Web address: http://192.168.4.1
   ESP32 INMP441 VOICE RECORDER READY
   ```

---

## Hardware Voice Acquisition Studio (Web Interface)

Aprixia provides full hardware monitoring and bidirectional control directly inside the React clinical application:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  APRIXIA CLINICAL  |  [Patients]  [Assessments]  [Hardware Studio]    [ESP32: ONLINE | 2 REC] ▼ │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Persistent Global Telemetry Capsule (`HardwareStatusCapsule`)**:
   - Mounted in the application header across all clinical routes.
   - Continuously polls device health via `/api/v1/device/status`.
   - Displays real-time connection state (`Connected` / `Recording` / `Offline`) with an interactive popover showing free LittleFS bytes, sample rate, uptime, and active Wi-Fi clients.
2. **Dedicated Studio View (`/recorder`)**:
   - **Telemetry Metric Strip**: 4-card KPI grid displaying Connection Status, Session Uptime, Sampling Rate ($16\text{ kHz}$), and Total Stored Recordings.
   - **FlowForge Storage Allocation Bar**: Visual memory gauge illustrating used vs. free LittleFS flash capacity with automatic low-storage warnings.
   - **Live Recording Console**: Remote trigger buttons (**Start Recording** / **Stop Recording**) with synchronized $30\text{-second}$ countdown timer.
   - **Acoustic Asset Vault Table**: Interactive table listing all audio assets on the board with audio duration, file size, in-browser playback preview, inline file renaming, safe deletion, and direct download.
3. **One-Click "Attach to Assessment" Modal**:
   - Available both in the Vault table and directly inside the Assessment Intake & Review workspaces.
   - Clinicians can select any device-captured recording and link it directly to an active patient assessment (`POST /api/v1/device/recordings/{filename}/attach`).
   - The backend downloads the audio binary over Wi-Fi, passes it through acoustic format validation, saves it into the assessment's directory, and registers it in the SQLite database ready for immediate ML inference.

---

## Machine Learning Architecture & Clinical Profiling

### 856-Dimensional Multimodal Feature Space

The `CASMLEngine` singleton processes raw audio through an aligned two-stream feature extractor:

```text
Raw Audio (WAV, 16 kHz Mono)
   ├── Stream A: openSMILE eGeMAPSv02 ───────────────>   88 Acoustic Functionals
   │   • F0 & Formants (F1, F2, F3)                    (Pitch, Prosody, Formant Dynamics)
   │   • Jitter, Shimmer, HNR, Spectral Flux
   │
   └── Stream B: HuBERT-Base-LS960 (Layer 9) ────────>  768 Contextual Embeddings
       • Hidden Layer 9 Mean-Pooled Representation     (Deep Phonetic Motor Representations)
                                                               │
                                                               ▼
                                                  Concatenated 856-dim Vector
                                                               │
                                                               ▼
                                                   StandardScaler Normalization
                                                               │
                                                               ▼
                                                 Calibrated Logistic Regression
                                                  (C=0.1, L2, Balanced Weights)
                                                               │
                                                               ▼
                                                  Clinical CAS Risk Probability
```

### Clinical Risk Classification Thresholds

Implemented identically in [`calculate_aggregate_metrics`](ml-child-speech-disorder-classification/ml_engine.py):

| Mean Risk Score | Risk Classification | Recommended Clinical Action |
|---|---|---|
| **$\ge 65.0\%$** | `Elevated CAS Risk` | Comprehensive Motor Speech Evaluation (VMPAC / DEMSS) Recommended |
| **$40.0\% \text{ – } 64.99\%$** | `Borderline / Monitoring Needed` | Active Clinical Monitoring & Re-Screening Advised |
| **$< 40.0\%$** | `Low Risk / Typical Speech Sound Development` | Acoustic Parameters Consistent with Typical Pediatric Norms |

### Acoustic Radar Profile (3 Clinical Domains)

Acoustic divergence is benchmarked against confirmed pediatric Typically Developing norms (`td_baselines.joblib`):

$$\text{Z-Score} = \left| \frac{x_{\text{patient}} - \mu_{\text{TD}}}{\sigma_{\text{TD}}} \right|$$

- **Domain 1: Pitch & Prosody**: Evaluates $F_0$ variability, pitch coefficient of variation, and prosodic contour stability.
- **Domain 2: Voice Quality**: Evaluates vocal jitter, shimmer, harmonics-to-noise ratio (HNR), and spectral flux.
- **Domain 3: Vowel Formants**: Measures $F_1$, $F_2$, and $F_3$ formant frequency dispersion and bandwidth consistency reflecting articulatory vowel space precision.

### Token Inconsistency Metric ($\sigma$)

In accordance with ASHA diagnostic guidelines for CAS, trial-to-trial motor speech variability is computed across repeated token utterances:

$$\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (p_i - \bar{p})^2}$$

Elevated token inconsistency ($\sigma > 15\%$) combined with elevated mean risk provides a strong acoustic indicator of motor planning inconsistency.

### Validation & Parity Highlights

- **Leave-One-Speaker-Out (LOSO) Balanced Accuracy**: **94.48%** evaluated across clinical corpora.
- **Cross-Implementation Parity**: Strict numerical divergence $< 10^{-10}$ verified between FastAPI backend and Streamlit reference prototype:
  - `sample_cas_child_03F.wav` (Confirmed CAS child): **99.9996%** risk score.
  - `sample_non_cas_speech_delay_01M.wav` (Speech Sound Delay child): **0.0003%** risk score.
  - `sample_typically_developing_01M.wav` (Typically Developing child): **0.0164%** risk score.
- **Inference Latency**: Mean **417 ms** per utterance on CPU.

---

## Repository Structure

```text
Apraxia/
├── hardware/                                      # Embedded hardware firmware & acquisition setup
│   └── esp32_voice_recorder/
│       └── esp32_voice_recorder.ino              # Complete ESP32 I2S + OLED + LittleFS + REST API firmware
├── backend/                                       # FastAPI REST application & SQLite persistence
│   ├── main.py                                    # Lifespan startup, ML warmup, router mounts, CORS
│   ├── database.py                                # SQLAlchemy SQLite engine & session management
│   ├── requirements.txt                           # Core backend API dependencies (includes httpx)
│   ├── models/                                    # SQLAlchemy ORM models: Patient, Assessment, Recording
│   ├── routes/                                    # REST API route handlers
│   │   ├── patients.py                            # Patient registration & ID assignment
│   │   ├── assessments.py                         # Assessment dossier lifecycle & query endpoints
│   │   ├── recordings.py                          # Audio upload, validation & streaming endpoints
│   │   ├── results.py                             # ML execution, results, radar profile, CSV export
│   │   ├── dashboard.py                           # Clinical summary statistics & timeline overview
│   │   └── device.py                              # ESP32 Hardware proxy & direct-attach router
│   ├── schemas/                                   # Pydantic v2 validation contracts
│   │   ├── patient.py, assessment.py, recording.py, results.py, dashboard.py
│   │   └── device.py                              # Hardware telemetry & recording action schemas
│   ├── services/                                  # Business logic & hardware coordination
│   │   ├── assessment_service.py                  # Assessment orchestration & SQLite transactions
│   │   ├── audio_service.py                       # Audio format whitelisting, sanitization & decoding
│   │   └── device_service.py                      # Async HTTP communication with ESP32 board
│   └── tests/                                     # 86 automated unit, contract, security & device tests
│       ├── test_all_endpoints.py                  # Comprehensive test of all 31 registered API endpoints
│       ├── test_device_proxy.py                   # Dedicated ESP32 proxy, telemetry & attach tests
│       ├── test_parity_and_endpoints.py           # ML parity & core endpoint lifecycle tests
│       ├── test_api_contract.py                   # OpenAPI schema compliance & contracts
│       └── test_security_performance.py           # Path traversal, upload security & latency benchmarks
├── frontend/                                      # React 19 + TypeScript + Vite clinical web application
│   ├── src/
│   │   ├── app/                                   # App shell and React Router definitions
│   │   ├── components/
│   │   │   ├── layout/                            # AppLayout, navigation, and HardwareStatusCapsule
│   │   │   ├── assessment/                        # Intake, Review, RecorderAttachModal, AudioLab
│   │   │   └── recorder/                          # Hardware Voice Studio components:
│   │   │       ├── RecorderMetricsStrip.tsx       # Telemetry KPI metric cards
│   │   │       ├── FlowForgeStorageBar.tsx        # LittleFS memory allocation gauge
│   │   │       ├── RecordingConsoleDeck.tsx       # Live recording controller & countdown timer
│   │   │       ├── DeviceRecordingsVault.tsx      # Acoustic asset table & action controls
│   │   │       └── AttachToAssessmentModal.tsx    # Direct-to-assessment modal dialog
│   │   ├── hooks/                                 # Custom React hooks (useDeviceTelemetry, useAudioPlayer)
│   │   ├── pages/                                 # Route views: Dashboard, Patients, Assessments,
│   │   │                                          # AudioLaboratory, and RecorderSession (/recorder)
│   │   ├── services/                              # Axios API client layer:
│   │   │   ├── apiClient.ts                       # Base HTTP client with base URL resolution
│   │   │   ├── assessmentApi.ts                   # Core clinical API methods
│   │   │   └── deviceApi.ts                       # Hardware device proxy API methods
│   │   └── styles/                                # Tailwind CSS v4 & custom recorder styling
│   ├── tests/clinical_browser_e2e.spec.ts         # Playwright end-to-end browser test suite (10 scenarios)
│   ├── playwright.config.ts                       # Playwright configuration
│   └── package.json                               # Frontend dependencies & build scripts
├── ml-child-speech-disorder-classification/       # Shared ML engine & standalone Streamlit prototype
│   ├── ml_engine.py                               # Thread-safe CASMLEngine singleton shared with backend
│   ├── app.py                                     # Reference Streamlit clinical screening prototype
│   ├── test_ml_parity.py                          # 18 ML pipeline integrity and numerical parity tests
│   ├── requirements.txt                           # Full scientific Python environment
│   ├── *.joblib                                   # Pre-trained classifier, scaler & TD baseline artifacts
│   └── samples/                                   # Verified benchmark reference audio recordings
├── engineering-reports/                           # Complete technical audit & QA documentation
├── screenshots/                                   # Clinical platform interface screenshots
├── .gitignore                                     # Strict data-leakage & HIPAA/GDPR exclusion rules
└── README.md
```

---

## REST API Reference

The Aprixia backend registers **31 REST endpoints** organized across 7 functional domains:

### 1. Hardware Device Proxy API (`/api/v1/device`)

| Method & Route | Request Body / Params | Description |
|---|---|---|
| `GET /api/v1/device/status` | — | Returns real-time ESP32 connection status, storage breakdown, sample rate, uptime, and Wi-Fi clients. |
| `GET /api/v1/device/recordings` | — | Lists all WAV audio files stored in ESP32 LittleFS flash with file size and duration. |
| `GET /api/v1/device/recordings/{filename}/download` | Path: `filename` | Streams raw WAV audio binary directly from ESP32 to client. |
| `DELETE /api/v1/device/recordings/{filename}` | Path: `filename` | Deletes an audio recording from ESP32 LittleFS flash memory. |
| `POST /api/v1/device/recordings/{filename}/rename` | `{"new_name": str}` | Renames a recording stored on ESP32 LittleFS flash. |
| `POST /api/v1/device/recordings/{filename}/attach` | `{"assessment_id": str, "custom_filename": Optional[str]}` | **Direct Ingestion**: Downloads audio binary from ESP32 and ingests it into assessment dossier with validation and persistence. |
| `POST /api/v1/device/record/start` | — | Remotely triggers audio recording initiation on ESP32 hardware. |
| `POST /api/v1/device/record/stop` | — | Remotely stops recording and finalizes standard WAV RIFF header on ESP32. |

### 2. Standalone ESP32 Native API (`http://192.168.4.1`)

The ESP32 firmware natively serves these endpoints over Wi-Fi SoftAP (with full CORS support):

| Endpoint | Method | Response / Payload | Purpose |
|---|---|---|---|
| `/` | `GET` | HTML Web Page | Standalone mobile-friendly web portal for direct downloads and management. |
| `/api/status` | `GET` | JSON Object | Live telemetry, recording state, LittleFS storage bytes, and uptime. |
| `/api/recordings` | `GET` | JSON Array | Array of stored WAV files with byte size and calculated duration. |
| `/api/recordings/download` | `GET` (`?file=REC0001.wav`) | `audio/wav` Binary | Direct file download with attachment header. |
| `/api/recordings/delete` | `POST` (`{"file": "REC0001.wav"}`) | JSON Object | Remove recording from LittleFS. |
| `/api/recordings/rename` | `POST` (`{"old_file": "...", "new_name": "..."}`) | JSON Object | Rename recording file. |
| `/api/record/start` | `POST` | JSON Object | Begin I2S recording. |
| `/api/record/stop` | `POST` | JSON Object | Stop recording and finalize WAV header. |

### 3. Core Clinical API

| Domain | Method & Endpoint | Purpose |
|---|---|---|
| **Health** | `GET /api/v1/health` | Service readiness, ML warmup status, model architecture metadata. |
| **Dashboard** | `GET /api/v1/dashboard/stats` | Clinical KPI counts (total patients, assessments, completed screenings). |
| | `GET /api/v1/dashboard/overview` | Formatted clinical dashboard overview with chronological activity. |
| **Patients** | `GET /api/v1/patients/next-id` | Generate collision-safe next pediatric identifier (`APR-XXXX`). |
| | `POST /api/v1/patients` | Register a pediatric patient record. |
| | `GET /api/v1/patients` | Paginated search and directory of registered patients (`?search=&limit=`). |
| | `GET /api/v1/patients/{patient_id}` | Retrieve individual patient demographic profile. |
| **Assessments** | `POST /api/v1/assessments` | Create a new screening assessment dossier. |
| | `GET /api/v1/assessments` | Paginated archive of dossiers (`?status=&search=&page=&page_size=`). |
| | `GET /api/v1/assessments/stats` | Breakdown counts of assessments by status. |
| | `GET /api/v1/assessments/{assessment_id}` | Comprehensive assessment detail with linked patient and recordings. |
| | `GET /api/v1/assessments/{assessment_id}/status` | Lightweight processing status polling endpoint. |
| | `DELETE /api/v1/assessments/{assessment_id}` | Cascading deletion of assessment and linked recordings. |
| **Recordings** | `POST /api/v1/assessments/{assessment_id}/recordings` | Multipart upload (`files`) with MIME, size, and duration validation. |
| | `GET /api/v1/assessments/{assessment_id}/recordings/{recording_id}` | Audio validation and inference metadata. |
| | `GET /api/v1/assessments/{assessment_id}/recordings/{recording_id}/audio` | Stream stored audio recording binary. |
| | `DELETE /api/v1/assessments/{assessment_id}/recordings/{recording_id}` | Delete individual recording from disk and database. |
| **ML & Results** | `POST /api/v1/assessments/{assessment_id}/process` | Trigger in-process ML feature extraction, inference, and radar profiling. |
| | `GET /api/v1/assessments/{assessment_id}/results` | Aggregate risk scores, clinical classification, and utterance breakdown. |
| | `GET /api/v1/assessments/{assessment_id}/acoustic-profile` | 3-domain Z-score radar deviations vs. TD reference baselines. |
| | `GET /api/v1/assessments/{assessment_id}/recordings/{recording_id}/audio-analysis` | Downsampled waveform, 64-band log-mel spectrogram, $F_0$ pitch, and eGeMAPS metrics. |
| | `GET /api/v1/assessments/{assessment_id}/export` | Download complete clinical screening dossier as CSV. |

---

## Getting Started

### System Prerequisites

- **Python 3.11+** (64-bit)
- **Node.js 18+** and **npm 9+**
- **C++ Build Tools / Libsndfile** (standard with modern Python wheels)
- **ESP32 Dev Module** with INMP441 microphone (for hardware voice acquisition)
- **Internet access during first startup** (to download Hugging Face `facebook/hubert-base-ls960` weights once; all subsequent runs operate 100% offline)

---

### Step 1: Python Virtual Environment & ML Setup

From the repository root (`Apraxia/`):

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# (On macOS/Linux: source .venv/bin/activate)

# Upgrade pip and install full ML & backend dependencies
python -m pip install --upgrade pip
pip install -r ml-child-speech-disorder-classification/requirements.txt
pip install -r backend/requirements.txt
```

---

### Step 2: Start the FastAPI Backend & Device Gateway

Run from the **repository root** so Python resolves module paths correctly:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Documentation: `http://127.0.0.1:8000/docs`
- Service Health Check: `GET http://127.0.0.1:8000/api/v1/health`

> **Hardware Host Configuration:** By default, the backend proxies hardware requests to `http://192.168.4.1` (the default ESP32 SoftAP IP). If running the ESP32 in station mode on a local network, specify the custom host using the environment variable:
> ```powershell
> $env:ESP32_HOST="http://192.168.1.150"
> ```

---

### Step 3: Start the React 19 Clinical Frontend

In a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

- Clinical Web Application: `http://localhost:5173`
- Optional environment override (`frontend/.env.local`):
  ```env
  VITE_API_BASE_URL=http://127.0.0.1:8000
  ```

---

### Step 4: Connect to the ESP32 Hardware Device

1. Power on the flashed ESP32 device via USB. The OLED display will show `VOICE RECORDER READY`.
2. On your computer or tablet, connect to the Wi-Fi network:
   - **SSID**: `ESP32_Voice_Recorder`
   - **Password**: `Voice@1234`
3. Navigate to **Hardware Studio** in the Aprixia web application (`http://localhost:5173/recorder`).
4. The header status capsule will transition to **ESP32 UNIT: ONLINE**. You can now remotely trigger speech recording sessions or attach captured audio directly to assessments.

---

### Step 5: (Optional) Standalone Streamlit Research Prototype

To inspect the research baseline interface:

```powershell
cd ml-child-speech-disorder-classification
streamlit run app.py
```

---

## Testing & Quality Assurance

Aprixia incorporates a multi-tier automated test harness covering embedded device proxying, REST API contracts, path traversal security, ML numerical parity, and browser end-to-end clinical workflows.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APRIXIA AUTOMATED TEST HARNESS                     │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ Backend Test Suite       │ 86 Tests Passing (100%)  │ Unit, Contract, Proxy │
│ ML Parity Test Suite     │ 18 Tests Passing (100%)  │ Numerical Fidelity    │
│ Frontend Production Build│ 0 Errors (tsc -b & vite) │ Type Safety & Bundler │
│ Playwright Browser Suite │ 10 Scenarios Passing     │ End-to-End E2E Flows  │
└──────────────────────────┴──────────────────────────┴───────────────────────┘
```

### Backend Unit, Integration, Contract & Device Suites (86 Tests)

Run from the repository root:

```powershell
python -m unittest discover -s backend/tests -v
```

Test suite coverage breakdown:
- [`backend/tests/test_all_endpoints.py`](backend/tests/test_all_endpoints.py) (**22 tests**): Comprehensive verification across all 31 registered API endpoints, ensuring zero HTTP errors, correct serialization, and expected response models.
- [`backend/tests/test_device_proxy.py`](backend/tests/test_device_proxy.py) (**14 tests**): ESP32 telemetry parsing, recording vault listing, WAV binary streaming, remote record initiation/termination, LittleFS error handling, and end-to-end direct-to-assessment audio ingestion.
- [`backend/tests/test_parity_and_endpoints.py`](backend/tests/test_parity_and_endpoints.py) (**16 tests**): Deterministic model prediction parity, multi-utterance aggregation, CSV export, and RFC 8259-compliant JSON serialization.
- [`backend/tests/test_api_contract.py`](backend/tests/test_api_contract.py) (**12 tests**): OpenAPI schema compliance, query validation parameters, and multipart upload request/response contracts.
- [`backend/tests/test_security_performance.py`](backend/tests/test_security_performance.py) (**22 tests**): Windows and POSIX path traversal defense, upload MIME whitelisting, $50\text{ MB}$ payload boundary enforcement, sub-$200\text{ ms}$ audio rejection, memory stability, and inference latency benchmarking.

### ML Pipeline Parity Suite (18 Tests)

Verifies identical feature dimensions ($856$), scaling factors, and inference scores between FastAPI and Streamlit:

```powershell
python ml-child-speech-disorder-classification/test_ml_parity.py
```

### Frontend Type-Check & Production Build

```powershell
cd frontend
npm run lint
npm run build
```

### End-to-End Browser Suite (Playwright — 10 Scenarios)

```powershell
cd frontend
npm run test:e2e
```

Drives headless Chromium through live patient intake, assessment dossier generation, real `.wav` audio upload, ML execution, synchronized acoustic laboratory inspection, CSV export, and negative boundary validation.

---

## Privacy, Security & Data-Leakage Safeguards

- **100% In-Process Local Execution**: Transformer weights (`facebook/hubert-base-ls960`) and openSMILE functionals run exclusively on local host compute. No speech audio, acoustic features, or patient metadata are ever transmitted to third-party APIs or cloud infrastructure.
- **Microcontroller Audio Isolation**: The ESP32 operates as an isolated Wi-Fi Access Point without internet routing. Audio recorded to LittleFS flash cannot leak externally and is wiped upon clinician deletion.
- **Filename Sanitization & Path Traversal Mitigations**: Both the FastAPI backend (`backend/services/audio_service.py`) and the ESP32 firmware (`esp32_voice_recorder.ino`) enforce strict filename whitelisting (`[a-zA-Z0-9_\-\.]`), stripping null bytes, Unix directory traversal (`../`), and Windows drive letters (`C:\`).
- **Strict Payload & Format Boundaries**:
  - Whitelist: `.wav`, `.mp3`, `.m4a`, `.flac`.
  - Size ceiling: $50\text{ MB}$ maximum upload size; $0\text{-byte}$ empty files rejected immediately.
  - Duration lower bound: Audio $< 200\text{ ms}$ rejected to prevent mathematical instability in $F_0$ pitch extraction and HuBERT attention layers.
- **Repository Data-Leakage Protections (`.gitignore`)**:
  - Raw pediatric research corpora (UltraSuite `UPX`, `UXSSD`, `UXTD`, ENNI, TORGO) strictly excluded.
  - Clinical databases (`backend/aprixia_clinical.db*`, `*.sqlite*`) and patient uploads (`backend/uploads/`) barred from version control.
  - Generated clinical dossiers and CSV reports barred from version control.

---

## Clinical User Interface Gallery

| Clinical View | Description | Screenshot Reference |
|---|---|---|
| **Clinical Dashboard** | Summary of clinical workload, patient counts, risk distribution, and quick intake actions. | [`screenshots/01a-dashboard-overview.png`](screenshots/01a-dashboard-overview.png) |
| **Patients Directory** | Searchable directory of registered pediatric patients with age and assessment history. | [`screenshots/02-patients-directory.png`](screenshots/02-patients-directory.png) |
| **Assessments Archive** | Comprehensive status-filtered archive of past screening dossiers with diagnostic outcomes. | [`screenshots/03-assessments-archive.png`](screenshots/03-assessments-archive.png) |
| **New Assessment Intake** | Multi-utterance speech recording upload workspace with file validation feedback. | [`screenshots/04-new-assessment.png`](screenshots/04-new-assessment.png) |
| **Clinical Review & Attestation** | Clinician review workspace with SLP signature attestation and diagnostic notes. | [`screenshots/05-assessment-review-redesigned.png`](screenshots/05-assessment-review-redesigned.png) |
| **Results Dossier & Radar** | Calibrated CAS risk score, token inconsistency ($\sigma$), and 3-domain acoustic radar plot. | [`screenshots/06a-assessment-results-overview.png`](screenshots/06a-assessment-results-overview.png) |
| **Synchronized Audio Lab** | Time-locked Plotly inspection of waveform ($\le 2000$ pts), log-mel spectrogram, and $F_0$ contour. | [`screenshots/07-audio-laboratory.png`](screenshots/07-audio-laboratory.png) |
| **Clinical Reports & Export** | One-click export of clinical dossiers into RFC 8259-compliant CSV formats. | [`screenshots/08-reports.png`](screenshots/08-reports.png) |

---

## Engineering & Audit Documentation

For deep-dive technical reports, consult the [`engineering-reports/`](engineering-reports/) directory:

- [`engineering-reports/executive-summary.md`](engineering-reports/executive-summary.md) — Comprehensive architecture and multi-track audit outcomes.
- [`engineering-reports/final-acceptance-report.md`](engineering-reports/final-acceptance-report.md) — Quality gate verification checklist.
- [`engineering-reports/ml-integrity-report.md`](engineering-reports/ml-integrity-report.md) — Multimodal feature engineering fidelity and zero-leakage verification.
- [`engineering-reports/model-parity-report.md`](engineering-reports/model-parity-report.md) — Numerical parity across FastAPI and Streamlit implementations.
- [`engineering-reports/api-endpoint-matrix.md`](engineering-reports/api-endpoint-matrix.md) — Complete REST route specifications and response schemas.
- [`engineering-reports/security-audit.md`](engineering-reports/security-audit.md) — Security boundaries, path traversal defense, and local processing enforcement.
- [`engineering-reports/performance-report.md`](engineering-reports/performance-report.md) — Inference latency benchmarks and memory footprint profiling.
- [`engineering-reports/qa-test-report.md`](engineering-reports/qa-test-report.md) — Unit and integration test execution logs.
- [`engineering-reports/e2e-test-report.md`](engineering-reports/e2e-test-report.md) — Playwright browser automation test results.

---

## License & Clinical Attribution

This project is developed for clinical and acoustic speech research. Audio recordings from external research databases (UltraSuite UPX/UXSSD/UXTD, ENNI, TORGO) remain the intellectual property of their respective creators and are used under their academic evaluation licenses.

> **Citation & Inquiries:** When utilizing the Aprixia platform or the multimodal HuBERT Layer 9 + openSMILE feature architecture in academic research, please cite the corresponding acoustic evaluation reports and ensure that all human participant data is handled in accordance with local IRB/ethics guidelines and HIPAA/GDPR health data privacy regulations.
