# Aprixia Bug Register & Defect Resolution Log

## Defect Severity Classification
- **P0**: Data corruption, incorrect ML prediction, security vulnerability.
- **P1**: Broken clinical assessment flow, missing API endpoint, unhandled crash.
- **P2**: Visualization defect, synchronization misalignment, non-critical UX issue.
- **P3**: Minor cosmetic or styling inconsistency.

## Defect Resolution Matrix

| Bug ID | Severity | Module | Description | Root Cause | Resolution & Verification | Status |
|---|:---:|---|---|---|---|:---:|
| **BUG-001** | **P0** | ML Pipeline | Model re-initialization on every upload request | Lacked in-process singleton; transformer weights reloaded repeatedly | Implemented thread-safe `CASMLEngine.get_instance()` with persistent weights | **RESOLVED** |
| **BUG-002** | **P0** | Backend | Missing audio duration gate | Audio $< 200$ ms caused openSMILE matrix failure | Added validation in `audio_service.py` rejecting files $< 0.20$s with HTTP 400 | **RESOLVED** |
| **BUG-003** | **P1** | Frontend | Mock fallback stores in production workflows | `assessmentService.ts` used in-memory arrays when API failed | Fully purged mock stores; wired all calls to live FastAPI backend | **RESOLVED** |
| **BUG-004** | **P1** | Backend | Missing dashboard aggregation endpoints | Frontend dashboard had no backend statistics API | Implemented `/api/v1/dashboard/stats` and `/overview` with live SQL counts | **RESOLVED** |
| **BUG-005** | **P2** | Audio Lab | Waveform and spectrogram zoom desynchronization | Plotly relayout handler did not propagate `xaxis2.range` bounds | Updated `SynchronizedAudioLaboratoryChart.tsx` to link subplot axes dynamically | **RESOLVED** |
| **BUG-006** | **P2** | Backend | Potential unbounded waveform payload size | Raw audio arrays caused megabyte-sized JSON responses | Implemented peak downsampling strictly bounding waveforms to $\le 2000$ points | **RESOLVED** |
| **BUG-007** | **P2** | Audio Lab | Artificial 3-slice mathematical division in audio display | Legacy UI code divided recording duration arbitrarily | Replaced with real audio sample rendering and genuine openSMILE features | **RESOLVED** |
| **BUG-008** | **P3** | Frontend | Static insight percentages | `78%` / `12%` hardcoded in `AssessmentInsights.tsx` | Replaced with dynamic completion/risk calculations and honest empty states | **RESOLVED** |

## Remaining Bugs
- **Active P0 Bugs**: 0
- **Active P1 Bugs**: 0
- **Active P2 Bugs**: 0
- **Active P3 Bugs**: 0

**All identified defects resolved and verified by automated tests.**
