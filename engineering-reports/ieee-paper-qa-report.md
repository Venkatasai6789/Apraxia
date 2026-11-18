# Aprixia IEEE Paper Evidence & Methodology Q&A Report

**Project:** Aprixia — Childhood Apraxia of Speech (CAS) Screening System  
**Date:** 2026-11-11  
**Workspace:** `c:\Users\pvenk\Downloads\Apraxia`  
**Verified Local Runtime:** Dell Inc. Inspiron 14 7430 2-in-1, 13th Gen Intel Core i7-1355U (10 physical cores / 12 logical processors, 1700 MHz max clock), 15.69 GB RAM, Windows 11 (`Windows-10-10.0.26200-SP0`), Python 3.11.9, PyTorch `2.15.0+cpu` (`cuda=False`), `transformers==5.22.1`, `opensmile==2.6.0`, `librosa==0.11.0`, `scikit-learn==1.9.1` (artifacts pickled under `1.6.1`).

> [!IMPORTANT]
> **Five Critical Methodological Findings to State Explicitly in the Paper:**
> 1. **No post-hoc probability calibration:** [`cas_classifier.joblib`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/cas_classifier.joblib) is an unwrapped `LogisticRegression(C=0.1, class_weight='balanced', penalty='l2', solver='lbfgs', max_iter=1000)` (no Platt scaling or isotonic regression). Because `class_weight='balanced'` was applied to a `1561:19` utterance split, probabilities reflect class-reweighted logistic outputs rather than true clinical prevalence priors.
> 2. **TD radar baseline cohort discrepancy:** Although commented `"Using Typically Developing (TD) children"`, [`07_executed_full.ipynb`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/07_executed_full.ipynb) Cell 5 computes [`td_mean_egemaps.joblib`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/td_mean_egemaps.joblib) and [`td_baselines.joblib`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/td_baselines.joblib) over `class_label == 0`, which averages all **6 non-CAS training speakers** (`2 UXTD + 2 UXSSD + 2 non-CAS UPX`), not strictly `UXTD`-only.
> 3. **Internal training preprocessing confound:** All **1,561 CAS utterances** (`03F, 07M, 08M, 10M`) were HTK `.lab` `CHILD`-only segmented with 20 ms boundary trims, whereas all **19 non-CAS control utterances** (`sample_master_features.csv`) were extracted from raw unsegmented `/core/` WAVs.
> 4. **Test folder mislabeling in `evaluate_all_datasets.py`:** `test_datasets/01_confirmed_apraxia_cas/` mixes 32 `UPX_03F_*_child.wav` CAS utterances (all scored `98.18%–100.0%`) with 8 raw `core_01F_*` and `core_04M_*` non-CAS recordings (`inconsistent phonological disorder` and `phonological delay`, all scored `0.0%–2.07%`), causing [`real_world_evaluation_summary.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/test_datasets/real_world_evaluation_summary.csv) to under-report UPX sensitivity as `80.0%` (`32/40`) instead of `100%` on the actual CAS clips (`32/32`).
> 5. **Final clinical threshold:** **`65.0%`** (not `60.0%`) is the final Elevated Risk cutoff across all production Python code and notebooks; `60.0%` in [`ml-integrity-report.md`](file:///c:/Users/pvenk/Downloads/Apraxia/engineering-reports/ml-integrity-report.md#L48-L51) is an internal documentation typo.

---

## A. Missing Code and Results

### Q (A.1): Can you push or attach the `ml-child-speech-disorder-classification/` folder? I need the training notebook or script, `ml_engine.py`, `app.py`, and `multi_dataset_generalization_report.csv`.

**A:** I cannot push to GitHub or any remote repository (workspace policy prohibits remote pushes, and no git remote is configured in `git remote -v`), and chat cannot directly upload file attachments. However, all requested files exist locally on disk and can be attached directly from these paths:

| Artifact | Purpose | Local File Link |
|---|---|---|
| Folder | Complete ML module, notebooks, artifacts, audit CSVs | [ml-child-speech-disorder-classification](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification) |
| Training & LOSO Notebook | Trains 856-dim fusion `LogisticRegression`, runs 10-speaker LOSO, exports `.joblib` files | [07_executed_full.ipynb](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/07_executed_full.ipynb) |
| Colab Multi-Dataset Evaluation Notebook | Stress-tests noise/codec + Hugging Face (`K-Univ/Pathological-child-voice`) + UXSSD/UXTD/UPX | [08_executed_final.ipynb](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/08_executed_final.ipynb) |
| Local Multi-Corpus Benchmark Script | Evaluates UPX, UXSSD, UXTD, TalkBank ENNI, and TORGO (`71` files) | [evaluate_all_datasets.py](file:///c:/Users/pvenk/Downloads/Apraxia/test_datasets/evaluate_all_datasets.py) |
| Shared ML Inference Engine | Singleton `CASMLEngine` used by both FastAPI and Streamlit | [ml_engine.py](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/ml_engine.py) |
| Original Streamlit App | Legacy/standalone Streamlit screening UI | [app.py](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/app.py) |
| Generalization Report CSV | Identical SHA-256 (`F190A4...`) in both locations | [multi_dataset_generalization_report.csv](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/multi_dataset_generalization_report.csv) / [root copy](file:///c:/Users/pvenk/Downloads/Apraxia/multi_dataset_generalization_report.csv) |

---

### Q (A.2): Do you have the per-speaker LOSO predictions (true label and predicted probability)? If not, can you share `cas_master_features.csv` so I can re-run LOSO and compute the metrics myself?

**A:** [`07_executed_full.ipynb`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/07_executed_full.ipynb) did not originally save per-utterance LOSO predictions to disk. To provide exact predictions without altering the model artifacts, I re-executed the exact 10-fold Leave-One-Speaker-Out loop from Cell 3 (`StandardScaler` + `LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)` on the 856-dim `eGeMAPS_88 + HuBERT_L9` vector across all 1,580 utterances) and verified exact numerical equivalence with the serialized full-fit production model (`cas_classifier.joblib` max weight diff `1.65e-14`, intercept diff `2.27e-13`, scaler mean diff `0.0`).

- **Generated per-utterance LOSO CSV (`1,580` rows: `global_speaker_id,true_label,pred_prob,pred_label_0.5`):** [loso_predictions.csv](file:///C:/Users/pvenk/.gemini/antigravity/brain/fbe73d30-9c92-444c-acd6-19a4f628533a/scratch/loso_predictions.csv)
- **Original training feature files on disk:**
  - [cas_master_features.csv](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/cas_master_features.csv) (`1,561` CAS rows x `3,941` columns, `~73.7 MB`)
  - [sample_master_features.csv](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/sample_master_features.csv) (`19` non-CAS control rows) + [pretrained_encoder_features.pt](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/pretrained_encoder_features.pt) (HuBERT L9 / WavLM L6 / Wav2Vec2 L12 embeddings for the 19 control utterances)

#### Reproduced Per-Speaker LOSO Summary (`HuBERT_L9 + eGeMAPS`, `N = 1,580`)

| Speaker | True Label | Utterances | Pred Pos (`>= 0.50`) | Mean Prob | Median Prob | Range `[Min, Max]` | Utterance Acc (`>= 0.50`) | Tier Breakdown (`Elev >=0.65` / `Bord [0.40,0.65)` / `Low <0.40`) |
|---|---|---:|---:|---:|---:|---|---:|---|
| `UPX_03F` | 1 (CAS) | 376 | 372 | 0.9841 | 0.9997 | `[0.1114, 1.0000]` | 0.9894 | 370 / 3 / 3 |
| `UPX_07M` | 1 (CAS) | 363 | 361 | 0.9876 | 0.9991 | `[0.2372, 1.0000]` | 0.9945 | 360 / 2 / 1 |
| `UPX_08M` | 1 (CAS) | 482 | 480 | 0.9910 | 0.9999 | `[0.0125, 1.0000]` | 0.9959 | 478 / 2 / 2 |
| `UPX_10M` | 1 (CAS) | 340 | 340 | 0.9967 | 0.9995 | `[0.6269, 1.0000]` | 1.0000 | 339 / 1 / 0 |
| `UPX_01F` | 0 (Non-CAS) | 5 | 1 (`0.5515`) | 0.1233 | 0.0265 | `[0.0008, 0.5515]` | 0.8000 | 0 / 1 / 4 |
| `UPX_04M` | 0 (Non-CAS) | 3 | 0 | 0.0160 | 0.0209 | `[0.0001, 0.0270]` | 1.0000 | 0 / 0 / 3 |
| `UXSSD_01M` | 0 (Non-CAS) | 2 | 0 | 0.0075 | 0.0075 | `[0.0000, 0.0150]` | 1.0000 | 0 / 0 / 2 |
| `UXSSD_03F` | 0 (Non-CAS) | 3 | 1 (`0.5673`) | 0.3970 | 0.4685 | `[0.1553, 0.5673]` | 0.6667 | 0 / 2 / 1 |
| `UXTD_01M` | 0 (Non-CAS) | 3 | 0 | 0.1488 | 0.0091 | `[0.0001, 0.4372]` | 1.0000 | 0 / 1 / 2 |
| `UXTD_03F` | 0 (Non-CAS) | 3 | 0 | 0.0748 | 0.0015 | `[0.0003, 0.2226]` | 1.0000 | 0 / 0 / 3 |

---

### Q (A.3): Besides the 94.48% balanced accuracy, do you have sensitivity, specificity, AUC, F1 or a confusion matrix?

**A:** Yes. Directly from [`07_executed_full.ipynb`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/07_executed_full.ipynb) Cell 3 (`Upgraded_Model_A: HuBERT_L9 + eGeMAPS`, `p >= 0.50` utterance threshold) and the exact reproduced `loso_predictions.csv`:

- **Utterance-level LOSO metrics at `p >= 0.50`:**
  - **Accuracy:** `0.9937` (`99.37%`)
  - **Balanced Accuracy:** `0.9448` (`94.48%`)
  - **Sensitivity (Recall / TPR):** `0.9949` (`99.49%`, `1553 / 1561`)
  - **Specificity (TNR):** `0.8947` (`89.47%`, `17 / 19`)
  - **Precision (PPV):** `0.9987` (`99.87%`, `1553 / 1555`) *(computed from exact LOSO reproduction; not printed in Cell 3)*
  - **F1 Score:** `0.9968`
  - **ROC-AUC:** `0.9986`
  - **Confusion Matrix `[[TN, FP], [FN, TP]]` (`p >= 0.50`):** **`[[17, 2], [8, 1553]]`**
    *(Both false positives fell in the `[0.40, 0.65)` borderline band: `UPX_01F` at `0.5515` and `UXSSD_03F` at `0.5673`.)*

- **Utterance-level LOSO metrics at the clinical thresholds (`0.40`, `0.50`, `0.65`):**

| Cutoff (`prob >=`) | TP | FP | TN | FN | Accuracy | Balanced Acc | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.40` (Borderline+) | 1555 | 4 | 15 | 6 | 0.9937 | 0.8928 | 0.9962 | 0.7895 | 0.9968 |
| `0.50` (Notebook LOSO) | 1553 | 2 | 17 | 8 | 0.9937 | 0.9448 | 0.9949 | 0.8947 | 0.9968 |
| `0.65` (Elevated CAS Risk) | 1547 | 0 | 19 | 14 | 0.9911 | 0.9955 | 0.9910 | 1.0000 | 0.9955 |

- **Speaker-level aggregation (`N = 10` speakers, using per-speaker mean predicted probability):** `TP=4, FP=0, TN=6, FN=0` (**Accuracy = `1.0000`, Balanced Accuracy = `1.0000`, Sensitivity = `1.0000`, Specificity = `1.0000`**, identical at cutoffs `0.40`, `0.50`, and `0.65`; **Speaker-level ROC-AUC = `1.0000`**; highest non-CAS mean is `UXSSD_03F` at `0.3970`, lowest CAS mean is `UPX_03F` at `0.9841`).

---

## B. Dataset

### Q (B.1): Which corpora were used for training, and which only for testing? The candidates are UXTD, UXSSD, UPX, ENNI and TORGO.

**A:**

| Corpus | Role in Final CAS System (`07` / `08` / `test_datasets`) | Details & Caveats |
|---|---|---|
| **UPX** (UltraSuite UltraPhonix) | **Training + In-distribution Testing** | **Training:** 4 CAS speakers (`03F, 07M, 08M, 10M` = `1,561` utt, class `1`) + 2 non-CAS SSD speakers (`01F, 04M` = `8` utt, class `0`). **Testing:** `100` randomly sampled `df_cas` rows (`08_executed_final.ipynb`), `30` clean/noisy/8kHz-bandlimited `UPX` CAS recordings (`08_executed_final.ipynb`), and `40` files in `test_datasets/01_confirmed_apraxia_cas/` *(same training speakers)*. |
| **UXSSD** (UltraSuite) | **Training + In-distribution Testing** | **Training:** 2 speakers (`01M, 03F` = `5` utt, class `0`). **Testing:** the same `5` files appear in both `08_executed_final.ipynb` (`core-uxssd-sample.zip`) and `test_datasets/02_non_apraxia_speech_sound_delay_uxssd/` (`0/5` flagged). |
| **UXTD** (UltraSuite) | **Training + In-distribution Testing** | **Training:** 2 speakers (`01M, 03F` = `6` utt, class `0`). **Testing:** the same `6` files appear in both `08_executed_final.ipynb` (`core-uxtd-sample.zip`) and `test_datasets/03_typically_developing_controls_uxtd/` (`0/6` flagged). |
| **ENNI** (TalkBank) | **Testing Only (OOD)** | `10` healthy child narrative files (`test_datasets/05_external_healthy_child_enni`, class `0`): `6/10` FP (`mean = 55.07%`, specificity `40.0%`). Never in training. |
| **TORGO** | **Testing Only (OOD)** | `5` adult dysarthria files (`F01_Session1_*`, label `1`: `3/5` flagged, `mean = 59.76%`) + `5` healthy adult control files (`FC01_Session1_*`, label `0`: `0/5` flagged, `mean = 0.05%`) in `test_datasets/06_*` and `07_*`. Never in training. |
| **`K-Univ/Pathological-child-voice`** (Hugging Face) | **Testing Only (OOD)** | `40` pediatric speech samples (`split="train[:40]"`) tested in `08_executed_final.ipynb`: `mean = 59.17%`, `60.0% >= 65%`. Never in training. |

> [!WARNING]
> **Two reporting traps in the UltraSuite test subsets:**
> 1. The `UPX`, `UXSSD`, and `UXTD` evaluation subsets in `08_executed_final.ipynb` and `test_datasets/` reuse speakers present in the 10-speaker training pool (only `ENNI`, `TORGO`, and Hugging Face `K-Univ` are unseen external corpora).
> 2. In [`dataset_test_manifest.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/test_datasets/dataset_test_manifest.csv), `test_datasets/01_confirmed_apraxia_cas/` accidentally places 8 non-CAS recordings (`core_01F_*` and `core_04M_*`) under `Confirmed CAS` (`ground_truth_label=1`). The classifier scored all 8 non-CAS clips `0.00%–2.07%` and all 32 actual `UPX_03F_*_child.wav` CAS clips `98.18%–100.0%`.

---

### Q (B.2): Where do the CAS labels come from? Which speakers are CAS-positive, and who diagnosed them?

**A:**
- **CAS-positive speakers (all 4):** **`UPX_03F`**, **`UPX_07M`**, **`UPX_08M`**, and **`UPX_10M`** (out of 20 speakers in [`UPX_info.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/UPX_info.csv), these are the only 4 with `SSD SUBTYPE = childhood apraxia of speech`).
- **Source & diagnostician:** From [`cas_diagnostic_evidence.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/phase3_cas_audit/cas_diagnostic_evidence.csv) and [`cas_participant_inventory.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/phase3_cas_audit/cas_participant_inventory.csv), labels come from official **UltraSuite UPX release metadata (*Eshky et al., Interspeech 2018*)**, assigned during **UltraPhonix biofeedback clinical trial suitability and intake assessment (Cleland et al., Queen Margaret University) by a qualified Speech and Language Therapist (SLT)**.
- **Individual diagnostic records:** **Don't know / not published** — raw SLT clinical case notes and specific standardized assessment scores (e.g., DEMSS/VMPAC) are confidential and omitted from the public release (`raw_clinical_notes_available = No`).

---

### Q (B.3): How many speakers and utterances are in each class (CAS vs. non-CAS), and what is the age range?

**A:**
- **Training Cohort (`N = 1,580` utterances across `10` speakers, combined age range `6.00–13.33` yrs):**
  - **CAS (Class `1`):** **4 speakers, 1,561 utterances**, ages **8y 11m – 13y 4m (`8.92–13.33` yrs)**:
    - `UPX_03F`: Female, `10y 11m` (`10.92`), `376` utterances (`38.00` min)
    - `UPX_07M`: Male, `8y 11m` (`8.92`), `363` utterances (`37.14` min)
    - `UPX_08M`: Male, `10y 2m` (`10.17`), `482` utterances (`54.31` min)
    - `UPX_10M`: Male, `13y 4m` (`13.33`), `340` utterances (`28.29` min)
  - **Non-CAS (Class `0`):** **6 speakers, 19 utterances**, ages **6y 0m – 11y 10m (`6.00–11.83` yrs)** (verified from [`UPX_info.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/UPX_info.csv), [`UXSSD_info.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/UXSSD_info.csv), [`UXTD_info.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/UXTD_info.csv)):
    - `UPX_01F`: Female, `8y 8m` (`8.67`), *inconsistent phonological disorder*, `5` utterances
    - `UPX_04M`: Male, `7y 2m` (`7.17`), *phonological delay*, `3` utterances
    - `UXSSD_01M`: Male, `6y 0m` (`6.00`), *non-CAS SSD*, `2` utterances
    - `UXSSD_03F`: Female, `8y 7m` (`8.58`), *non-CAS SSD*, `3` utterances
    - `UXTD_01M`: Male, `11y 10m` (`11.83`), *typically developing*, `3` utterances
    - `UXTD_03F`: Female, `10y 5m` (`10.42`), *typically developing*, `3` utterances
- **External Test-Only Cohorts:**
  - `ENNI`: `10` healthy child utterances (speaker count & child ages: **don't know / not in codebase**).
  - `TORGO`: `5` dysarthria utterances (`F01_Session1_*`, speaker `F01`) + `5` healthy control utterances (`FC01_Session1_*`, speaker `FC01`) (exact adult ages: **don't know / not in codebase**).
  - `K-Univ/Pathological-child-voice`: `40` utterances (`train[:40]`) (speaker count & ages: **don't know / not in codebase**).

---

### Q (B.4): Is the non-CAS class typically-developing children only, or does it also include children with speech-sound disorders or delay?

**A:** **Both typically developing (TD) children and children with non-CAS speech-sound disorders/delays.** Only 2 of the 6 non-CAS training speakers (6/19 utterances) are typically developing (`UXTD_01M`, `UXTD_03F`); the remaining 4 speakers (13/19 utterances) have non-CAS speech-sound disorders (`UXSSD_01M`, `UXSSD_03F`, `UPX_01F` *inconsistent phonological disorder*, `UPX_04M` *phonological delay*).

---

### Q (B.5): Were all classes recorded in the same setup? This decides how much space the corpus-confound limitation needs.

**A:** **Only partially within the UltraSuite training data, and NOT across external test sets.** Two distinct confounds should be highlighted:

1. **Same Hardware/Setting in Training, BUT Segmentation Confound Between CAS and Non-CAS:**
   - All 10 training children (`UPX`, `UXSSD`, `UXTD`) come from the same Queen Margaret University / Edinburgh **UltraSuite** clinical ultrasound-accompanied speech sessions (`22,050 Hz`, 16-bit `PCM_16`, mono WAV, resampled to `16 kHz`).
   - **However:**
     - All **1,561 CAS utterances** ([`cas_annotation_availability.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/phase3_cas_audit/cas_annotation_availability.csv)) were **HTK `.lab` `CHILD`-only segmented** (20 ms boundary trim, SLT speech and long silences removed, concatenated).
     - All **19 non-CAS training utterances** ([`sample_audio_manifest.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/sample_audio_manifest.csv), `UltraSuite_Sample_Download.ipynb` Cell 29) were extracted from **raw unsegmented `/core/` WAVs**, creating an acute preprocessing confound (`CHILD`-segmented vs raw `/core/`) on top of the `1561:19` utterance imbalance.
2. **Strong Out-of-Distribution / Corpus Confounds on External Evaluation Sets:**
   - **TalkBank ENNI** (`test_datasets/05_external_healthy_child_enni`): `24,000 Hz`, 32-bit `FLOAT`, mono WAV; Canadian child spontaneous narrative storytelling (different acoustic booth/mic, task, dialect). Yields **6/10 false positives** (`40.0%` specificity, `55.07%` mean CAS score).
   - **TORGO** (`test_datasets/06_*` & `07_*`): `16,000 Hz`, 16-bit `PCM_16`, mono WAV; adult dysarthric (`F01`, `3/5` flagged) vs healthy adult controls (`FC01`, `0/5` flagged), differing in age/vocal tract anatomy and recording rig.
   - **Hugging Face `K-Univ/Pathological-child-voice`**: External pediatric pathological cohort (`40` samples, `59.17%` mean score, `60.0% >= 65%`); microphone/recording protocol and exact clinical subtype distribution in `train[:40]` are **don't know / not documented in `08_executed_final.ipynb`**.

#### Cross-Dataset Generalization Results Summary (Observed in Codebase)

From [`multi_dataset_generalization_report.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/multi_dataset_generalization_report.csv) (`08_executed_final.ipynb`, Colab):

| Cohort | Samples | Mean Score (%) | Median (%) | Min (%) | Max (%) | Elevated CAS Rate (`>= 65%`) |
|---|---:|---:|---:|---:|---:|---:|
| CAS Clean Baseline (UPX) | 30 | 99.91 | 100.00 | 98.18 | 100.00 | 100.0% |
| CAS + 15dB Clinical Noise | 30 | 99.82 | 99.97 | 98.14 | 100.00 | 100.0% |
| CAS + 8kHz Phone Mic Bandwidth | 30 | 99.94 | 100.00 | 98.92 | 100.00 | 100.0% |
| Confirmed CAS (UltraSuite UPX sample) | 100 | 99.90 | 99.99 | 98.21 | 100.00 | 100.0% |
| External Child Benchmark (HF `K-Univ`) | 40 | 59.17 | 83.77 | 0.00 | 100.00 | 60.0% |
| Non-CAS Speech Delay (UXSSD) | 5 | 0.24 | 0.17 | 0.00 | 0.51 | 0.0% |
| Typically Developing (UXTD) | 6 | 0.10 | 0.06 | 0.01 | 0.24 | 0.0% |

From [`real_world_evaluation_summary.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/test_datasets/real_world_evaluation_summary.csv) and [`subgroup_aggregate_metrics.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/test_datasets/subgroup_aggregate_metrics.csv) (`evaluate_all_datasets.py`, `p >= 50.0%` binary cutoff):

| Cohort / Subgroup | Format (Native) | N | Mean Score (%) | Sensitivity (%) | Specificity (%) | Balanced Acc (%) | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Confirmed CAS (UPX folder as labeled) | `22.05 kHz PCM_16` | 40 | 80.00 | 80.00 (`32/40`) | — | — | Contains `32` true `03F` CAS (`32/32` TP, `98.18–100%`) + `8` misfiled non-CAS `01F/04M` (`0–2.07%`) |
| Speech Sound Delay (UXSSD) | `22.05 kHz PCM_16` | 5 | 0.24 | — | 100.00 (`5/5`) | — | Training speakers (`01M, 03F`) |
| Typically Developing (UXTD) | `22.05 kHz PCM_16` | 6 | 0.10 | — | 100.00 (`6/6`) | — | Training speakers (`01M, 03F`) |
| External Healthy Child (TalkBank ENNI) | `24.00 kHz FLOAT` | 10 | 55.07 | — | 40.00 (`4/10`) | — | Unseen OOD narrative storytelling (`6` FP) |
| External Motor Disorder (TORGO Dysarthria) | `16.00 kHz PCM_16` | 5 | 59.76 | 60.00 (`3/5`) | — | — | Unseen adult dysarthria speaker `F01` |
| External Matched Controls (TORGO Healthy) | `16.00 kHz PCM_16` | 5 | 0.05 | — | 100.00 (`5/5`) | — | Unseen healthy adult speaker `FC01` |
| **Aggregate External OOD (TORGO + ENNI)** | Mixed | **20** | — | **60.00** | **60.00** | **60.00** | F1 `42.86%`, ROC-AUC `73.33%` |

---

## C. Method Choices

### Q (C.1): Was HuBERT layer 9 chosen by a layer sweep? If so, do you have the per-layer results?

**A:** It was chosen from a **4-layer representation-geometry & speaker-bias probe (`Layers 3, 6, 9, 12`)**, **not** a full 12-layer sweep and **not** by per-layer LOSO classification accuracy. Layers `1, 2, 4, 5, 7, 8, 10, 11` were **not evaluated** (**don't know**).

1. **Stage 1 — 4-Encoder x 4-Layer Geometry Sweep on 19-Utterance Sample** ([`04_pretrained_encoder_comparison.ipynb`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/04_pretrained_encoder_comparison.ipynb) Cell 6, mean-pooled 768-dim embeddings):

| Encoder | Layer | Feature Variance | Intra-Speaker Cosine Sim | Inter-Speaker Cosine Sim | Speaker Gap (`Intra - Inter`) |
|---|---|---:|---:|---:|---:|
| `wav2vec2_base` (`facebook/wav2vec2-base-960h`) | Layer 3 | 0.002981 | 0.819979 | 0.778358 | 0.041620 |
| `wav2vec2_base` | Layer 6 | 0.002567 | 0.830028 | 0.805824 | 0.024204 |
| `wav2vec2_base` | Layer 9 | 0.003243 | 0.904976 | 0.896250 | 0.008726 |
| `wav2vec2_base` | Layer 12 | 0.002188 | 0.965321 | 0.962320 | 0.003000 |
| **`hubert_base` (`facebook/hubert-base-ls960`)** | **Layer 3** | **0.002134** | **0.857250** | **0.833703** | **0.023547** |
| **`hubert_base`** | **Layer 6** | **0.002044** | **0.860561** | **0.838207** | **0.022354** |
| **`hubert_base`** | **Layer 9** | **0.002682** | **0.834067** | **0.812097** | **0.021971** *(lowest HuBERT gap)* |
| **`hubert_base`** | **Layer 12** | **0.005873** | **0.828958** | **0.801322** | **0.027636** |
| `wavlm_base_plus` (`microsoft/wavlm-base-plus`) | Layer 3 | 0.001866 | 0.869197 | 0.846095 | 0.023101 |
| `wavlm_base_plus` | Layer 6 | 0.002443 | 0.830163 | 0.799654 | 0.030509 |
| `wavlm_base_plus` | Layer 9 | 0.002851 | 0.835817 | 0.806403 | 0.029414 |
| `wavlm_base_plus` | Layer 12 | 0.001359 | 0.838830 | 0.805663 | 0.033167 |
| `data2vec_audio` (`facebook/data2vec-audio-base-960h`) | Layer 3 | 0.001394 | 0.876350 | 0.855958 | 0.020392 |
| `data2vec_audio` | Layer 6 | 0.002043 | 0.796280 | 0.777688 | 0.018592 |
| `data2vec_audio` | Layer 9 | 0.002470 | 0.905013 | 0.896609 | 0.008404 |
| `data2vec_audio` | Layer 12 | 0.004996 | 0.940694 | 0.930358 | 0.010336 |

2. **Stage 2 — Geometry Audit on the Full `1,561`-Utterance CAS Cohort** ([`cas_encoder_geometry_analysis.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/cas_encoder_geometry_analysis.csv)):

| Encoder & Layer | Mean Feature Var | Effective Rank (of 768) | PC1 Var (%) | Top-5 PCs Var (%) | PCs for 90% Var | Mean Cosine Sim | Speaker Identity Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`HuBERT_L9`** | 0.00554 | **74.32** | 14.16% | 39.92% | 146 | 0.712 | 0.0365 |
| `HuBERT_L12` | 0.01072 | 40.53 | 17.99% | 48.81% | 65 | 0.704 | 0.0480 |
| `WavLM_L6` | 0.00388 | 53.69 | 14.21% | 44.15% | 96 | 0.677 | 0.0501 |
| `Wav2Vec2_L6` | 0.00441 | 75.82 | 12.94% | 38.63% | 145 | 0.662 | 0.0398 |
| `Wav2Vec2_L12` | 0.00203 | 21.20 | 25.76% | 56.76% | 22 | 0.924 | 0.0050 |

---

### Q (C.2): Did you compare against eGeMAPS only, HuBERT only, or other classifiers (SVM, random forest, MLP)? Any numbers?

**A:**
- **Evaluated on the `1,580`-utterance CAS cohort in [`07_executed_full.ipynb`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/07_executed_full.ipynb) Cell 3** (all 4 used identical `LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)` at `p >= 0.50`, despite a stale comment `# Linear Support Vector Classifier`):

| Model Config (`07_executed_full.ipynb`) | Dims | Accuracy | Balanced Acc | Sensitivity | Specificity | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Repo_Baseline (Wav2Vec2_L12)` | 768 | 0.9873 | 0.8896 | 0.9898 | 0.7895 | 0.9936 | 0.9926 |
| **`Acoustic_Baseline (eGeMAPS_88)`** | 88 | **0.9696** | **0.8287** | **0.9731** | **0.6842** | **0.9844** | **0.9773** |
| **`Upgraded_Model_A (HuBERT_L9 + eGeMAPS)`** *(production)* | 856 | **0.9937** | **0.9448** | **0.9949** | **0.8947** | **0.9968** | **0.9986** |
| `Upgraded_Model_B (WavLM_L6 + eGeMAPS)` | 856 | 0.9937 | 0.9448 | 0.9949 | 0.8947 | 0.9968 | 0.9979 |

- **HuBERT-L9 only (`768` dims without eGeMAPS):** **Not tested** in any notebook (**don't know**).
- **Other classifiers (SVM, Random Forest, MLP) on CAS:** **Not tested** on the CAS cohort (**don't know**). `SVC` and `RandomForestClassifier` were imported in `07_executed_full.ipynb` Cell 1 but never called. *(An earlier pre-CAS notebook [`04-final_code.ipynb`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/04-final_code.ipynb) implemented RBF-SVM, SMOTE+SVM, CatBoost, MLP, and Wav2Vec2_L12+SVM on the broader SSD-vs-TD classification task; its cell outputs were stripped on disk, and [`README.md`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/README.md#L122-L128) only reports that `Wav2Vec 2.0 + SVM` achieved 94% accuracy, 88% recall, and 94% UAR on 16 unseen SSD/TD children.)*

---

### Q (C.3): How were the 65% and 40% thresholds chosen? Also confirm 65% is final, since one report says 60%.

**A:**
- **Confirmed final thresholds:** **`>= 65.0%`** (*Elevated CAS Risk*), **`40.0%–64.9%`** (*Borderline / Monitoring Needed*), and **`< 40.0%`** (*Low Risk / Typical*) are final across all code: [`ml_engine.py`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/ml_engine.py#L296-L305), [`app.py`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/app.py#L443-L457), [`assessment_service.py`](file:///c:/Users/pvenk/Downloads/Apraxia/backend/services/assessment_service.py#L499-L505), `07_executed_full.ipynb` Cell 4, and [`evaluate_all_datasets.py`](file:///c:/Users/pvenk/Downloads/Apraxia/test_datasets/evaluate_all_datasets.py#L168-L173). The `60.0%` value in [`ml-integrity-report.md`](file:///c:/Users/pvenk/Downloads/Apraxia/engineering-reports/ml-integrity-report.md#L48-L51) is an internal documentation typo.
- **How they were chosen:** **Don't know / heuristic design choice** — no ROC operating-point optimization, Youden's $J$ statistic, or validation sweep exists in any notebook or script. They were hardcoded directly in `07_executed_full.ipynb` Cell 4 to create an intermediate `"BORDERLINE / MONITOR"` band. *(Empirically on `loso_predictions.csv`, `>= 0.65` yields zero non-CAS false positives (`19/19` TN, `1547/1561` TP), whereas `evaluate_all_datasets.py` still uses `prob >= 50.0` for binary confusion metrics while using `>= 65.0` for tier labels.)*

---

### Q (C.4): Was any probability calibration applied (Platt scaling or isotonic regression), or is it plain logistic regression?

**A:** **Plain L2 logistic regression — NO post-hoc calibration was applied.** Direct inspection of [`cas_classifier.joblib`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/cas_classifier.joblib) confirms an unwrapped `sklearn.linear_model._logistic.LogisticRegression(C=0.1, class_weight='balanced', penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)`. Neither `CalibratedClassifierCV` nor Platt/isotonic calibration appears anywhere in the repository; `"Calibrated"` in code comments and UI strings refers informally to `LogisticRegression.predict_proba()`. Because `class_weight='balanced'` was fitted on `1561:19`, predicted scores reflect reweighted log-odds rather than calibrated clinical priors.

---

### Q (C.5): Which eGeMAPS features feed the three radar domains, and which cohort were the typically-developing baselines computed from?

**A:**
- **Exact Radar Domain Feature Mapping** ([`clean_dimensions.joblib`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/clean_dimensions.joblib), [`07_executed_full.ipynb`](file:///c:/Users/pvenk/Downloads/Apraxia/ml-child-speech-disorder-classification/07_executed_full.ipynb) Cell 5 — **3 domains, 6 sub-axes, 11 `eGeMAPSv02` functionals total**, scored via mean $|(x_f - \mu_f)/\sigma_f|$):

| Radar Domain | Sub-Axis | Exact `eGeMAPSv02` Feature Columns | Stored Reference Baseline (`td_baselines.joblib`) |
|---|---|---|---|
| **Pitch / Prosody** | `Pitch Level` | `F0semitoneFrom27.5Hz_sma3nz_amean`, `F0semitoneFrom27.5Hz_sma3nz_percentile50.0` | `0.82897` |
| **Pitch / Prosody** | `Pitch Variability` | `F0semitoneFrom27.5Hz_sma3nz_stddevNorm`, `F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2` | `0.81337` |
| **Voice Quality** | `Jitter / Shimmer` | `jitterLocal_sma3nz_amean`, `shimmerLocaldB_sma3nz_amean` | `0.81609` |
| **Voice Quality** | `Harmonics Clarity` | `HNRdBACF_sma3nz_amean` | `0.71554` |
| **Vowel Formants** | `Formant 1` | `F1frequency_sma3nz_amean`, `F1bandwidth_sma3nz_amean` | `0.76258` |
| **Vowel Formants** | `Formant 2` | `F2frequency_sma3nz_amean`, `F2bandwidth_sma3nz_amean` | `0.80476` |

- **Cohort used to compute the TD baselines (`td_mean_egemaps.joblib`, `td_std_egemaps.joblib`, `td_baselines.joblib`):**
  - Computed in `07_executed_full.ipynb` Cell 5 by filtering `df_unified['class_label'] == 0` and averaging per-speaker means across **all 6 non-CAS training speakers (19 utterances)**: **`UXTD_01M`, `UXTD_03F`, `UXSSD_01M`, `UXSSD_03F`, `UPX_01F`, and `UPX_04M`**.
  - Despite the code comment `"Using Typically Developing (TD) children"`, this reference **includes 4 non-CAS SSD/delay speakers alongside the 2 UXTD speakers** (empirically verified: `td_mean_egemaps.joblib` max diff vs all 6 speakers = `0.0`, vs strictly 2 UXTD speakers = `55.17`).

---

### Q (C.6): Were the latency figures (417 ms per utterance) actually measured? If so, on what CPU?

**A:** **Yes, `417 ms` was measured by a unit-test benchmark on this laptop CPU, though the sub-stage breakdown table in `performance-report.md` was estimated.**

- **Verified CPU & System Hardware:** **13th Gen Intel(R) Core(TM) i7-1355U** (10 cores / 12 logical processors, `1700 MHz` max clock), Dell Inc. Inspiron 14 7430 2-in-1, `15.69 GB` RAM, Windows 11, CPU-only PyTorch `2.15.0+cpu`.
- **Origin of `417 ms`:** An earlier execution of `test_consecutive_inference_latency_benchmark` in [`test_security_performance.py`](file:///c:/Users/pvenk/Downloads/Apraxia/backend/tests/test_security_performance.py#L362-L393) (`N=4` consecutive warm runs of `CASMLEngine.extract_audio_features_and_predict()` on the **3.56 s** sample `sample_cas_child_03F.wav` @ 16 kHz) reported `Mean=0.417s, Min=0.392s, Max=0.436s, Std=0.019s`.
- **Caveat on sub-stage numbers in [`performance-report.md`](file:///c:/Users/pvenk/Downloads/Apraxia/engineering-reports/performance-report.md#L6-L13):** That report attributes `28 + 74 + 272 + 4 + 39 = 417 ms` across 5 stages including `39 ms` for the 64-band Mel spectrogram, even though `test_consecutive_inference_latency_benchmark` does **not** invoke `generate_audio_analysis_payload()` (spectrogram + F0 tracking). Those per-stage values were therefore not produced by that benchmark.
- **Freshly Measured Benchmarks on the Same `i7-1355U` CPU (`sample_cas_child_03F.wav`, 3.560 s, warm singleton):**

| Benchmark / Stage | Sample Size | Mean | Min | Max | Std | Scope Notes |
|---|---|---:|---:|---:|---:|---|
| `test_consecutive_inference_latency_benchmark` (re-run) | `N=4` | **508 ms** | 493 ms | 515 ms | 9 ms | `extract_audio_features_and_predict()` only (warm) |
| Audio load (`librosa.load`, BytesIO 16 kHz) | `N=5` | **2.4 ms** | 1.4 ms | 4.0 ms | 0.9 ms | Decoding in-memory WAV |
| openSMILE `eGeMAPSv02` (88 dims) | `N=5` | **103.7 ms** | 81.8 ms | 155.2 ms | 26.9 ms | Functional extraction |
| HuBERT Layer 9 forward pass (768 dims) | `N=5` | **369.8 ms** | 338.4 ms | 412.9 ms | 28.8 ms | CPU transformer inference |
| Feature alignment + `StandardScaler` + LR | `N=5` | **6.6 ms** | 3.6 ms | 7.5 ms | 1.5 ms | 856-dim vector scoring |
| Full `extract_audio_features_and_predict()` | `N=5` | **579.6 ms** | 523.2 ms | 621.3 ms | 38.7 ms | Includes temp WAV write + thread lock |
| UI `generate_audio_analysis_payload()` | `N=5` | **1,232.5 ms** | 894.8 ms | 1,553.7 ms | 242.1 ms | Waveform + 64-band Mel spectrogram + `librosa.yin` F0 track |
| Batch benchmark (`1.30–12.07 s` clips, [`dataset_test_manifest.csv`](file:///c:/Users/pvenk/Downloads/Apraxia/test_datasets/dataset_test_manifest.csv)) | `N=71` | — | **617 ms** | **3,557 ms** | — | Uncached-per-file loop (excluding model init) |

---

## Quick Reference: Items Verified as Absent (`Don't Know / Not in Codebase`)

- SLT assessor identities, intake protocols, and individual diagnostic test sub-scores for UltraSuite UPX (`03F, 07M, 08M, 10M`).
- Participant speaker counts, age distributions, and recording hardware metadata for TalkBank `ENNI`, `TORGO` (`F01`/`FC01` ages), and Hugging Face `K-Univ/Pathological-child-voice`.
- Encoder representations / LOSO metrics for HuBERT layers `1, 2, 4, 5, 7, 8, 10, 11`.
- Classification accuracy comparison for `HuBERT_L9`-only (`768d`) or non-linear/ensemble classifiers (`SVM`, `Random Forest`, `MLP`, `CatBoost`) on the `1,580`-utterance CAS cohort.
- Formal ROC/Youden threshold selection procedure for `65.0%` and `40.0%`.
