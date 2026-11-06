# Aprixia Machine Learning Pipeline & Architectural Integrity Report

## 1. Preserved Machine Learning Architecture
The speech processing pipeline maintains strict fidelity to the original pre-trained research artifacts without any re-training, hyperparameter modification, or parameter pruning.

### Feature Extraction Stack
1. **Raw Audio Normalization**:
   - Resampled to exactly 16,000 Hz, single channel (mono).
   - Peak amplitude normalized.
2. **openSMILE eGeMAPSv02 Functional Set**:
   - Feature set: `eGeMAPSv02` (extended Geneva Minimalistic Acoustic Parameter Set).
   - Feature level: `Functionals` (summary statistics across utterance).
   - Extracted features: exactly 88 acoustic descriptors (frequency, energy, spectral, and temporal dynamics).
3. **HuBERT Self-Supervised Speech Representations**:
   - Model: `facebook/hubert-base-ls960`.
   - Feature Layer: **Hidden Layer 9** (extracted via `output_hidden_states=True`, index `[9]`).
   - Mean temporal pooling across frames producing a 768-dimensional embedding.
4. **Multimodal Feature Concat & Ordering**:
   - 88 eGeMAPS functionals + 768 HuBERT embeddings = **856 total dimensions**.
   - Aligned strictly against `cas_feature_cols.joblib` to guarantee column sequence preservation.
5. **StandardScaler & Classifier Inference**:
   - Transformed via pre-trained `cas_scaler.joblib`.
   - Ingested by `cas_classifier.joblib`:
     - Estimator: `LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, penalty='l2')`.
     - Output probability: $\text{Prob}(\text{Class} = 1) \times 100.0$.

## 2. In-Process Singleton Design (`CASMLEngine`)
To avoid recurrent neural network initialization overhead on incoming requests:
- `CASMLEngine` is implemented as a thread-safe singleton (`threading.Lock()`).
- Transformers tokenizer, HuBERT feature extractor, weights, and openSMILE SMILE instances are held persistently in memory.
- `INFERENCE_LOCK` guarantees thread safety across concurrent FastAPI background tasks.

## 3. Acoustic Radar Profile Calculation
The radar chart visualizes relative deviation from a Typically Developing (TD) pediatric baseline:
- Baseline statistics: `td_baselines.joblib` (derived from typically developing control cohorts).
- Dimensions: 3 clinical axes mapped into 6 radar spokes:
  1. Pitch / Prosody (F0 variation, pitch dynamics)
  2. Voice Quality (HNR, jitter, shimmer)
  3. Vowel Formants (F1, F2 formant articulation clarity)
- Normalization: Z-score transformed relative to TD mean and standard deviation.

## 4. Multi-Utterance Aggregation Rules
When an assessment contains multiple speech recordings, scores are aggregated according to established clinical criteria:
- **Mean Score**: $\bar{s} = \frac{1}{N} \sum_{i=1}^N s_i$
- **Median Score**: $\text{median}(s)$
- **Token Inconsistency**: Standard deviation $\sigma_s = \sqrt{\frac{1}{N} \sum (s_i - \bar{s})^2}$, quantifying characteristic CAS trial-to-trial variability.
- **Exclusion of Failed Recordings**: Unusable recordings ($< 200$ ms or corrupt) are segregated into error states and omitted from score averaging.
- **Classification Thresholds**:
  - $\text{Score} \ge 60.0\%$: **Elevated CAS Risk**
  - $40.0\% \le \text{Score} < 60.0\%$: **Borderline / Inconclusive**
  - $\text{Score} < 40.0\%$: **Low Risk / Typical**
