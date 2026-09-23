# CallConnect Speech — Top 3 Dataset Insights
**Author:** Data Analyst (DA)  
**Dataset:** RAVDESS Speech Audio (1,440 recordings)

---

### Insight 1: Perfect Class & Speaker Symmetry with One Intentional Asymmetry
- **Observation:** 7 out of 8 emotion classes contain exactly 192 samples (96 normal + 96 strong intensity) across 24 actors. **Neutral** contains exactly 96 samples because it has no 'strong' intensity variant.
- **Data Science Implication:** Macro-averaged metrics (Macro F1, Macro Recall) must be used as primary evaluation criteria rather than raw accuracy to avoid giving undue weight to majority classes.

### Insight 2: Strong Acoustic Energy & Pitch Separation Between High/Low Arousal Pairs
- **Observation:** High-arousal emotions (Angry, Happy, Fearful, Surprised) exhibit markedly higher RMS energy (>0.035) and elevated spectral centroids (>1800 Hz) compared to low-arousal states (Calm, Sad, Neutral with RMS <0.018).
- **Data Science Implication:** Combining spectral dynamics (Spectral Centroid, Bandwidth, Contrast) with temporal energy features (RMS, ZCR) and 13 MFCC delta coefficients will provide clear hyperplanes separating high-arousal from low-arousal emotions.

### Insight 3: Speaker-Dependent Variation Exceeds Emotion-Specific Variance
- **Observation:** Speaker fundamental pitch and vocal tract length create significant actor-level clustering in feature space. 
- **Data Science Implication:** Standard K-Fold CV would cause severe optimistic bias and data leakage. **GroupKFold (groups=actor_id)** is mandatory so that models are evaluated purely on unseen speaker vocal profiles.
