# Run: 20260316_122348

## BPM + Musical Key as Model Inputs (kfold)

Comparison against the previous kfold (`80c1d8d`, Mar 15 — `20260315_231932`) which introduced artist metadata. This run (`4d81e7f`, Mar 16) adds BPM and musical key as additional inputs to the model.

| Metric | Before (artist only) | After (+ BPM + key) | Delta |
|---|---|---|---|
| **micro F1** | 0.308 ± 0.009 | 0.275 ± 0.034 | **-11%** |
| **macro F1** | 0.231 ± 0.017 | 0.184 ± 0.030 | **-20%** |
| micro precision | 0.232 ± 0.017 | 0.211 ± 0.054 | -9% |
| macro precision | 0.240 ± 0.025 | 0.180 ± 0.072 | -25% |
| micro recall | 0.465 ± 0.028 | 0.485 ± 0.158 | +4% |
| macro recall | 0.348 ± 0.024 | 0.351 ± 0.101 | +1% |

Adding BPM and musical key caused a regression across all F1 metrics (-11% micro, -20% macro). Recall is essentially unchanged.

The most telling signal is the **variance explosion** — std roughly tripled on precision and quintupled on recall. Fold 3 in particular collapsed into a "predict everything" pattern (micro recall 0.71, micro precision 0.14), dragging macro F1 down sharply while the other two folds were more calibrated but still below the previous baseline.

The trained single-split model (same commit, `20260316_115848_single`) shows the opposite failure mode: very high precision (0.61) and very low recall (0.15), suggesting the model is struggling to balance the two after the new inputs are added.

Likely causes to investigate:
- **BPM encoding** — raw float values span a wide range and may dominate or confuse learning without normalization.
- **Key encoding** — if key is passed as an integer (0–11), the model may interpret it as ordinal rather than categorical; one-hot encoding would be safer.
- **5-epoch budget** — the new features may simply need more training time to be leveraged correctly.
