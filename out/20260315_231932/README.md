# Run: 20260315_231932

## Artist Metadata Impact Analysis (kfold)

Comparison of kfold runs before (`c37ae3b`, Feb 28) and after (`80c1d8d`, Mar 15) adding artist metadata. Same methodology and splits — direct apples-to-apples comparison.

| Metric | Before | After | Delta |
|---|---|---|---|
| **micro F1** | 0.234 ± 0.043 | 0.308 ± 0.009 | **+32%** |
| **macro F1** | 0.182 ± 0.029 | 0.231 ± 0.017 | **+27%** |
| micro precision | 0.150 ± 0.037 | 0.232 ± 0.017 | **+55%** |
| **macro precision** | 0.129 ± 0.030 | 0.240 ± 0.025 | **+86%** |
| micro recall | 0.577 ± 0.045 | 0.465 ± 0.028 | -19% |
| macro recall | 0.488 ± 0.061 | 0.348 ± 0.024 | -29% |

F1 improved significantly (+27–32%), driven by a large precision gain (+55–86%). The model is predicting fewer false positives at the cost of missing more tracks overall.

Variance also dropped sharply — std dev on micro F1 went from 0.043 to 0.009, indicating much more consistent behavior across folds with artist features.

The recall trade-off is real: the recall guard (≥0.65) failed on more folds in this run. Dubstep, Headbanger, and Wubzstep missed the guard in folds 2 and 3.
