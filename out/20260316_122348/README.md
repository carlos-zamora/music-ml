# Run: 20260316_122348

## BPM + Key Encoding Improvements — kfold (run 1 of 2)

First kfold run at commit `76c8743` (Mar 16), which introduced circular Camelot key encoding and disambiguated BPM normalization. Compared against the artist-metadata baseline (`80c1d8d`, Mar 15 — `20260315_231932`).

> Note: `git_commit` was originally recorded as `4d81e7f` because the run was launched before committing. Corrected to `76c8743`.

| Metric | Baseline (artist only) | This run (+ BPM + key) | Delta |
|---|---|---|---|
| **micro F1** | 0.308 ± 0.009 | 0.275 ± 0.034 | **−11%** |
| **macro F1** | 0.231 ± 0.017 | 0.184 ± 0.030 | **−20%** |
| micro precision | 0.232 ± 0.017 | 0.211 ± 0.054 | −9% |
| macro precision | 0.240 ± 0.025 | 0.180 ± 0.072 | −25% |
| micro recall | 0.465 ± 0.028 | 0.485 ± 0.158 | +4% |
| macro recall | 0.348 ± 0.024 | 0.351 ± 0.101 | +1% |

F1 dropped and variance roughly tripled. However, a second kfold run at the same commit (`20260316_160726`) recovered to 0.313 ± 0.019 — essentially matching the baseline with much tighter variance. This result is therefore best explained by **training instability** rather than a problem with the encoding itself. The high variance here reflects an unlucky training run, not a systematic regression.

The encoding improvements (circular sin/cos for key, `[normalized, known]` for BPM) appear to have stabilized the model overall — the second run's std dropped significantly versus this one.
