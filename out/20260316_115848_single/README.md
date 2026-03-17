# Run: 20260316_115848_single

## BPM + Key Encoding Improvements — single (run 1 of 2)

First single-split run at commit `76c8743` (Mar 16), which introduced circular Camelot key encoding and disambiguated BPM normalization. Compared against the artist-metadata baseline (`80c1d8d`, Mar 15 — `20260315_141827_single`).

> Note: `git_commit` was originally recorded as `4d81e7f` because the run was launched before committing. Corrected to `76c8743`.

| Metric | Baseline (artist only) | This run (+ BPM + key) | Delta |
|---|---|---|---|
| **test micro F1** | 0.188 | 0.240 | **+28%** |
| **test macro F1** | 0.158 | 0.187 | **+18%** |
| test micro precision | 0.571 | 0.611 | +7% |
| test macro precision | 0.294 | 0.388 | +32% |
| test micro recall | 0.113 | 0.149 | +32% |
| test macro recall | 0.118 | 0.158 | +34% |
| test loss | 0.227 | 0.244 | +8% |

Single-split shows improvement, but a second single run at the same commit (`20260316_160211_single`) scored lower (micro F1 0.200), confirming that single-split results have high variance and can flatter or penalize randomly. The kfold run at the same commit is the more reliable signal.
