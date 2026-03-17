# Run: 20260316_181230

## 15-Epoch Training (BPM + Key Features)

First kfold at 15 epochs (`b9a3210`, Mar 16). Previous run (`76c8743`, `20260316_160726`) used the same encoding improvements but only 5 epochs. Also the first run to use the disk-based mel spectrogram cache.

| Metric | 5 epochs (`160726`) | 15 epochs (this run) | Delta |
|---|---|---|---|
| **micro F1** | 0.313 ± 0.019 | 0.336 ± 0.017 | **+7%** |
| **macro F1** | 0.230 ± 0.016 | 0.267 ± 0.026 | **+16%** |
| micro precision | 0.247 ± 0.051 | 0.272 ± 0.027 | +10% |
| **macro precision** | 0.226 ± 0.042 | 0.306 ± 0.025 | **+35%** |
| micro recall | 0.471 ± 0.086 | 0.444 ± 0.034 | −6% |
| macro recall | 0.365 ± 0.084 | 0.353 ± 0.051 | −3% |

F1 improved meaningfully (+7% micro, +16% macro), driven by a large precision gain (+35% macro). The model is making fewer false positives — it's getting more selective. Recall dipped slightly, which is the expected trade-off as precision rises.

The variance drop is equally important: recall std fell from 0.086 → 0.034 and precision std from 0.042 → 0.025. The model is now consistent across folds, suggesting the extra epochs gave it enough time to stabilize its use of BPM and key alongside audio and artist features.

**vs. the artist-only baseline** (`80c1d8d`, `20260315_231932`): micro F1 0.308 → 0.336 (+9%), macro F1 0.231 → 0.267 (+16%). BPM and key are now definitively contributing — the encoding improvements plus the longer training budget together beat the previous best.

**Next:** macro recall (0.353) is slightly below the artist-only baseline (0.348 was already modest) despite the F1 gain. Adding class-weighted loss (`pos_weight = neg_count / pos_count` per playlist in `BCEWithLogitsLoss`) is the most direct lever for lifting recall on underrepresented playlists without touching the architecture.
