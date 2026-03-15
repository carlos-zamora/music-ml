# Run: 20260315_102628_single

## Artist Metadata Impact Analysis

Comparison of the two most recent single-split runs — before (`bad400d`, Mar 14) and after (`8519e5b`, Mar 15) adding artist metadata.

| Metric | Before | After | Delta |
|---|---|---|---|
| **micro F1** | 0.210 | 0.210 | ≈ flat |
| **macro F1** | 0.036 | 0.092 | **+155%** |
| micro precision | 0.452 | 0.519 | +15% |
| **macro precision** | 0.076 | 0.222 | **+192%** |
| micro recall | 0.137 | 0.131 | -4% |
| macro recall | 0.047 | 0.067 | +43% |
| loss | 0.250 | 0.243 | -3% |

Artist metadata had a clear and significant effect on **per-playlist discrimination**. Macro F1 nearly tripled and macro precision nearly tripled, while micro F1 stayed flat. This means the model got much better at targeting the *right* playlists rather than spraying the same predictions at everything — previously, playlists shared most of their false positives because the model was essentially blind to artist identity.

The slight recall drop (micro: 0.137 → 0.131) is the expected trade-off: a more precise model misses a few more tracks it would have caught with a looser trigger.
