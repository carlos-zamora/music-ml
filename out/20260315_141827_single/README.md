# Run: 20260315_141827_single

## Marker-Based Partition Sampling

Comparison against the previous single-split run (`8519e5b`, Mar 15) which used equidistant partitions. This run (`80c1d8d`, Mar 15) replaces equidistant partition offsets with manually annotated rekordbox position markers when a track has enough markers (≥ `--parts` count), falling back to equidistant otherwise. 1,232 of 1,234 tracks had markers.

| Metric | Before (equidistant) | After (markers) | Delta |
|---|---|---|---|
| **val micro F1** | 0.252 | 0.271 | **+7%** |
| **val macro F1** | 0.164 | 0.177 | **+8%** |
| val micro precision | 0.619 | 0.660 | +7% |
| val macro precision | 0.302 | 0.347 | +15% |
| val micro recall | 0.153 | 0.171 | +12% |
| val macro recall | 0.109 | 0.124 | +14% |
| val loss | 0.217 | 0.220 | +1% |
| **test micro F1** | 0.210 | 0.188 | -11% |
| **test macro F1** | 0.092 | 0.158 | **+72%** |
| test micro precision | 0.519 | 0.571 | +10% |
| test macro precision | 0.222 | 0.294 | +32% |
| test micro recall | 0.131 | 0.113 | -14% |
| test macro recall | 0.067 | 0.118 | +76% |
| test loss | 0.243 | 0.227 | **-7%** |

The most striking result is test macro F1 jumping from 0.092 to 0.158 (+72%) and macro recall nearly doubling. This reflects better per-playlist consistency — the model is no longer disproportionately influenced by whatever arbitrary 10-second window equidistant sampling happened to land in. Using annotated marker positions (which mark structural boundaries: drops, builds, breakdowns) gives the model a more representative view of each track's character.

Test micro F1 declined slightly (0.210 → 0.188), which is a known trade-off: gains on rare/harder playlists tend to come at the cost of some accuracy on dominant ones, pulling the pooled micro metric down while the per-playlist macro metric rises.

Single-split runs have high variance, so this result should be validated with a kfold run before drawing strong conclusions.
