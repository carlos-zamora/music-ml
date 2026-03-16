# Run: 20260316_115848_single

## BPM + Musical Key as Model Inputs (single)

Comparison against the previous single-split run (`80c1d8d`, Mar 15 — `20260315_141827_single`) which introduced marker-based partition sampling. This run (`4d81e7f`, Mar 16) adds BPM and musical key as additional inputs to the model.

| Metric | Before (markers baseline) | After (+ BPM + key) | Delta |
|---|---|---|---|
| **test micro F1** | 0.188 | 0.240 | **+28%** |
| **test macro F1** | 0.158 | 0.187 | **+18%** |
| test micro precision | 0.571 | 0.611 | +7% |
| test macro precision | 0.294 | 0.388 | +32% |
| test micro recall | 0.113 | 0.149 | +32% |
| test macro recall | 0.118 | 0.158 | +34% |
| test loss | 0.227 | 0.244 | +8% |

On this single split, test F1 improved meaningfully (+18–28%). However, the kfold run from the same commit (`20260316_122348`) tells a different story: F1 dropped 11–20% and variance tripled, which is a much more reliable signal given the instability of single-split results.

The high precision (0.61) paired with low recall (0.15) in this run suggests the model is being overly conservative — likely an artifact of this particular train/test split and the recall-guarded threshold tuning. The kfold's fold 3 showed the mirror pathology: very aggressive predictions with recall ~0.71 but precision collapsing to ~0.14.

**Conclusion:** The single-split result flatters the change. The kfold result is more trustworthy and shows a regression. BPM and key need further work before they add value — see `20260316_122348/README.md` for investigation directions.
