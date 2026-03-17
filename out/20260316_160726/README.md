# Run: 20260316_160726

## BPM + Key Encoding Improvements — kfold (run 2 of 2)

Second kfold run at commit `76c8743` (Mar 16). Same code as `20260316_122348` — circular Camelot key encoding and disambiguated BPM normalization. Run 1 showed a spurious regression; this run confirms the actual behaviour of the encoding changes.

| Metric | Baseline (artist only) | This run (+ BPM + key) | Delta |
|---|---|---|---|
| **micro F1** | 0.308 ± 0.009 | 0.313 ± 0.019 | **+2%** |
| **macro F1** | 0.231 ± 0.017 | 0.230 ± 0.016 | −0% |
| micro precision | 0.232 ± 0.017 | 0.247 ± 0.051 | +6% |
| macro precision | 0.240 ± 0.025 | 0.226 ± 0.042 | −6% |
| micro recall | 0.465 ± 0.028 | 0.471 ± 0.086 | +1% |
| macro recall | 0.348 ± 0.024 | 0.365 ± 0.084 | +5% |

The encoding improvements recovered the baseline — F1 is essentially flat versus the artist-only model. This is the expected outcome for a first pass: the new features are no longer actively misleading the model (as the old `bpm/250` + ambiguous zero + integer key index were), but 5 epochs on a small dataset isn't enough for the model to extract meaningful signal from them yet.

Variance is still higher than the baseline (std on recall 0.028 → 0.086), indicating the model hasn't fully stabilized with the new inputs. The circular encoding removes the need to learn key adjacency from scratch, but the optimizer still needs more time to weight BPM and key appropriately relative to audio and artist features.

**Next steps:** increase epochs (10–15) to give the model time to leverage BPM and key; or ablate BPM and key separately to see which contributes.
