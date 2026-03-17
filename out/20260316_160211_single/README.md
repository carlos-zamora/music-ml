# Run: 20260316_160211_single

## BPM + Key Encoding Improvements — single (run 2 of 2)

Second single-split run at commit `76c8743` (Mar 16). Same code as `20260316_115848_single`.

| Metric | Run 1 (`115848_single`) | This run | Delta |
|---|---|---|---|
| test micro F1 | 0.240 | 0.200 | −17% |
| test macro F1 | 0.187 | 0.134 | −28% |
| test micro precision | 0.611 | 0.591 | −3% |
| test macro precision | 0.388 | 0.384 | −1% |
| test micro recall | 0.149 | 0.120 | −19% |
| test macro recall | 0.158 | 0.099 | −37% |

The gap between the two single runs (micro F1 0.240 vs 0.200) illustrates the unreliability of single-split evaluation — same code, different random split, meaningfully different numbers. Neither result should be interpreted in isolation. The kfold runs at the same commit (`20260316_160726`) are the authoritative signal.
