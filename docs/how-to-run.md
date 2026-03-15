# How To Run

Use this workflow from `D:\projects\music-ml` when you add new Rekordbox music.

## 0) One-Time Project Setup

### Install the package

```powershell
pip install -e .
```

### Configure environment

Create a `.env` file in the project root (optional — defaults shown):

```
DATABASE_PATH=./data/music.db
DEBUG=false
REKORDBOX_XML_PATH=C:/Users/carlo/Music/rekordbox/rekordbox.xml
ALLOWED_FOLDERS=Dubstep,Riddim
```

- `DATABASE_PATH`: path to the SQLite database file. The `data/` directory is created automatically.
- `DEBUG`: set to `true` to enable SQLAlchemy query logging.
- `REKORDBOX_XML_PATH`: path to your Rekordbox XML export file.
- `ALLOWED_FOLDERS`: comma-separated list of top-level Rekordbox folder names to import (e.g. `Dubstep,Riddim`).

### Initialize the database

```powershell
python scripts/init_db.py
```

This creates the SQLite database and all tables (e.g. `tracks`). Only needs to be run once, or after schema changes.

## 1) Export or Update Rekordbox XML
- In Rekordbox, make sure your latest collection/playlists are included in your XML export.
- Confirm `REKORDBOX_XML_PATH` in your `.env` points to the exported file.

## 2) Register New Music Into This Project
- Run:

```powershell
python scripts\import_rekordbox.py
```

- This upserts tracks and playlists from the XML into the SQLite database. Existing records are updated in place; new ones are inserted.
- `ALLOWED_FOLDERS` in `.env` controls which top-level Rekordbox folders are imported.

## 3) Quick Dataset Sanity Check
- Optional: check number of tracks and playlists

```powershell
python -c "from src.db.session import SessionLocal; from src.db.models import Track, Playlist; s=SessionLocal(); print('tracks=',s.query(Track).count(),'playlists=',s.query(Playlist).count()); s.close()"
```

## 4) Train Model (Single Split Default Path)
- Run:

```powershell
python scripts/run.py train
```

- Options (all optional):
  - `--epochs` (default: `20`)
  - `--batch-size` (default: `16`)
  - `--report-dir` (default: `out`)
- Outputs in `out\<timestamp>_single\`:
  - `model.pth`
  - `summary.json`
  - `per_playlist_metrics.csv` (includes TP/FP/TN/FN per playlist)

## 5) Optional: K-Fold Evaluation
- Run:

```powershell
python scripts/run.py kfold
```

- Options (all optional):

| Option | Default | Description |
|---|---|---|
| `--epochs` | `5` | Training epochs per fold |
| `--folds` | `3` | Number of CV folds |
| `--batch-size` | `16` | Dataloader batch size |
| `--lr` | `1e-3` | Optimizer learning rate |
| `--seed` | `42` | Seed for deterministic splits |
| `--recall-min` | `0.65` | Minimum recall for guarded playlists |
| `--report-dir` | `out` | Output directory for reports |

- K-fold outputs go to `out\<timestamp>\`.

## 5a) Understanding Evaluation Output

Both single-split and k-fold runs write evaluation artifacts under `out\...`.

### `summary.json`

This file contains the high-level run summary.

- In a single-split run (`..._single\summary.json`), the structure is:
  - `mode`: identifies the run type (`"single_split"`). Expected value here is fixed text.
  - `labels`: ordered list of playlist labels used by the model. Informational only.
  - `validation`:
    - `loss`: model error from the loss function. Range: `>= 0`. Lower is better.
    - `metrics`:
      - `micro_f1`: F1 across all label decisions pooled together. Range: `0..1`. Higher is better.
      - `macro_f1`: average F1 across playlists, weighting each playlist equally. Range: `0..1`. Higher is better.
      - `micro_precision`: fraction of predicted positives that were correct across all labels pooled together. Range: `0..1`. Higher is better.
      - `macro_precision`: average precision across playlists. Range: `0..1`. Higher is better.
      - `micro_recall`: fraction of true positives found across all labels pooled together. Range: `0..1`. Higher is better.
      - `macro_recall`: average recall across playlists. Range: `0..1`. Higher is better.
  - `test`:
    - `loss`: test-set loss. Range: `>= 0`. Lower is better.
    - `metrics`: same metric fields as validation.

- In a k-fold run (`<timestamp>\summary.json`), the structure is broader:
  - `config`: evaluation settings used for the run.
    - includes fold count, epoch count, batch size, threshold tuning settings, recall guard, and output directory. Informational only.
  - `guarded_playlists`: the top playlists currently protected by the recall guard. Informational only.
  - `folds`: one entry per fold
    - `fold`: fold number. Informational only.
    - `train_size`: number of training tracks in that fold. Informational only.
    - `val_size`: number of validation tracks in that fold. Informational only.
    - `test_size`: number of test tracks in that fold. Informational only.
    - `thresholds`: per-playlist thresholds selected for that fold. Each value is typically `0..1`; lower predicts more often, higher predicts more conservatively.
    - `validation`:
      - `epoch`: best validation epoch selected. Range: positive integer. Informational only.
      - `train_loss`: training loss at selected epoch. Range: `>= 0`. Lower is better.
      - `val_loss`: validation loss at selected epoch. Range: `>= 0`. Lower is better.
      - `metrics`: validation metrics
      - `per_playlist`: per-playlist validation stats. See CSV column meanings below.
      - `guard_satisfied`: whether recall guard passed. `true` is better.
      - `guard_recalls`: recall values for guarded playlists. Each value is `0..1`; higher is better.
    - `test`:
      - `loss`: test loss for that fold. Range: `>= 0`. Lower is better.
      - `metrics`
  - `aggregate_metrics`: mean and standard deviation across folds for:
    - `micro_f1`: mean/std of pooled F1 across folds. Mean range: `0..1`; higher is better. Std range: `>= 0`; lower is better.
    - `macro_f1`: mean/std of equal-weighted playlist F1 across folds. Mean range: `0..1`; higher is better. Std range: `>= 0`; lower is better.
    - `micro_precision`: mean/std of pooled precision across folds. Mean range: `0..1`; higher is better. Std range: `>= 0`; lower is better.
    - `macro_precision`: mean/std of average playlist precision across folds. Mean range: `0..1`; higher is better. Std range: `>= 0`; lower is better.
    - `micro_recall`: mean/std of pooled recall across folds. Mean range: `0..1`; higher is better. Std range: `>= 0`; lower is better.
    - `macro_recall`: mean/std of average playlist recall across folds. Mean range: `0..1`; higher is better. Std range: `>= 0`; lower is better.

### `per_playlist_metrics.csv`

This file contains one row per playlist evaluation result.

- In a single-split run:
  - rows are tagged by `fold`:
    - `validation`
    - `test`

- In a k-fold run:
  - rows are tagged by `fold`:
    - numeric fold index (`1`, `2`, etc.) for per-fold test results
    - `aggregate` for totals combined across all folds

Columns:
- `playlist`: playlist name. Informational only.
- `fold`: which split/fold this row belongs to. Informational only.
- `support`: number of ground-truth positive tracks for that playlist in the evaluated split. Range: integer `>= 0`. More support means the metric is more statistically stable.
- `threshold`: decision threshold used for that playlist. Typical range: `0..1`. Lower predicts more often; higher predicts more conservatively.
- `precision`: `TP / (TP + FP)`. Of the songs predicted into the playlist, how many were correct. Range: `0..1`. Higher is better.
- `recall`: `TP / (TP + FN)`. Of the songs that belong in the playlist, how many the model found. Range: `0..1`. Higher is better.
- `f1`: harmonic mean of precision and recall. Range: `0..1`. Higher is better.
- `tp`: true positives. Correct positive predictions. Range: integer `>= 0`. Higher is better.
- `fp`: false positives. Songs incorrectly predicted into the playlist. Range: integer `>= 0`. Lower is better.
- `tn`: true negatives. Correct negative predictions. Range: integer `>= 0`. Higher is generally better, but less useful than `tp/fp/fn` when classes are imbalanced.
- `fn`: false negatives. Songs that belonged in the playlist but were missed. Range: integer `>= 0`. Lower is better.

How to use it:
- High `fp` means the model is over-predicting that playlist.
- High `fn` means the model is missing songs that belong in that playlist.
- Low `support` means the playlist is sparse, so metric swings may be noisy.
- Compare `validation` vs `test` rows (single split) or per-fold vs `aggregate` rows (k-fold) to spot instability.

## 6) Predict Playlists for Specific Tracks
- Run:

```powershell
python scripts/run.py predict --model out/<timestamp>_single/model.pth --filter "Riddim"
```

- `--model`: path to a saved `.pth` model file (required)
- `--filter`: regex string matched against track names (required)

## 7) Typical Update Cycle
1. Export Rekordbox XML.
2. Run `scripts/import_rekordbox.py`.
3. Train/evaluate.
4. Compare `out\` reports to prior runs.
5. Keep best model in its run directory under `out\` and track notes in GitHub issues.
