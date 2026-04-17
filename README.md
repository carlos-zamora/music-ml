# music-ml

A PyTorch CNN that classifies tracks for Rekordbox playlists. The model is trained on mel spectrograms fused with learned artist embeddings, circular key encodings, and normalized BPM features.

## How it works

1. **Data**: Track audio is read from disk and converted to mel spectrograms via librosa. Track metadata (artist, BPM, Camelot key) is stored in a SQLite database populated from a Rekordbox XML export.
2. **Model** (`SimpleAudioCNN`): A 3-block CNN encodes the spectrogram into a 128-dim audio feature. That is concatenated with a learned artist embedding (32-dim), circular key encoding (4-dim), and normalized BPM (2-dim), then passed through a linear classifier predicting multi-label playlist membership.
3. **Evaluation**: Supports k-fold cross-validation with per-playlist threshold tuning and a configurable recall guard for high-priority playlists.

## Setup

```bash
pip install -e .
```

Create a `.env` in the project root (all optional — defaults shown):

```
DATABASE_PATH=./data/music.db
REKORDBOX_XML_PATH=C:/Users/carlo/Music/rekordbox/rekordbox.xml
ALLOWED_FOLDERS=Dubstep,Riddim
DEBUG=false
```

Initialize the database (run once):

```bash
python scripts/init_db.py
```

## Workflow

```bash
# 1. Import tracks from Rekordbox XML
python scripts/import_rekordbox.py

# 2a. Train a model (single split)
python scripts/run.py train [--epochs 20] [--batch-size 16] [--report-dir out]

# 2b. Or run k-fold evaluation
python scripts/run.py kfold [--epochs 15] [--folds 3] [--batch-size 16] [--lr 1e-3] [--seed 42] [--recall-min 0.65] [--report-dir out]

# 3. Predict playlists for tracks matching a regex
python scripts/run.py predict --model out/<timestamp>_single/model.pth --filter "Riddim"
```

Shared options for all subcommands: `--db ./data/music.db`, `--parts 5`, `--part-len 10`

## Output

```
out/
  <timestamp>_single/    # train run
    model.pth
    summary.json
    per_playlist_metrics.csv
  <timestamp>/           # kfold run (no model saved)
    summary.json
    per_playlist_metrics.csv
```

`summary.json` includes aggregate metrics (micro/macro F1, precision, recall) and a `git_commit` field for traceability. `per_playlist_metrics.csv` has TP/FP/TN/FN, precision, recall, and F1 per playlist per fold.

See [`docs/how-to-run.md`](docs/how-to-run.md) for a detailed walkthrough including how to interpret evaluation output.

## Project structure

| Path | Purpose |
|---|---|
| `scripts/run.py` | CLI entry point |
| `src/SimpleAudioCNN.py` | Model definition, train, predict, save |
| `src/evaluation.py` | `EvalConfig`, k-fold logic, metric helpers |
| `src/PlaylistDataset.py` | PyTorch `Dataset` backed by SQLite |
| `src/artist_vocab.py` | Artist vocabulary, title parsing, `collate_fn` |
| `src/track_features.py` | BPM normalization, circular key encoding |
| `src/load_mel.py` | Mel spectrogram computation with local cache |
| `src/terminal.py` | Rich-based display layer |
| `scripts/import_rekordbox.py` | Rekordbox XML → SQLite upsert |
| `scripts/init_db.py` | Creates SQLite schema (run once) |
| `data/music.db` | SQLite database (tracks + playlists) |
