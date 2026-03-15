# Testing

The project uses [pytest](https://pytest.org). Tests live in `tests/` and cover pure functions first — no real audio files or database required.

## Running Tests

```bash
pytest                                             # all tests
pytest tests/test_artist_vocab.py                  # one file
pytest -k "test_parse"                             # by name pattern
pytest --cov=src --cov-report=term-missing         # with coverage
```

First-time setup (if not already installed):

```bash
pip install pytest pytest-cov
```

## Test Files

| File | What it covers |
|---|---|
| `tests/test_artist_vocab.py` | `parse_featured_artists`, `ArtistVocab` encode/OOV/dedup, `make_collate_fn` padding |
| `tests/test_generate_partitions.py` | Partition math, even spacing, boundary checks |
| `tests/test_evaluation_metrics.py` | `predict_labels`, `compute_multilabel_metrics`, `compute_per_playlist_confusion`, `tune_thresholds` |
| `tests/test_model_shapes.py` | CNN forward pass output shapes, logit (not sigmoid) assertion |
| `tests/test_playlist_dataset.py` | Label encoding, `find_tracks` regex, `__getitem__` guard before `setPartitionConfig` |

Shared fixtures (fake tracks, small vocab, synthetic mel, tiny model) are in `tests/conftest.py`.

## CI

Tests run automatically on every push and pull request to `main` via `.github/workflows/tests.yml`. The workflow installs CPU-only PyTorch (~200 MB instead of ~2 GB) so CI stays fast.