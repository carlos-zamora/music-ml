import csv
import librosa
import numpy as np
import sys
import types
import urllib.request
from pathlib import Path

PANNS_VERSION = 1
PANNS_DIM = 527
PANNS_SR = 32000

# Project-local paths. Kept under data/ so weights and labels live alongside the
# SQLite DB and feature caches instead of polluting HOME.
_MODEL_DIR = Path("./data/panns_model")
_CHECKPOINT_PATH = _MODEL_DIR / "Cnn14_mAP=0.431.pth"
_LABELS_CSV_PATH = _MODEL_DIR / "class_labels_indices.csv"
_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
_LABELS_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
# Size gate matches panns_inference's own check; partial downloads are treated as missing.
_MIN_CHECKPOINT_BYTES = int(3e8)

_model = None


# Downloads a file via urllib if missing or truncated. Writes to a .part temp file
# first to avoid leaving partial downloads at the final path on interrupt.
def _ensure_download(path: Path, url: str, min_bytes: int = 1) -> None:
    if path.exists() and path.stat().st_size >= min_bytes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url} -> {path} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


# Pre-registers a `panns_inference.config` module in sys.modules so panns_inference's
# real config.py (which hardcodes ~/panns_data/ and shells out to wget at import time)
# never executes. Populates the same public names the real module exposes, loaded
# from our project-local CSV.
def _install_config_shim() -> None:
    if "panns_inference.config" in sys.modules:
        return

    with open(_LABELS_CSV_PATH, "r", encoding="utf-8") as f:
        lines = list(csv.reader(f, delimiter=","))

    ids, labels = [], []
    for row in lines[1:]:
        ids.append(row[1])
        labels.append(row[2])

    shim = types.ModuleType("panns_inference.config")
    shim.sample_rate = PANNS_SR
    shim.labels_csv_path = str(_LABELS_CSV_PATH)
    shim.labels = labels
    shim.ids = ids
    shim.classes_num = len(labels)
    shim.lb_to_ix = {label: i for i, label in enumerate(labels)}
    shim.ix_to_lb = {i: label for i, label in enumerate(labels)}
    shim.id_to_ix = {id_: i for i, id_ in enumerate(ids)}
    shim.ix_to_id = {i: id_ for i, id_ in enumerate(ids)}
    sys.modules["panns_inference.config"] = shim


# Lazily loads and returns a cached AudioTagging model instance.
def _get_model():
    global _model
    if _model is None:
        _ensure_download(_LABELS_CSV_PATH, _LABELS_CSV_URL)
        _ensure_download(_CHECKPOINT_PATH, _CHECKPOINT_URL, min_bytes=_MIN_CHECKPOINT_BYTES)
        _install_config_shim()
        from panns_inference import AudioTagging
        _model = AudioTagging(checkpoint_path=str(_CHECKPOINT_PATH), device="cpu")
    return _model


# Computes the PANNs CNN14 AudioSet tag probability vector for a track.
# When cache_dir is provided, reads from a persisted .npy file if one exists for this
# track, and writes one after computing if not. Cache files are keyed by
# {track.rekordbox_id}.npy — rekordbox_id is stable across DB rebuilds (unlike the
# autoincrement primary key), so caches survive re-importing the library.
# Returns a (527,) float32 array of per-tag probabilities in [0, 1]. On extraction
# failure, caches and returns a NaN sentinel of the same shape.
# In:
# - track: Track ORM object.
# - cache_dir: optional path to a directory for persisted .npy feature files.
# Out:
# - np.ndarray shaped (527,) with probabilities in [0, 1] (or NaN on failure).
def extract_tags(track, cache_dir=None):
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{track.rekordbox_id}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    try:
        audio, _ = librosa.load(path=track.path, sr=PANNS_SR, mono=True)
        audio = audio[np.newaxis, :]  # (1, samples)
        clipwise_output, _ = _get_model().inference(audio)
        vec = np.asarray(clipwise_output[0], dtype=np.float32)
    except Exception:
        vec = np.full(PANNS_DIM, np.nan, dtype=np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vec)

    return vec


# Encodes a PANNs tag vector into a model input vector with a trailing known-flag,
# matching the convention used by normalize_bpm and encode_key_circular in track_features.py.
# NaN sentinels (from failed extraction) and None produce an all-zero vector with known=0.
# In:
# - vec: (527,) array or None.
# Out:
# - list of 528 floats: 527 tag probabilities followed by known flag (0.0 or 1.0).
def encode_tags(vec):
    if vec is None or not np.all(np.isfinite(vec)):
        return [0.0] * PANNS_DIM + [0.0]
    return [float(x) for x in vec] + [1.0]
