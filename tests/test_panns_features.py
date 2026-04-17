import types
import numpy as np
import pytest
from unittest.mock import patch

from panns_features import PANNS_DIM, encode_tags, extract_tags


# --- encode_tags ---

def test_encode_tags_valid_vector():
    vec = np.linspace(0.0, 1.0, PANNS_DIM, dtype=np.float32)
    out = encode_tags(vec)
    assert len(out) == PANNS_DIM + 1
    assert out[:PANNS_DIM] == pytest.approx(vec.tolist())
    assert out[-1] == 1.0


def test_encode_tags_none():
    out = encode_tags(None)
    assert len(out) == PANNS_DIM + 1
    assert out[-1] == 0.0
    assert all(x == 0.0 for x in out[:PANNS_DIM])


def test_encode_tags_nan_sentinel():
    vec = np.full(PANNS_DIM, np.nan, dtype=np.float32)
    out = encode_tags(vec)
    assert len(out) == PANNS_DIM + 1
    assert out[-1] == 0.0


# --- extract_tags caching ---

def _fake_track(tmp_path, track_id=7):
    return types.SimpleNamespace(id=track_id, rekordbox_id=str(track_id), path=str(tmp_path / "nope.mp3"), sample_rate=44100)


def test_extract_tags_cache_hit_skips_model(tmp_path):
    track = _fake_track(tmp_path)
    expected = np.arange(PANNS_DIM, dtype=np.float32) / PANNS_DIM
    cache_path = tmp_path / f"{track.rekordbox_id}.npy"
    np.save(cache_path, expected)

    with patch("panns_features._get_model") as get_model:
        out = extract_tags(track, cache_dir=str(tmp_path))
        get_model.assert_not_called()

    assert out.shape == (PANNS_DIM,)
    np.testing.assert_allclose(out, expected)


def test_extract_tags_cache_miss_writes_file(tmp_path):
    track = _fake_track(tmp_path, track_id=11)
    fake_audio = np.zeros(32000, dtype=np.float32)
    fake_clipwise = np.ones((1, PANNS_DIM), dtype=np.float32) * 0.25
    fake_embedding = np.zeros((1, 2048), dtype=np.float32)

    fake_model = types.SimpleNamespace(
        inference=lambda audio: (fake_clipwise, fake_embedding)
    )
    with patch("panns_features.librosa.load", return_value=(fake_audio, 32000)):
        with patch("panns_features._get_model", return_value=fake_model):
            out = extract_tags(track, cache_dir=str(tmp_path))

    cache_path = tmp_path / f"{track.rekordbox_id}.npy"
    assert cache_path.exists()
    assert out.shape == (PANNS_DIM,)
    np.testing.assert_allclose(out, fake_clipwise[0])


def test_extract_tags_failure_caches_nan(tmp_path):
    track = _fake_track(tmp_path, track_id=13)
    with patch("panns_features.librosa.load", side_effect=RuntimeError("bad file")):
        out = extract_tags(track, cache_dir=str(tmp_path))

    cache_path = tmp_path / f"{track.rekordbox_id}.npy"
    assert cache_path.exists()
    assert out.shape == (PANNS_DIM,)
    assert np.all(np.isnan(out))
