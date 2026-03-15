import types
import numpy as np
import pytest
from unittest.mock import patch
from PlaylistDataset import PlaylistDataset


def make_track(title, artist, playlists=None):
    playlist_ns = [types.SimpleNamespace(name=p) for p in (playlists or [])]
    return types.SimpleNamespace(
        title=title,
        artist=artist,
        path="/fake/audio.mp3",
        length=180,
        sample_rate=44100,
        playlists=playlist_ns,
    )


@pytest.fixture
def sample_tracks():
    return [
        make_track("Track One",   "Artist A", playlists=["Rock"]),
        make_track("Track Two",   "Artist B", playlists=["Jazz"]),
        make_track("Track Three", "Artist C"),
    ]


@pytest.fixture
def sample_dataset(sample_tracks, small_vocab):
    return PlaylistDataset(sample_tracks, ["Rock", "Jazz"], vocab=small_vocab)


# --- length and playlist list ---

def test_from_db_loads_tracks(sample_tracks, small_vocab):
    ds = PlaylistDataset(sample_tracks, ["Rock", "Jazz"], vocab=small_vocab)
    assert len(ds) == 3


def test_playlist_order(sample_dataset):
    assert sample_dataset.playlists() == ["Rock", "Jazz"]


# --- label encoding ---

def test_label_tensor_encodes_membership(sample_dataset):
    sample_dataset.setPartitionConfig(5, 10)
    with patch("load_mel.librosa.load", return_value=(np.zeros(44100 * 10), 44100)):
        with patch("terminal.print_track_row"):
            mels, artist_idx, label = sample_dataset[0]

    rock_idx = sample_dataset.playlists().index("Rock")
    jazz_idx = sample_dataset.playlists().index("Jazz")
    assert label[rock_idx].item() == pytest.approx(1.0)
    assert label[jazz_idx].item() == pytest.approx(0.0)


# --- getitem without config ---

def test_getitem_raises_without_partition_config(sample_dataset):
    with pytest.raises(AttributeError):
        _ = sample_dataset[0]


# --- find_tracks ---

def test_find_tracks_regex(sample_dataset):
    results = sample_dataset.find_tracks(r"Track (One|Two)")
    titles = [t.title for t in results]
    assert "Track One" in titles
    assert "Track Two" in titles
    assert "Track Three" not in titles


def test_find_tracks_no_match(sample_dataset):
    results = sample_dataset.find_tracks(r"Nonexistent")
    assert results == []
