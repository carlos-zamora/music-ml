import types
import numpy as np
import pytest


@pytest.fixture
def fake_track():
    """Minimal Track stand-in (no DB needed)."""
    return types.SimpleNamespace(
        artist="Test Artist",
        title="Test Song (feat. Guest)",
        length=180,
        path="/fake/audio.mp3",
        sample_rate=44100,
        bpm=150.0,
        musical_key="7A",
        playlists=[],
        markers=[],
    )


@pytest.fixture
def small_vocab():
    """ArtistVocab with 3 known artists. Index 0 = OOV."""
    from artist_vocab import ArtistVocab
    tracks = [types.SimpleNamespace(artist=a) for a in ["Artist A", "Artist B", "Artist C"]]
    return ArtistVocab(tracks)


@pytest.fixture
def synthetic_mel():
    """Random array shaped like a real mel output: (128, 938)."""
    return np.random.default_rng(0).standard_normal((128, 938)).astype(np.float32)


@pytest.fixture
def tiny_model():
    """3-class CNN with a vocab of 5. Fast enough to run in any test."""
    from SimpleAudioCNN import SimpleAudioCNN
    return SimpleAudioCNN(num_classes=3, num_artists=5)


@pytest.fixture
def tiny_model_inputs():
    """Canonical inputs matching tiny_model (batch=4, 3 classes, 5 artists)."""
    import torch
    return {
        "x":          torch.randn(4, 1, 128, 938),
        "artist_idx": torch.zeros(4, dtype=torch.long),
        "bpm":        torch.tensor([[0.6], [0.5], [0.7], [0.4]]),
        "key_idx":    torch.tensor([7, 12, 0, 3], dtype=torch.long),
    }
