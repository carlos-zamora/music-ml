import torch
import pytest


def test_encode_audio_output_shape(tiny_model):
    x = torch.randn(4, 1, 128, 938)
    features = tiny_model.encode_audio(x)
    assert features.shape == (4, 128)


def test_classify_with_mask_shape(tiny_model):
    audio_feat = torch.randn(4, 128)
    artist_idx = torch.zeros(4, 2, dtype=torch.long)
    artist_mask = torch.ones(4, 2, dtype=torch.bool)
    logits = tiny_model.classify(audio_feat, artist_idx, artist_mask)
    assert logits.shape == (4, 3)


def test_forward_output_shape(tiny_model):
    x = torch.randn(4, 1, 128, 938)
    artist_idx = torch.zeros(4, dtype=torch.long)
    logits = tiny_model(x, artist_idx)
    assert logits.shape == (4, 3)


def test_output_is_logits_not_probs(tiny_model):
    torch.manual_seed(42)
    x = torch.randn(8, 1, 128, 938)
    artist_idx = torch.zeros(8, dtype=torch.long)
    logits = tiny_model(x, artist_idx)
    # Raw logits from an untrained model will have values outside [0, 1]
    in_unit_interval = (logits >= 0) & (logits <= 1)
    assert not in_unit_interval.all(), "Expected raw logits, but all values are in [0, 1]"
