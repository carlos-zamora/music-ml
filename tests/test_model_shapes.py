import torch
import pytest


def test_encode_audio_output_shape(tiny_model):
    x = torch.randn(4, 1, 128, 938)
    features = tiny_model.encode_audio(x)
    assert features.shape == (4, 128)


def test_classify_with_mask_shape(tiny_model, tiny_model_inputs):
    audio_feat  = torch.randn(4, 128)
    artist_idx  = torch.zeros(4, 2, dtype=torch.long)
    artist_mask = torch.ones(4, 2, dtype=torch.bool)
    bpm         = tiny_model_inputs["bpm"]
    key_idx     = tiny_model_inputs["key_idx"]
    logits = tiny_model.classify(audio_feat, artist_idx, bpm, key_idx, artist_mask)
    assert logits.shape == (4, 3)


def test_classify_without_mask_shape(tiny_model, tiny_model_inputs):
    audio_feat = torch.randn(4, 128)
    artist_idx = torch.zeros(4, 1, dtype=torch.long)
    bpm        = tiny_model_inputs["bpm"]
    key_idx    = tiny_model_inputs["key_idx"]
    logits = tiny_model.classify(audio_feat, artist_idx, bpm, key_idx)
    assert logits.shape == (4, 3)


def test_forward_output_shape(tiny_model, tiny_model_inputs):
    logits = tiny_model(
        tiny_model_inputs["x"],
        tiny_model_inputs["artist_idx"],
        tiny_model_inputs["bpm"],
        tiny_model_inputs["key_idx"],
    )
    assert logits.shape == (4, 3)


def test_output_is_logits_not_probs(tiny_model, tiny_model_inputs):
    torch.manual_seed(42)
    logits = tiny_model(
        tiny_model_inputs["x"],
        tiny_model_inputs["artist_idx"],
        tiny_model_inputs["bpm"],
        tiny_model_inputs["key_idx"],
    )
    in_unit_interval = (logits >= 0) & (logits <= 1)
    assert not in_unit_interval.all(), "Expected raw logits, but all values are in [0, 1]"


def test_key_embedding_unknown_index(tiny_model, tiny_model_inputs):
    # Index 0 is the padding/unknown slot — should produce all-zero embedding
    audio_feat = torch.randn(4, 128)
    artist_idx = torch.zeros(4, 1, dtype=torch.long)
    bpm        = tiny_model_inputs["bpm"]
    key_idx    = torch.zeros(4, dtype=torch.long)  # all unknown
    logits = tiny_model.classify(audio_feat, artist_idx, bpm, key_idx)
    assert logits.shape == (4, 3)
