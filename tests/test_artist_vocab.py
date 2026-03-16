import types
import torch
import pytest
from artist_vocab import parse_featured_artists, ArtistVocab


# --- parse_featured_artists ---

def test_parse_feat_dot():
    assert parse_featured_artists("Song (feat. Artist B)") == ["Artist B"]


def test_parse_ft_abbreviation():
    result = parse_featured_artists("Song ft. Someone")
    assert "Someone" in result


def test_parse_multiple_artists_ampersand():
    result = parse_featured_artists("Song (feat. A & B)")
    assert "A" in result
    assert "B" in result


def test_parse_remix():
    result = parse_featured_artists("Song [DJ Name Remix]")
    assert "DJ Name" in result


def test_parse_no_match():
    assert parse_featured_artists("Plain Title") == []


# --- ArtistVocab ---

def test_vocab_size_includes_oov(small_vocab):
    # 3 artists + 1 OOV slot = 4
    assert small_vocab.size() == 4


def test_encode_known_artist_not_zero(small_vocab):
    assert small_vocab.encode("Artist A") > 0


def test_encode_unknown_returns_zero(small_vocab):
    assert small_vocab.encode("Nobody Known") == 0


def test_encode_case_insensitive(small_vocab):
    assert small_vocab.encode("ARTIST A") == small_vocab.encode("artist a")


def test_encode_all_deduplicates(small_vocab):
    # "Artist A feat. Artist A" — same artist appears as primary and featured
    indices = small_vocab.encode_all("Artist A", "Track (feat. Artist A)")
    assert indices.count(small_vocab.encode("Artist A")) == 1


def test_encode_all_fallback_to_oov(small_vocab):
    # Unknown primary artist, no featured artists in plain title → [0]
    result = small_vocab.encode_all("Unknown Artist", "Plain Title")
    assert result == [0]


# --- make_collate_fn ---

def test_collate_fn_shapes(small_vocab):
    import torch

    collate = small_vocab.make_collate_fn()

    mel     = torch.zeros(1, 1, 128, 938)
    bpm     = torch.tensor([0.6])
    key_idx = torch.tensor(7, dtype=torch.long)
    label   = torch.zeros(3)

    # Three items with 1, 2, and 3 artists respectively → max_len = 3
    batch = [
        (mel, [1],       bpm, key_idx, label),
        (mel, [1, 2],    bpm, key_idx, label),
        (mel, [1, 2, 3], bpm, key_idx, label),
    ]

    mels_out, padded, mask, bpms_out, key_idxs_out, labels_out = collate(batch)

    assert padded.shape == (3, 3)
    assert mask.shape == (3, 3)
    assert bpms_out.shape == (3, 1)
    assert key_idxs_out.shape == (3,)
    # First row: only position 0 is valid
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is False
    # Last row: all three positions are valid
    assert mask[2].all().item() is True
