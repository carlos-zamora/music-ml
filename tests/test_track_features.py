import pytest
from track_features import encode_camelot_key, normalize_bpm, BPM_MAX, CAMELOT_SIZE


# --- encode_camelot_key: Camelot notation input ---

def test_camelot_a_side_lower_bound():
    assert encode_camelot_key("1A") == 1

def test_camelot_a_side_upper_bound():
    assert encode_camelot_key("12A") == 12

def test_camelot_b_side_lower_bound():
    assert encode_camelot_key("1B") == 13

def test_camelot_b_side_upper_bound():
    assert encode_camelot_key("12B") == 24

def test_camelot_midrange():
    assert encode_camelot_key("7A") == 7
    assert encode_camelot_key("7B") == 19

def test_camelot_all_indices_in_range():
    for suffix, offset in (("A", 0), ("B", 12)):
        for num in range(1, 13):
            idx = encode_camelot_key(f"{num}{suffix}")
            assert 1 <= idx <= CAMELOT_SIZE, f"{num}{suffix} → {idx} out of range"

def test_camelot_all_indices_unique():
    seen = set()
    for suffix in ("A", "B"):
        for num in range(1, 13):
            idx = encode_camelot_key(f"{num}{suffix}")
            assert idx not in seen, f"Duplicate index {idx}"
            seen.add(idx)


# --- encode_camelot_key: standard pitch notation input ---

def test_standard_minor_am():
    # Am = 8A = index 8
    assert encode_camelot_key("Am") == 8

def test_standard_minor_em():
    # Em = 9A = index 9
    assert encode_camelot_key("Em") == 9

def test_standard_major_c():
    # C = 8B = index 20
    assert encode_camelot_key("C") == 20

def test_standard_sharp_notation():
    # F#m = 11A = index 11
    assert encode_camelot_key("F#m") == 11

def test_standard_flat_notation():
    # Bbm = 3A = index 3
    assert encode_camelot_key("Bbm") == 3


# --- encode_camelot_key: unknown/invalid input ---

def test_none_returns_zero():
    assert encode_camelot_key(None) == 0

def test_empty_string_returns_zero():
    assert encode_camelot_key("") == 0

def test_invalid_num_returns_zero():
    assert encode_camelot_key("13A") == 0
    assert encode_camelot_key("0B") == 0

def test_garbage_returns_zero():
    assert encode_camelot_key("XYZ") == 0


# --- normalize_bpm ---

def test_none_bpm_returns_zero():
    assert normalize_bpm(None) == pytest.approx(0.0)

def test_zero_bpm_returns_zero():
    assert normalize_bpm(0.0) == pytest.approx(0.0)

def test_max_bpm_returns_one():
    assert normalize_bpm(BPM_MAX) == pytest.approx(1.0)

def test_half_bpm_returns_half():
    assert normalize_bpm(BPM_MAX / 2) == pytest.approx(0.5)

def test_over_max_bpm_clamped_to_one():
    assert normalize_bpm(BPM_MAX * 2) == pytest.approx(1.0)

def test_typical_edm_bpm_in_range():
    for bpm in [128.0, 140.0, 150.0, 174.0]:
        result = normalize_bpm(bpm)
        assert 0.0 < result < 1.0, f"BPM {bpm} normalized to {result}"
