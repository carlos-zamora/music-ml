import math
import pytest
from track_features import encode_camelot_key, encode_key_circular, normalize_bpm, BPM_MIN, BPM_RANGE, CAMELOT_SIZE


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

def test_none_bpm_returns_unknown():
    assert normalize_bpm(None) == [0.0, 0.0]

def test_known_bpm_sets_flag():
    result = normalize_bpm(150.0)
    assert result[1] == pytest.approx(1.0)

def test_unknown_bpm_clears_flag():
    assert normalize_bpm(None)[1] == pytest.approx(0.0)

def test_bpm_at_range_low():
    assert normalize_bpm(BPM_MIN)[0] == pytest.approx(0.0)

def test_bpm_at_range_high():
    assert normalize_bpm(BPM_MIN + BPM_RANGE)[0] == pytest.approx(1.0)

def test_bpm_at_midpoint():
    assert normalize_bpm(BPM_MIN + BPM_RANGE / 2)[0] == pytest.approx(0.5)

def test_bpm_above_range_clamped():
    assert normalize_bpm(BPM_MIN + BPM_RANGE * 2)[0] == pytest.approx(1.0)

def test_bpm_below_range_clamped():
    assert normalize_bpm(BPM_MIN - 50)[0] == pytest.approx(0.0)

def test_typical_edm_bpm_in_range():
    for bpm in [128.0, 140.0, 150.0, 174.0]:
        result = normalize_bpm(bpm)
        assert 0.0 <= result[0] <= 1.0, f"BPM {bpm} normalized to {result[0]}"


# --- encode_key_circular ---

def test_unknown_key_returns_zeros():
    assert encode_key_circular(None) == [0.0, 0.0, 0.0, 0.0]

def test_empty_key_returns_zeros():
    assert encode_key_circular("") == [0.0, 0.0, 0.0, 0.0]

def test_known_key_sets_flag():
    assert encode_key_circular("1A")[3] == pytest.approx(1.0)

def test_minor_key_mode_zero():
    assert encode_key_circular("7A")[2] == pytest.approx(0.0)

def test_major_key_mode_one():
    assert encode_key_circular("7B")[2] == pytest.approx(1.0)

def test_position_1_sin_cos():
    # Position 1: angle = 0 → sin=0, cos=1
    result = encode_key_circular("1A")
    assert result[0] == pytest.approx(0.0, abs=1e-9)
    assert result[1] == pytest.approx(1.0, abs=1e-9)

def test_position_7_sin_cos():
    # Position 7: angle = pi → sin=0, cos=-1
    result = encode_key_circular("7A")
    assert result[0] == pytest.approx(0.0, abs=1e-9)
    assert result[1] == pytest.approx(-1.0, abs=1e-9)

def test_adjacent_positions_equal_arc_distance():
    # All adjacent Camelot positions should be the same angular distance apart
    keys = [f"{n}A" for n in range(1, 13)]
    feats = [encode_key_circular(k) for k in keys]
    distances = []
    for i in range(len(feats)):
        a, b = feats[i], feats[(i + 1) % len(feats)]
        distances.append(math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))
    assert all(pytest.approx(distances[0], abs=1e-9) == d for d in distances)

def test_12_and_1_wrap_around():
    # 12A and 1A should be as close as any other adjacent pair
    k1 = encode_key_circular("1A")
    k2 = encode_key_circular("2A")
    k12 = encode_key_circular("12A")
    dist_12_to_1 = math.sqrt((k12[0] - k1[0])**2 + (k12[1] - k1[1])**2)
    dist_1_to_2 = math.sqrt((k1[0] - k2[0])**2 + (k1[1] - k2[1])**2)
    assert dist_12_to_1 == pytest.approx(dist_1_to_2, abs=1e-9)

def test_same_position_different_mode_same_circle_coords():
    # 5A and 5B share the same Camelot number, so sin/cos should be identical
    a = encode_key_circular("5A")
    b = encode_key_circular("5B")
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])
    assert a[2] != b[2]  # but mode differs

def test_standard_notation_routes_correctly():
    # Am = 8A; should match direct "8A" encoding
    assert encode_key_circular("Am") == pytest.approx(encode_key_circular("8A"))
