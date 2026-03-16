import types
import pytest
from load_mel import generate_partitions


@pytest.fixture
def track():
    return types.SimpleNamespace(length=180)


def test_returns_correct_count(track):
    result = generate_partitions(track, num_of_partitions=5, partition_length=10)
    assert len(result) == 5


def test_first_starts_at_zero(track):
    result = generate_partitions(track, num_of_partitions=5, partition_length=10)
    assert result[0].position == 0.0


def test_correct_partition_length(track):
    result = generate_partitions(track, num_of_partitions=5, partition_length=10)
    for p in result:
        assert p.length == 10


def test_positions_are_evenly_spaced(track):
    n = 5
    partition_length = 10
    result = generate_partitions(track, num_of_partitions=n, partition_length=partition_length)
    usable = track.length - partition_length  # 170
    step = usable / (n - 1)                  # 42.5
    for i, p in enumerate(result):
        assert p.position == pytest.approx(i * step)


def test_last_partition_within_track(track):
    result = generate_partitions(track, num_of_partitions=5, partition_length=10)
    last = result[-1]
    assert last.position + last.length <= track.length


def test_single_partition_no_crash(track):
    result = generate_partitions(track, num_of_partitions=1, partition_length=10)
    assert len(result) == 1
    assert result[0].position == 0.0


# --- marker-based partitions ---

def test_markers_used_when_sufficient(track):
    markers = [10.0, 50.0, 90.0, 130.0, 160.0]
    result = generate_partitions(track, num_of_partitions=5, partition_length=10, marker_positions=markers)
    positions = [p.position for p in result]
    assert positions == pytest.approx(sorted(markers))


def test_markers_sorted_before_selection(track):
    # Unsorted markers — function should sort and take first N
    markers = [160.0, 10.0, 90.0, 50.0, 130.0]
    result = generate_partitions(track, num_of_partitions=3, partition_length=10, marker_positions=markers)
    assert result[0].position == pytest.approx(10.0)
    assert result[1].position == pytest.approx(50.0)
    assert result[2].position == pytest.approx(90.0)


def test_marker_clamped_when_past_track_end(track):
    # Marker at 175s with partition_length=10 → max_start = 170
    markers = [175.0, 50.0, 90.0, 130.0, 10.0]
    result = generate_partitions(track, num_of_partitions=5, partition_length=10, marker_positions=markers)
    for p in result:
        assert p.position + p.length <= track.length


def test_markers_ignored_when_fewer_than_requested(track):
    # Only 3 markers but 5 partitions requested → equidistant fallback
    markers = [10.0, 50.0, 90.0]
    result = generate_partitions(track, num_of_partitions=5, partition_length=10, marker_positions=markers)
    # Equidistant: usable=170, step=42.5
    assert result[0].position == pytest.approx(0.0)
    assert result[1].position == pytest.approx(42.5)


def test_empty_markers_uses_equidistant(track):
    result_no_markers  = generate_partitions(track, num_of_partitions=5, partition_length=10)
    result_empty_list  = generate_partitions(track, num_of_partitions=5, partition_length=10, marker_positions=[])
    result_none        = generate_partitions(track, num_of_partitions=5, partition_length=10, marker_positions=None)
    for a, b, c in zip(result_no_markers, result_empty_list, result_none):
        assert a.position == pytest.approx(b.position)
        assert a.position == pytest.approx(c.position)
