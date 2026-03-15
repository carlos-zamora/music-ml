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
