import librosa, numpy as np
from pathlib import Path

DEFAULT_TARGET_WIDTH = 938
DEFAULT_PARTITION_DURATION = 10
DEFAULT_N_MELS = 128

class Partition:
    # Creates a segment descriptor for loading one section of audio.
    # In:
    # - position: offset in seconds where the segment starts.
    # - length: duration in seconds for the segment.
    def __init__(self, position, length):
        self.position = position
        self.length = length

# Generates a list of Partitions for the given track.
# Uses marker positions when enough are available, otherwise falls back to
# equidistant partitions.
# In:
# - num_of_partitions: how many partitions we want
# - partition_length: the length of each partition in seconds
# - marker_positions: optional list of marker positions in seconds; if at least
#   num_of_partitions markers are provided they are used instead of equidistant spacing
# Out:
# - partitions: list of partitions of the given track
def generate_partitions(track, num_of_partitions, partition_length, marker_positions=None):
    trackLength = track.length
    max_start = trackLength - partition_length

    if marker_positions and len(marker_positions) >= num_of_partitions:
        sorted_positions = sorted(marker_positions)[:num_of_partitions]
        return [Partition(max(0.0, min(pos, max_start)), partition_length) for pos in sorted_positions]

    # Equidistant fallback
    usable_length = trackLength - partition_length
    step_size = usable_length / (num_of_partitions - 1) if num_of_partitions > 1 else 0
    return [Partition(i * step_size, partition_length) for i in range(num_of_partitions)]

# Calculates the mel spectogram of several partitions of an audio file
# In:
# - track: Track ORM object
# - partitions: list of Partition objects denoting which segments of the track to load
# - target_width: desired width of the mel spectogram
# - cache_dir: optional path to a directory for persisted .npy mel cache files
# Out:
# - mels: list of calculated mel spectograms (maps 1:1 to partitions)
def load_mels(track, partitions, target_width=DEFAULT_TARGET_WIDTH, cache_dir=None):
    mels = []
    for partition in partitions:
        mels.append(load_mel(track,
                             partition.position,
                             partition.length,
                             target_width,
                             cache_dir=cache_dir))
    return mels

# Calculate the mel spectrogram of an audio file.
# When cache_dir is provided, reads from a persisted .npy file if one exists for this
# track+offset, and writes one after computing if not. Cache files are keyed by
# {track.id}_{offset_ms}.npy — offset in integer milliseconds avoids float filename issues.
# In:
# - track: Track ORM object
# - offset: start time of the segment in seconds.
# - duration: segment length in seconds.
# - target_width: desired width of the mel spectrogram
# - cache_dir: optional path to a directory for persisted .npy mel cache files
def load_mel(track, offset=0, duration=DEFAULT_PARTITION_DURATION, target_width=DEFAULT_TARGET_WIDTH, cache_dir=None):
    if cache_dir is not None:
        offset_ms = int(round(offset * 1000))
        cache_path = Path(cache_dir) / f"{track.id}_{offset_ms}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    # load 10 second segment of audio track at offset calculated above
    y, sr = librosa.load(path=track.path,
                         sr=float(track.sample_rate),
                         offset=offset,
                         duration=duration)
    
    # compute and normalize mel spectogram
    mel = librosa.feature.melspectrogram(y=y,
                                         sr=sr,
                                         n_mels=DEFAULT_N_MELS,
                                         n_fft=2048,
                                         hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_normalized = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    # Ensure consistent width by padding or truncating
    current_width = mel_normalized.shape[1]
    if current_width < target_width:
        # Pad with zeros on the right
        padding = target_width - current_width
        mel_normalized = np.pad(mel_normalized, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    elif current_width > target_width:
        # Truncate to target width
        mel_normalized = mel_normalized[:, :target_width]

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mel_normalized)

    return mel_normalized
