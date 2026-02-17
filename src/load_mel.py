import json
import librosa, numpy as np

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

# Generates a list of Parititions for the given track
# In:
# - num_of_partitions: how many partitions we want
# - partition_length: the length of each partition
# Out:
# - partitions: list of partitions of the given track
def generate_partitions(track, num_of_partitions, partition_length):
    partitions = []
    trackLength = int(track["length"])
    
    # figure out step size from usable amount of the track
    # we don't want the partition to extend past the end of the track
    usable_length = trackLength - partition_length
    step_size = usable_length / (num_of_partitions - 1)
    for i in range(num_of_partitions):
        pos = i * step_size
        partitions.append(Partition(pos, partition_length))
    return partitions

# Calculates the mel spectogram of several partitions of an audio file
# In:
# - track: loaded JSON from tracks.json
# - partitions: list of Partition objects denoting which segments of the track to load
# - target_width: desired width of the mel spectogram
# Out:
# - mels: list of calculated mel spectograms (maps 1:1 to partitions)
def load_mels(track, partitions, target_width=DEFAULT_TARGET_WIDTH):
    mels = []
    for partition in partitions:
        mels.append(load_mel(track,
                             partition.position,
                             partition.length,
                             target_width))
    return mels

# Calculate the mel spectrogram of an audio file
# In:
# - track: loaded JSON from tracks.json
# - offset: start time of the segment in seconds.
# - duration: segment length in seconds.
# - target_width: desired width of the mel spectrogram
def load_mel(track, offset, duration=DEFAULT_PARTITION_DURATION, target_width=DEFAULT_TARGET_WIDTH):
    # load 10 second segment of audio track at offset calculated above
    y, sr = librosa.load(path=track["path"],
                         sr=float(track["sampleRate"]),
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
    
    return mel_normalized

if __name__ == "__main__":
    with open('D:\\projects\\music-ml\\out\\tracks.json', 'r') as f:
        library = json.load(f)
        count = 0
        for track in library["tracks"]:
            print("name: " + track["name"])
            print("mel: " + str(load_mel(track)))
            print('\n')
            count += 1
            if count > 10:
                break
