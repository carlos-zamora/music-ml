import json
import librosa, numpy as np

# Calculate the mel spectrogram of an audio file
# In:
# - track: loaded JSON from tracks.json
# - target_width: desired width of the mel spectrogram
def load_mel(track, target_width=938):
    # calculate offset so we load a part of the song that's active
    trackLength = int(track["length"])
    offset = 60 if 60+10 < trackLength else max(trackLength/2-10, 0)

    # load 10 second segment of audio track at offset calculated above
    y, sr = librosa.load(path=track["path"],
                         sr=float(track["sampleRate"]),
                         offset=offset,
                         duration=10)
    
    # compute and normalize mel spectogram
    mel = librosa.feature.melspectrogram(y=y,
                                         sr=sr,
                                         n_mels=128,
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