import re
import time
from load_mel import generate_partitions, load_mels
import torch, json
from torch.utils.data import Dataset

class PlaylistDataset(Dataset):
    # Initializes dataset state from parsed library JSON.
    # In:
    # - libraryJson: JSON object loaded from tracks.json.
    def __init__(self, libraryJson):
        # idx: simple track ID
        # val: track info JSON
        self.trackList = libraryJson['tracks']

        # idx: simple playlist ID
        # val: playlist name
        self.playlistList = []
        for playlist in libraryJson['playlists']:
            self.playlistList.append(playlist["name"])
        
        # how many items have been retrieved
        # used for debugging
        self.getCount = 0
    
    # Loads tracks JSON from disk and constructs a PlaylistDataset.
    # In:
    # - jsonPath: path to tracks.json.
    # Out:
    # - PlaylistDataset instance.
    @classmethod
    def from_json(cls, jsonPath: str):
        with open(jsonPath, 'r') as f:
            library = json.load(f)
        return cls(library)
    
    # Finds tracks whose names match a regex.
    # In:
    # - trackNameRegex: regex pattern applied to each track name.
    # Out:
    # - list of matching track JSON objects.
    def find_tracks(self, trackNameRegex: str):
        results = []
        for track in self.trackList:
            if re.search(trackNameRegex, track['name']):
                results.append(track)
        return results

    # Returns the number of tracks in the dataset.
    def __len__(self):
        return len(self.trackList)
    
    # Returns playlist names where index is the label ID.
    def playlists(self):
        return self.playlistList
    
    # Gets a playlist ID from its name.
    # In:
    # - playlistName: playlist name to look up.
    # Out:
    # - playlist index, or None if not found.
    def _getPlaylistId(self, playlistName: str):
        for i in range(0, len(self.playlistList)):
            if playlistName == self.playlistList[i]:
                return i
    
    # Configures partition sampling for each track.
    # In:
    # - num_of_partitions: number of partitions per track.
    # - partition_length: length in seconds for each partition.
    def setPartitionConfig(self, num_of_partitions, partition_length):
        self.num_of_partitions = num_of_partitions
        self.partition_length = partition_length
    
    # Loads features and labels for one track index.
    # In:
    # - i: track index in the dataset.
    # Out:
    # - tuple(stacked_mels, label) for model training/evaluation.
    def __getitem__(self, i):
        self.getCount += 1
        start_time = time.time()

        trackInfo = self.trackList[i]
        trackPlaylists = set(trackInfo['playlists'])

        print(f"\t[{self.getCount}] {trackInfo['name']}", end='')

        # load track's mel spectograms
        partitions = generate_partitions(trackInfo, self.num_of_partitions, self.partition_length)
        mels = load_mels(trackInfo, partitions)

        # generate tensors from mel spectograms
        melTensors = []
        for mel in mels:
            melTensors.append(torch.tensor(mel, dtype=torch.float32).unsqueeze(0))
        stacked_mels = torch.stack(melTensors, dim=0)

        # idx: playlist ID
        # val: binary rep of track being in the playlist
        label = torch.zeros(len(self.playlistList), dtype=torch.float32)
        for playlistName in self.playlistList:
            if playlistName in trackPlaylists:
                label[self._getPlaylistId(playlistName)] = 1.0

        time_span = time.time() - start_time
        print(f"\t\t{time_span:.2f} seconds")
        return stacked_mels, label

if __name__ == "__main__":
    ds = PlaylistDataset.from_json('D:\\projects\\music-ml\\out\\tracks.json')
    print(ds[0])
