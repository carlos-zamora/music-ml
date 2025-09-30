import re
from load_mel import *
import torch, json
from torch.utils.data import Dataset

class PlaylistDataset(Dataset):
    def __init__(self, libraryJson):
        # idx: simple track ID
        # val: track info JSON
        self.trackList = libraryJson['tracks']

        # idx: simple playlist ID
        # val: playlist name
        self.playlistList = []
        for playlist in libraryJson['playlists']:
            self.playlistList.append(playlist["name"])
    
    @classmethod
    def from_json(cls, jsonPath: str):
        with open(jsonPath, 'r') as f:
            library = json.load(f)
        return cls(library)
    
    def find_tracks(self, trackNameRegex: str):
        results = []
        for track in self.trackList:
            if re.search(trackNameRegex, track['name']):
                results.append(track)
        return results

    def __len__(self):
        return len(self.trackList)
    
    def playlists(self):
        return self.playlistList
    
    def _getPlaylistId(self, playlistName: str):
        for i in range(0, len(self.playlistList)):
            if playlistName == self.playlistList[i]:
                return i
    
    def __getitem__(self, i):
        trackInfo = self.trackList[i]
        trackPlaylists = set(trackInfo['playlists'])

        # load track's mel spectogram
        mel = load_mel(trackInfo)
        melTensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        # idx: playlist ID
        # val: binary rep of track being in the playlist
        label = torch.zeros(len(self.playlistList), dtype=torch.float32)
        for playlistName in self.playlistList:
            if playlistName in trackPlaylists:
                label[self._getPlaylistId(playlistName)] = 1.0

        return melTensor, label

if __name__ == "__main__":
    ds = PlaylistDataset.from_json('D:\\projects\\music-ml\\out\\tracks.json')
    print(ds[0])
