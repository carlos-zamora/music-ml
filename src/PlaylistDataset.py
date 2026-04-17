from load_mel import generate_partitions, load_mels
from panns_features import encode_tags, extract_tags
from terminal import print_track_row
from track_features import encode_key_circular, normalize_bpm
from torch.utils.data import Dataset
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import selectinload
from db.models import Track, Playlist, TrackMarker
import re
import time
import torch

class PlaylistDataset(Dataset):
    # Initializes dataset state from pre-loaded lists of Track ORM objects and playlist names.
    # In:
    # - trackList: list of Track ORM objects (with playlists eagerly loaded).
    # - playlistList: ordered list of playlist name strings.
    # - vocab: ArtistVocab instance (or None; can be assigned later via ds.vocab).
    def __init__(self, trackList, playlistList, vocab=None):
        self.trackList = trackList

        # cache of track mel spectograms, maps 1:1 to trackList
        self.mels = [None] * len(self.trackList)

        self.playlistList = playlistList
        self.vocab = vocab
        self.cache_dir = None
        self.panns_cache_dir = None

        # how many items have been retrieved
        # used for debugging
        self.getCount = 0

    # Loads tracks from a SQLite database and constructs a PlaylistDataset.
    # In:
    # - db_path: path to the SQLite database file.
    # Out:
    # - PlaylistDataset instance.
    @classmethod
    def from_db(cls, db_path: str):
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            tracks = session.execute(
                select(Track).options(
                    selectinload(Track.playlists),
                    selectinload(Track.markers),
                )
            ).scalars().all()
            playlists = session.execute(
                select(Playlist).order_by(Playlist.id)
            ).scalars().all()
            playlist_names = [p.name for p in playlists]
        return cls(list(tracks), playlist_names)

    # Finds tracks whose titles match a regex.
    # In:
    # - trackNameRegex: regex pattern applied to each track title.
    # Out:
    # - list of matching Track ORM objects.
    def find_tracks(self, trackNameRegex: str):
        results = []
        for track in self.trackList:
            if re.search(trackNameRegex, track.title):
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

    # Resets the getCount tracker to 0.
    def resetCount(self):
        self.getCount = 0

    # Loads features and labels for one track index.
    # In:
    # - i: track index in the dataset.
    # Out:
    # - tuple(stacked_mels, artist_indices, label) for model training/evaluation.
    def __getitem__(self, i):
        self.getCount += 1
        start_time = time.time()

        trackInfo = self.trackList[i]
        trackPlaylists = set(p.name for p in trackInfo.playlists)

        # load and cache track's mel spectograms
        wasCached = True
        mels = self.mels[i]
        if mels is None:
            marker_positions = [m.position_seconds for m in trackInfo.markers] or None
            partitions = generate_partitions(trackInfo, self.num_of_partitions, self.partition_length, marker_positions)
            mels = load_mels(trackInfo, partitions, cache_dir=self.cache_dir)
            self.mels[i] = mels
            wasCached = False

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

        artist_indices = self.vocab.encode_all(trackInfo.artist, trackInfo.title)
        bpm = torch.tensor(normalize_bpm(trackInfo.bpm), dtype=torch.float32)
        key_feat = torch.tensor(encode_key_circular(trackInfo.musical_key), dtype=torch.float32)
        panns_vec = extract_tags(trackInfo, cache_dir=self.panns_cache_dir)
        panns_feat = torch.tensor(encode_tags(panns_vec), dtype=torch.float32)

        time_span = time.time() - start_time
        print_track_row(self.getCount, len(self.trackList), trackInfo.title, trackInfo.artist,
                        trackInfo.bpm, trackInfo.musical_key, time_span, wasCached)
        return stacked_mels, artist_indices, bpm, key_feat, panns_feat, label
