import os
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from dotenv import load_dotenv
from sqlalchemy import select
from src.db.models import Track, Playlist
from src.db.session import SessionLocal
from src.terminal import print_import_summary


def normalize_filepath(path: str):
    if path.startswith("file://localhost/"):
        path = path.removeprefix("file://localhost/")
    return unquote(path)

load_dotenv()

XML_PATH = os.getenv("REKORDBOX_XML_PATH")
ALLOWED_FOLDERS = set(os.getenv("ALLOWED_FOLDERS", "").split(","))


# Parses rekordbox XML and returns track/playlist data for the allowed folders.
#
# XML structure:
#   PLAYLISTS
#   └── NODE  (root)
#       └── NODE  (folder)          ← one per top-level folder (e.g. "Dubstep")
#           └── NODE  (playlist)    ← one per playlist inside the folder
#               └── TRACK Key=...  ← reference by TrackID only (no metadata)
#   COLLECTION
#   └── TRACK TrackID=... Name=... Artist=... Location=... ...
#                                   ← full metadata for every track in library
#
# We scan PLAYLISTS first to build a map of which tracks are referenced and by
# which playlists, then scan COLLECTION and keep only those tracks.
#
# In:
# - xml_path: path to the rekordbox XML export file.
# - allowed_folders: set of top-level folder names to import.
# Out:
# - tracks_data: list of dicts with track fields + "playlist_names" key.
# - playlists_data: list of dicts with "name" and "track_ids" keys.
def parse_xml(xml_path, allowed_folders):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Maps rekordbox TrackID -> list of playlist names the track belongs to.
    # Only populated for tracks inside allowed folders.
    track_id_to_playlists = {}
    playlists_data = []

    for folder_node in root.find("PLAYLISTS").find("NODE").findall("NODE"):
        if folder_node.get("Name") not in allowed_folders:
            continue
        for playlist_node in folder_node.findall("NODE"):
            name = playlist_node.get("Name")
            track_ids = [t.get("Key") for t in playlist_node.findall("TRACK")]
            playlists_data.append({"name": name, "track_ids": track_ids})
            for tid in track_ids:
                track_id_to_playlists.setdefault(tid, []).append(name)

    tracks_data = []
    # Skip tracks not referenced by any allowed playlist, and skip .m4a files
    # (unsupported format).
    for t in root.find("COLLECTION").findall("TRACK"):
        tid = t.get("TrackID")
        if tid not in track_id_to_playlists:
            continue
        loc = t.get("Location")
        if loc.endswith(".m4a"):
            continue
        tracks_data.append({
            "rekordbox_id": tid,
            "title": t.get("Name"),
            "artist": t.get("Artist"),
            "path": normalize_filepath(loc),
            "bpm": float(t.get("AverageBpm")) if t.get("AverageBpm") else None,
            "musical_key": t.get("Tonality"),
            "length": int(t.get("TotalTime")),
            "sample_rate": int(t.get("SampleRate")),
            "playlist_names": track_id_to_playlists[tid],
        })

    return tracks_data, playlists_data


# Insert a new Track or update the existing one matched by rekordbox_id.
# In:
# - session: active SQLAlchemy session.
# - data: track dict from parse_xml (includes "playlist_names" key).
# Out:
# - the Track ORM object (new or existing), with scalar fields set.
def upsert_track(session, data):
    track = session.execute(
        select(Track).where(Track.rekordbox_id == data["rekordbox_id"])
    ).scalar_one_or_none()
    if track is None:
        track = Track()
        session.add(track)
    # Set all scalar fields; playlist_names is handled separately via the relationship.
    for k, v in data.items():
        if k != "playlist_names":
            setattr(track, k, v)
    return track


# Insert a new Playlist or return the existing one matched by name.
# In:
# - session: active SQLAlchemy session.
# - name: playlist name (unique natural key).
# Out:
# - the Playlist ORM object (new or existing).
def upsert_playlist(session, name):
    pl = session.execute(select(Playlist).where(Playlist.name == name)).scalar_one_or_none()
    if pl is None:
        pl = Playlist(name=name)
        session.add(pl)
    return pl


def main():
    tracks_data, playlists_data = parse_xml(XML_PATH, ALLOWED_FOLDERS)

    with SessionLocal() as session:
        # Upsert all playlists first so we can reference them when linking tracks.
        playlist_map = {pl["name"]: upsert_playlist(session, pl["name"]) for pl in playlists_data}

        for td in tracks_data:
            track = upsert_track(session, td)
            # flush() writes the track row so track.id is assigned before we
            # write to the track_playlists join table.
            session.flush()
            track.playlists = [playlist_map[n] for n in td["playlist_names"]]

        session.commit()
        print_import_summary(len(tracks_data), len(playlists_data))


if __name__ == "__main__":
    main()
