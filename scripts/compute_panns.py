import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from db.models import Track
from panns_features import PANNS_VERSION, extract_tags
from terminal import print_section, print_track_row


# Iterates every track, computes and caches its PANNs tag vector, and stamps
# tracks.panns_version on success. Idempotent: skips tracks already at the current
# PANNS_VERSION whose cache file exists.
# In:
# - db_path: path to SQLite database.
# - cache_dir: directory for persisted .npy feature files.
def compute_all(db_path: str, cache_dir: str) -> None:
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Session = sessionmaker(bind=engine)
    cache_path = Path(cache_dir)

    with Session() as session:
        tracks = session.execute(select(Track)).scalars().all()
        total = len(tracks)
        print_section(f"Computing PANNs tags for {total} tracks")

        for i, track in enumerate(tracks, start=1):
            expected = cache_path / f"{track.rekordbox_id}.npy"
            cached = track.panns_version == PANNS_VERSION and expected.exists()
            start = time.time()
            if not cached:
                extract_tags(track, cache_dir=cache_dir)
                track.panns_version = PANNS_VERSION
                session.add(track)
                if i % 20 == 0:
                    session.commit()
            elapsed = time.time() - start

            print_track_row(i, total, track.title, track.artist,
                            track.bpm, track.musical_key, elapsed, cached)

        session.commit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="./data/music.db")
    parser.add_argument("--cache-dir", default="./data/panns_cache")
    args = parser.parse_args()
    compute_all(args.db, args.cache_dir)
