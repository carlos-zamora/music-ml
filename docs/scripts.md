# Scripts Reference

All scripts are run from the project root (`D:\projects\music-ml`).

---

## `scripts/init_db.py`

Creates the SQLite database and all tables defined in `src/db/models.py`.

```powershell
python scripts/init_db.py
```

Run this once after cloning the repo, or again after any schema change (which requires deleting `data/music.db` first — see `src/db/models.py`).

**Reads from:** `.env` → `DATABASE_PATH`

---

## `scripts/import_rekordbox.py`

Parses the Rekordbox XML export and upserts tracks and playlists into the database. Safe to re-run — existing records are updated in place, not duplicated.

```powershell
python scripts/import_rekordbox.py
```

**Reads from:** `.env` → `REKORDBOX_XML_PATH`, `ALLOWED_FOLDERS`

| Env var | Example | Description |
|---|---|---|
| `REKORDBOX_XML_PATH` | `C:/Users/carlo/Music/rekordbox/rekordbox.xml` | Path to your Rekordbox XML export |
| `ALLOWED_FOLDERS` | `Dubstep,Riddim` | Comma-separated top-level folder names to import |

Only tracks that belong to a playlist inside one of the allowed folders are imported. `.m4a` files are skipped (unsupported format).

**Outputs:** populates `tracks`, `playlists`, and `track_playlists` tables in the database.
