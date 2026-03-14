# Database Management

The project uses **SQLite** with **SQLAlchemy** models and **Alembic** for schema migrations.

## Key Files

| File | Purpose |
|---|---|
| `src/db/models.py` | SQLAlchemy model definitions (`Track`, `Playlist`, `track_playlists`) |
| `src/db/session.py` | Engine and `SessionLocal` factory — reads `DATABASE_PATH` from `.env` |
| `alembic/env.py` | Alembic runtime config — imports `Base.metadata` for autogenerate |
| `alembic/versions/` | One `.py` file per migration, in order |
| `alembic.ini` | Alembic settings including the database URL |
| `data/music.db` | The SQLite database file (not committed to git) |

## How Alembic Works

Alembic tracks schema changes through **migration files** — Python scripts that contain the SQL needed to move the schema forward (upgrade) or backward (downgrade). The key distinction:

- `alembic revision --autogenerate` — **writes a migration file** by diffing your models against the live DB. It does **not** touch the database.
- `alembic upgrade head` — **executes** pending migration files against the database. This is what actually creates or alters tables.

Alembic records which migrations have been applied in a `alembic_version` table inside your database.

## One-Time Setup (Fresh Database)

If `data/music.db` does not exist yet:

```bash
# Generate the initial migration from your models
alembic revision --autogenerate -m "initial schema"

# Apply it — creates the database and all tables
alembic upgrade head
```

## Workflow: Changing the Schema

Whenever you add, remove, or modify a model in `src/db/models.py`:

```bash
# 1. Generate a migration describing what changed
alembic revision --autogenerate -m "short description of change"

# 2. Open the generated file in alembic/versions/ and review it
#    Make sure the upgrade() and downgrade() functions look correct.

# 3. Apply the migration
alembic upgrade head
```

Always review the generated file before applying — autogenerate is good but not perfect. It can miss things like column type changes or server-side defaults.

## Useful Commands

| Command | Description |
|---|---|
| `alembic revision --autogenerate -m "msg"` | Generate a migration by diffing models vs DB |
| `alembic upgrade head` | Apply all pending migrations |
| `alembic current` | Show which revision the DB is currently at |
| `alembic history --verbose` | List all migrations and their status |
| `alembic downgrade -1` | Roll back the last applied migration |
| `alembic downgrade base` | Roll back all migrations (empty database) |

## A Note on the Database URL

The application (`session.py`) reads `DATABASE_PATH` from your `.env` file and builds the URL dynamically. Alembic (`alembic.ini`) has its own hardcoded URL:

```ini
sqlalchemy.url = sqlite:///data/music.db
```

These must point to the same file. If you ever change `DATABASE_PATH` in `.env`, update `alembic.ini` to match.

## Gotchas

**Empty migration after `--autogenerate`?**
The database already has those tables. This happens when the DB was created with `Base.metadata.create_all()` directly (bypassing Alembic) before migrations were introduced. Fix: delete the database, delete the empty migration, then re-run autogenerate and upgrade.

**"No such table" error at runtime?**
You have a database file but `alembic upgrade head` has never been run against it. Run it now.

**Never use `Base.metadata.create_all()` to manage the schema.** Alembic is now the source of truth. Using `create_all()` creates tables without Alembic knowing about them, which breaks autogenerate.
