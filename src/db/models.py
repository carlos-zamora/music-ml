from sqlalchemy import String, Integer, Float, Table, Column, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

track_playlists = Table(
    "track_playlists",
    Base.metadata,
    Column("track_id", Integer, ForeignKey("tracks.id"), primary_key=True),
    Column("playlist_id", Integer, ForeignKey("playlists.id"), primary_key=True),
)

class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(primary_key=True)
    rekordbox_id: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str]
    artist: Mapped[str]
    bpm: Mapped[float | None]
    musical_key: Mapped[str | None]
    path: Mapped[str]
    length: Mapped[int]
    sample_rate: Mapped[int]
    playlists: Mapped[list["Playlist"]] = relationship(
        secondary=track_playlists, back_populates="tracks"
    )

class Playlist(Base):
    __tablename__ = "playlists"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    tracks: Mapped[list["Track"]] = relationship(
        secondary=track_playlists, back_populates="playlists"
    )
