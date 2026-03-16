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
    markers: Mapped[list["TrackMarker"]] = relationship(
        "TrackMarker", back_populates="track", cascade="all, delete-orphan"
    )

class Playlist(Base):
    __tablename__ = "playlists"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    tracks: Mapped[list["Track"]] = relationship(
        secondary=track_playlists, back_populates="playlists"
    )

class TrackMarker(Base):
    __tablename__ = "track_markers"

    id: Mapped[int] = mapped_column(primary_key=True)
    track_id: Mapped[int] = mapped_column(Integer, ForeignKey("tracks.id"), nullable=False)
    position_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    track: Mapped["Track"] = relationship(back_populates="markers")
