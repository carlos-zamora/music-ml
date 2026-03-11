from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Float

class Base(DeclarativeBase):
    pass

class Track(Base):
    __tablename__ = "tracks"
    
    id: Mapped[int] = mapped_column(primary_key=True)

    rekordbox_id: Mapped[str] = mapped_column(String, unique=True)
    title: Mapped[str]
    artist: Mapped[str]
    bpm: Mapped[float | None]
    musical_key: Mapped[str | None]
    path: Mapped[str]