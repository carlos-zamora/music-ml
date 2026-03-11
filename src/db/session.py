import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/music.db")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(
    DATABASE_URL,
    echo=DEBUG,
    future=True 
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)