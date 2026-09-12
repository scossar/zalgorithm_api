"""Configuration is read when an app is created, without opening databases."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    snapshot: Path | None = None
    sqlite_path: Path = Path("/data/sections.db")
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    collection: str = "zalgorithm"
    result_limit: int = 5
    candidate_limit: int = 200
    cors_origins: tuple[str, ...] = ("http://localhost:1313", "http://127.0.0.1:1313", "https://zalgorithm.com")

    def __post_init__(self):
        if not 1 <= self.chroma_port <= 65535:
            raise ValueError("CHROMA_PORT must be between 1 and 65535")
        if not 1 <= self.result_limit <= self.candidate_limit <= 10000:
            raise ValueError("Require 1 <= SEARCH_RESULT_LIMIT <= SEARCH_CANDIDATE_LIMIT <= 10000")
        if not self.collection.strip():
            raise ValueError("CHROMA_COLLECTION must not be empty")

    @classmethod
    def from_env(cls):
        snapshot = os.getenv("INDEX_SNAPSHOT")
        return cls(
            snapshot=Path(snapshot).expanduser() if snapshot else None,
            sqlite_path=Path(os.getenv("SQLITE_DB_PATH", "/data/sections.db")).expanduser(),
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            chroma_ssl=os.getenv("CHROMA_SSL", "false").lower() in {"true", "1"},
            collection=os.getenv("CHROMA_COLLECTION", "zalgorithm"),
            result_limit=int(os.getenv("SEARCH_RESULT_LIMIT", "5")),
            candidate_limit=int(os.getenv("SEARCH_CANDIDATE_LIMIT", "200")),
            cors_origins=tuple(s.strip() for s in os.getenv("CORS_ORIGINS", ",".join(cls.cors_origins)).split(",") if s.strip()),
        )
