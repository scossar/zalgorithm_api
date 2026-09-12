"""Synchronous Chroma work runs in worker threads, including query encoding."""

import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from tokenizers import Tokenizer

from .config import Settings


class QueryTooLong(ValueError):
    pass


def sqlite_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro"


def select_fragment_ids(
    collection, query: str, limit: int, candidate_limit: int
) -> list[int]:
    """Expand the ranked candidate window when chunks repeat the same fragment."""
    total = min(collection.count(), candidate_limit)
    size = min(total, limit * 2)
    if not size:
        return []
    while True:
        result = collection.query(
            query_texts=[query], n_results=size, include=["metadatas"]
        )
        groups = result.get("metadatas")
        if not groups:
            raise RuntimeError("Chroma returned no metadata for a nonempty collection")
        unique = []
        seen = set()
        for metadata in groups[0]:
            db_id = metadata.get("db_id") if metadata else None
            if type(db_id) is not int or db_id < 1:
                raise RuntimeError("Chroma result has an invalid fragment identity")
            if db_id not in seen:
                seen.add(db_id)
                unique.append(db_id)
                if len(unique) == limit:
                    return unique
        if size >= total:
            return unique
        size = min(total, size * 2)


class Backend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.temporary = None
        self.snapshot_name = None
        try:
            metadata = None
            if settings.snapshot is not None:
                # Resolve current once. A later indexer build cannot mix snapshots.
                release = settings.snapshot.resolve(strict=True)
                metadata = json.loads((release / "build.json").read_text())
                if (
                    metadata.get("schema_version") != 1
                    or metadata.get("embedding_model") != "all-MiniLM-L6-v2"
                ):
                    raise ValueError("Unsupported snapshot schema or embedding model")
                if metadata.get("versions", {}).get("chromadb") != "1.5.9":
                    raise ValueError(
                        "Snapshot Chroma version must match this API (1.5.9)"
                    )
                if metadata.get("collection") != settings.collection:
                    raise ValueError(
                        "Snapshot collection differs from CHROMA_COLLECTION"
                    )
                self.snapshot_name = release.name
                self.temporary = tempfile.TemporaryDirectory(prefix="zalgorithm-api-")
                working = Path(self.temporary.name)
                # PersistentClient can write even during queries. Work on a private
                # copy so serving never changes the completed indexer snapshot.
                shutil.copytree(release / "chroma", working / "chroma")
                shutil.copy2(release / "sqlite/sections.db", working / "sections.db")
                self.db_path = working / "sections.db"
            else:
                self.db_path = settings.sqlite_path.resolve(strict=True)
            with closing(sqlite3.connect(sqlite_uri(self.db_path), uri=True)) as db:
                db.execute(
                    "SELECT id, html_heading, html_fragment FROM sections LIMIT 1"
                ).fetchall()
                fragment_count = db.execute("SELECT count(*) FROM sections").fetchone()[
                    0
                ]
                if metadata and fragment_count != metadata["fragments"]:
                    raise ValueError(
                        "Snapshot SQLite count differs from build manifest"
                    )

            chroma_settings = ChromaSettings(anonymized_telemetry=False)
            if settings.snapshot is not None:
                self.client = chromadb.PersistentClient(
                    path=str(working / "chroma"), settings=chroma_settings
                )
            else:
                self.client = chromadb.HttpClient(
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                    ssl=settings.chroma_ssl,
                    settings=chroma_settings,
                )
            encoder = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
            self.collection = self.client.get_collection(
                settings.collection, embedding_function=encoder
            )
            encoder(["Initialize the query embedding model."])
            self.tokenizer = Tokenizer.from_str(encoder.tokenizer.to_str())
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            if metadata:
                digest = hashlib.sha256(self.tokenizer.to_str().encode()).hexdigest()
                if digest != metadata["tokenizer_sha256"]:
                    raise ValueError(
                        "Query tokenizer differs from the snapshot tokenizer"
                    )
                if self.collection.count() != metadata["chunks"]:
                    raise ValueError(
                        "Snapshot Chroma count differs from build manifest"
                    )
        except BaseException:
            self.close()
            raise

    def search(self, query: str) -> list[int]:
        if len(self.tokenizer.encode(query).ids) > 256:
            raise QueryTooLong("Query exceeds the embedding model's 256-token limit")
        return select_fragment_ids(
            self.collection,
            query,
            self.settings.result_limit,
            self.settings.candidate_limit,
        )

    def close(self):
        try:
            if self.client is not None:
                self.client.close()
                self.client = None
        finally:
            if self.temporary is not None:
                self.temporary.cleanup()
                self.temporary = None
