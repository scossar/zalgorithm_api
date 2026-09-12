"""FTS5 query interpretation and retrieval, independent of embeddings and HTTP."""

import re
import sqlite3
from contextlib import closing


class SearchInputError(ValueError):
    """A public, fixed-message query validation error."""


class KeywordIndexUnavailable(RuntimeError):
    pass


def keyword_expression(text: str, syntax: str = "simple") -> str:
    if "\x00" in text:
        raise SearchInputError("Keyword queries must not contain null characters")
    if syntax == "fts5":
        expression = text.strip()
    elif syntax == "simple":
        terms = []
        for match in re.finditer(r'"([^"]*)"|([^\s"]+)|(")', text):
            if match[3]:
                raise SearchInputError("Keyword query has an unclosed quoted phrase")
            term = match[1] if match[1] is not None else match[2]
            if not term.strip():
                raise SearchInputError("Keyword phrases must not be empty")
            terms.append('"' + term.replace('"', '""') + '"')
        expression = " AND ".join(terms)
    else:
        raise SearchInputError("Unknown keyword syntax")
    if not expression:
        raise SearchInputError("Keyword query must not be empty")

    # Let SQLite validate its own grammar against a disposable table. This keeps
    # malformed MATCH input (422) separate from failures of the real index (503).
    with closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE VIRTUAL TABLE sections_fts USING fts5(page_title, headings, body)")
        try:
            db.execute(
                "SELECT rowid FROM sections_fts WHERE sections_fts MATCH ?",
                (expression,),
            ).fetchall()
        except sqlite3.OperationalError as error:
            raise SearchInputError("Invalid keyword query syntax") from error
    return expression


def validate_keyword_index(db: sqlite3.Connection, manifest=None) -> bool:
    exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sections_fts'"
    ).fetchone()
    if manifest is not None and manifest.get("version") != 1:
        raise ValueError("Unsupported keyword index version")
    if not exists:
        if manifest is not None:
            raise ValueError("Snapshot declares a missing keyword index")
        return False
    db.execute("SELECT rowid, page_title, headings, body FROM sections_fts LIMIT 0")
    ids = {row[0] for row in db.execute("SELECT rowid FROM sections_fts")}
    section_ids = {row[0] for row in db.execute("SELECT id FROM sections")}
    if ids != section_ids:
        raise ValueError("Keyword index IDs differ from HTML fragment IDs")
    if manifest is not None and (
        manifest.get("fragments") != len(ids) or manifest.get("tokenizer") != "unicode61"
    ):
        raise ValueError("Keyword index differs from the snapshot manifest")
    # Also verify FTS5 is available and the virtual table supports MATCH.
    db.execute(
        "SELECT rowid FROM sections_fts WHERE sections_fts MATCH ? LIMIT 1",
        ('"keyword-index-startup-check"',),
    ).fetchall()
    return True


def matching_ids(db: sqlite3.Connection, expression: str) -> set[int]:
    """Filters need every match, independently of ranked result/candidate limits."""
    return {
        row[0]
        for row in db.execute(
            "SELECT rowid FROM sections_fts WHERE sections_fts MATCH ?", (expression,)
        )
    }


def ranked_ids(
    db: sqlite3.Connection,
    expression: str,
    limit: int,
    included: set[int] | None = None,
    excluded: set[int] | None = None,
) -> list[int]:
    result = []
    # Prefer title/heading matches. Filter before limiting so excluded top hits
    # cannot hide eligible lower-ranked results. Row ID breaks BM25 ties stably.
    with closing(db.execute(
        "SELECT rowid FROM sections_fts WHERE sections_fts MATCH ? "
        "ORDER BY bm25(sections_fts, 3.0, 2.0, 1.0), rowid",
        (expression,),
    )) as rows:
        for (identifier,) in rows:
            if included is not None and identifier not in included:
                continue
            if excluded and identifier in excluded:
                continue
            result.append(identifier)
            if len(result) == limit:
                break
    return result


def reciprocal_rank_fusion(*rankings: list[int], limit: int) -> list[int]:
    """Equal-weight RRF with k=60, one contribution per fragment per ranking."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(dict.fromkeys(ranking), start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (60 + rank)
    return sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:limit]
