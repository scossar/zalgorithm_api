# Zalgorithm API

The existing HTML API, updated for the fragment indexer's permanent numeric IDs.
It supports either a completed local indexer snapshot or a separate Chroma server.
No endpoint generates or changes embeddings or source HTML.

## Local development without Docker

Python 3.13 and `uv` are used for the application environment. Chroma is pinned to
1.5.9 to match `fragment_indexer`. Install from the lock and serve a completed build:

```bash
cd ~/projects/python/zalgorithm_api
uv sync --locked
INDEX_SNAPSHOT=../fragment_indexer/output/current \
  uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Test the existing API:

```bash
curl -X POST -d 'query=gradient descent' http://127.0.0.1:8000/api/query
curl http://127.0.0.1:8000/api/fragment/1
```

`INDEX_SNAPSHOT` points at the release root containing `build.json`, `chroma/`, and
`sqlite/sections.db`. At startup, the API resolves the path once and copies the two
databases into a private temporary directory. Chroma may write during reads, so
this prevents it from changing the indexer's completed snapshot. SQLite is opened
read-only. Startup verifies the snapshot schema, encoder/tokenizer, Chroma version,
collection name, and database counts. Resources and temporary copies are released
at shutdown. The first startup may download the encoder if it is not cached.

Restart the API after switching `output/current` to use a new release. Code reload
alone does not watch or automatically activate new index snapshots. Each worker
has its own copy and model, so start with one worker for local development.

## Endpoints

| Request | Successful response |
| --- | --- |
| `POST /api/query`, form field `query` | Up to five HTML fragments in search order |
| `GET /api/fragment/{row_id}` | Close button, HTML heading, and HTML fragment |

Success responses are still HTML, with the original HTMX close-button behavior.
Search results are deduplicated by `db_id`, not heading text. Candidate retrieval
expands when several embedding chunks represent the same fragment, up to a
configurable cap. Distinct fragments with identical heading text remain distinct.

Whitespace-only queries and empty result sets return empty HTML. Missing form
fields, invalid IDs, and oversized queries return 422. Query inputs are limited to
4096 characters and the encoder's actual 256-token limit, including special tokens;
they are not silently truncated. Missing/deleted fragment IDs return HTML with
status 404. Backend failures or Chroma/SQLite reference mismatches return a generic
503 response, with details logged server-side. API responses use `Cache-Control:
no-store` so HTTP caching does not retain outdated fragment responses.

The current Hugo hook does not provide a special 404/503 UI. Its `click once`
behavior also keeps already-loaded fragment HTML in the page until refresh.
Those client-side behaviors are separate from this API update.

## Separate Chroma server

Leave `INDEX_SNAPSHOT` unset to use the existing architecture:

```bash
CHROMA_HOST=127.0.0.1 CHROMA_PORT=8001 \
SQLITE_DB_PATH=/path/to/matching/sqlite/sections.db \
  uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The SQLite file and Chroma server must contain the same completed release. This
mode opens SQLite read-only and uses the explicitly selected all-MiniLM-L6-v2
encoder for queries. The Chroma collection must exist; startup does not create an
empty replacement. Blocking Chroma I/O and model inference run in a worker thread,
so they do not block FastAPI's event loop. Clients are closed at shutdown.

| Variable | Default |
| --- | --- |
| `INDEX_SNAPSHOT` | Unset: use the HTTP Chroma server |
| `SQLITE_DB_PATH` | `/data/sections.db` |
| `CHROMA_HOST` | `localhost` |
| `CHROMA_PORT` | `8000` |
| `CHROMA_SSL` | `false` |
| `CHROMA_COLLECTION` | `zalgorithm` |
| `SEARCH_RESULT_LIMIT` | `5` |
| `SEARCH_CANDIDATE_LIMIT` | `200` |
| `CORS_ORIGINS` | `http://localhost:1313,http://127.0.0.1:1313,https://zalgorithm.com` |

Snapshot mode takes precedence over the remote connection settings and
`SQLITE_DB_PATH`. CORS origins are a comma-separated list. Query encoding stays
explicitly MiniLM; changing models requires corresponding indexer and API changes.

## Docker configuration

The base `docker-compose.yml` retains separate API and Chroma services and named
data volumes. Python is now 3.13 in the API image, and the Chroma image is pinned to
1.5.9 instead of `latest`. Selecting only this file excludes local overrides:

```bash
docker compose -f docker-compose.yml config
```

The local, ignored `docker-compose.override.yml` points the API at the new indexer
snapshot as a read-only mount. It no longer mounts the old embeddings-generator
databases. Use just the API service for local snapshot mode:

```bash
FRAGMENT_SNAPSHOT=../fragment_indexer/output/current \
  docker compose up --build --no-deps api
```

Recreate the container after changing the mounted release, because an existing
bind mount can retain the previous symlink target. This repository does not
populate remote data volumes or deploy the website. No deployment commands were
run while preparing these changes.

`pyproject.toml` declares dependencies and `uv.lock` locks them. Docker's
`requirements.txt` is generated from that lock. Regenerate it after changing deps:

```bash
uv export --locked --no-dev --no-emit-project --no-hashes \
  --format requirements-txt --output-file requirements.txt
```

## Tests

```bash
uv run python -m unittest discover -s tests -v
TEST_INDEX_SNAPSHOT=../fragment_indexer/output/current \
  uv run python -m unittest discover -s tests -v
```

The first command runs unit/API tests and skips integration tests. The second also
runs a real MiniLM query against a private snapshot copy and starts a temporary
Chroma HTTP server bound to loopback. The server is stopped when its test finishes;
source snapshots are never modified. Tests cover ranking and duplicate chunks,
HTML/form compatibility, invalid queries and IDs, missing/deleted fragments,
backend errors, CORS, application cleanup, and both database modes.
