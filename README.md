# Zalgorithm API

The Zalgorithm blog's HTML API. It supports either a completed local indexer snapshot
or a separate Chroma server. No endpoint generates or changes embeddings or source HTML.

## Local development without Docker

Python 3.13 and `uv` are used for the application environment. Chroma is pinned to
1.5.9 to match `fragment_indexer`. Install from the lock and serve a completed build:

```bash
cd ~/projects/python/zalgorithm_api
uv sync --locked
INDEX_SNAPSHOT=../fragment_indexer/output/current \
  uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
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

| Request                               | Successful response                           |
| ------------------------------------- | --------------------------------------------- |
| `POST /api/query`, form field `query` | Up to five HTML fragments in search order; semantic by default     |
| `GET /api/fragment/{row_id}`          | Close button, HTML heading, and HTML fragment |

Success responses are still HTML, with the original HTMX close-button behavior.
Search results are deduplicated by `db_id`, not heading text. Candidate retrieval
expands when several embedding chunks represent the same fragment, up to a
configurable cap. Distinct fragments with identical heading text remain distinct.

Whitespace-only queries and empty result sets return empty HTML. Missing form
fields, invalid IDs, and oversized queries return 422. Each query/filter field is
limited to 4096 characters. Semantic and hybrid queries
also enforce the encoder's actual 256-token limit, including special tokens;
keyword-only queries and keyword expressions do not use that token limit. Inputs
are not silently truncated. Missing/deleted fragment IDs return HTML with
status 404. Backend failures or Chroma/SQLite reference mismatches return a generic
503 response, with details logged server-side. API responses use `Cache-Control:
no-store` so HTTP caching does not retain outdated fragment responses.

The current Hugo hook does not provide a special 404/503 UI. Its `click once`
behavior also keeps already-loaded fragment HTML in the page until refresh.
Those client-side behaviors are separate from this API update.

## Keyword search, filters, and hybrid search

The same `POST /api/query` route accepts these additional form fields. It still
returns the existing fragment HTML; supplying only `query` preserves semantic search.

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `mode` | `semantic` | `semantic`, `keyword`, or `hybrid` |
| `query` | Required | Semantic text, keyword text, or both in hybrid mode |
| `keyword_query` | Use `query` | Optional keyword expression for **hybrid only** |
| `include` | Unset | Require a keyword match in the fragment |
| `exclude` | Unset | Exclude fragments matching this keyword expression |
| `keyword_syntax` | `simple` | `simple` or `fts5`, applied to every keyword expression in the request |

Both filters can be used together and apply in all three modes. Blank filters are
ignored. `keyword_query` on another mode, an invalid mode/syntax, malformed keyword
expressions, and oversized fields return 422. A whitespace-only main query returns
empty HTML. Empty literal phrases and unclosed quotes are invalid.

```bash
# Keyword search alone
curl -X POST -d 'mode=keyword' --data-urlencode 'query=gradient descent' \
  http://127.0.0.1:8000/api/query

# Semantic results must match "gradient" and must not match "reinforcement"
curl -X POST --data-urlencode 'query=how learning works' \
  -d 'include=gradient' -d 'exclude=reinforcement' \
  http://127.0.0.1:8000/api/query

# Merge semantic and keyword rankings (same text for both)
curl -X POST -d 'mode=hybrid' --data-urlencode 'query=gradient descent' \
  http://127.0.0.1:8000/api/query

# Separate semantic text and an advanced keyword expression
curl -X POST -d 'mode=hybrid' -d 'keyword_syntax=fts5' \
  --data-urlencode 'query=how learning works' \
  --data-urlencode 'keyword_query=(gradient OR stochastic) NOT reinforcement' \
  http://127.0.0.1:8000/api/query
```

Simple syntax treats whitespace-separated terms as literals joined with `AND`;
quoted phrases stay together. `gradient descent` requires both words, while
`"gradient descent"` requires the phrase. Operator words such as `OR` are literal
words in simple mode. The `unicode61` tokenizer ignores case and normalizes most
Latin diacritics; punctuation follows SQLite tokenization. This is word matching,
not arbitrary substring matching, stemming, or typo correction.

Explicit `keyword_syntax=fts5` accepts SQLite FTS5 syntax: `AND`, `OR`, binary `NOT`,
parentheses, quoted phrases, prefixes such as `grad*`, `NEAR`, and column filters
such as `body:gradient`. Available columns are `page_title`, `headings`, and `body`.
Use `exclude=reinforcement` for a standalone exclusion; FTS5's `NOT` requires a
left-hand expression. Expressions are bound SQL parameters, and their grammar is
validated against a disposable FTS table before searching the actual index.

The indexer builds `sections_fts` inside `sqlite/sections.db`, with one row per
**complete fragment**, sharing `rowid` with `sections.id`. Titles, heading ancestry,
and extracted plain text are searchable. The existing `sections.page_url` remains
the canonical post path and can be retrieved using that ID. Responses do not yet
expose additional structured metadata or keyword highlighting.

Keyword results use BM25 with title/heading/body weights of 3/2/1. Inclusion and
exclusion collect **all** matching IDs before retrieval, independently of result
or candidate limits. Chroma receives a `db_id` metadata filter, so eligible
fragments can be found even when they were outside the original semantic top five.
Keyword matches anywhere in a fragment qualify all its embedding chunks.

Hybrid mode applies filters to both searches, deduplicates semantic chunks by
fragment ID, and merges the ranked lists using equal-weight reciprocal rank fusion:
`score = sum(1 / (60 + rank))`, with ranks starting at one. A fragment absent from a
list gets no contribution from that list. Ties use ascending fragment ID.
`SEARCH_CANDIDATE_LIMIT` caps semantic embedding chunks examined and the keyword
candidate list (default 200); duplicate chunks can make the semantic fragment list
shorter. Fusion happens before `SEARCH_RESULT_LIMIT` (default five). The candidate
cap can also make filtered semantic results shorter than the requested limit.

Rebuild with the updated `fragment_indexer` to enable keyword features, then restart
the API to load the new snapshot. Startup checks the keyword index version, fields,
and fragment IDs. Older snapshots without FTS remain usable for unfiltered semantic
search; keyword-dependent requests return 503 with a rebuild message. The API never
builds or changes the keyword index. Keyword-only requests do not encode queries or
query Chroma, although application startup still initializes the semantic backend.

## Separate Chroma server

Leave `INDEX_SNAPSHOT` unset to use the existing architecture:

```bash
CHROMA_HOST=127.0.0.1 CHROMA_PORT=8001 \
SQLITE_DB_PATH=/path/to/matching/sqlite/sections.db \
  uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The SQLite file and Chroma server must contain the same completed release. This
mode opens SQLite read-only and uses the explicitly selected all-MiniLM-L6-v2
encoder for queries. The Chroma collection must exist; startup does not create an
empty replacement. Blocking Chroma I/O and model inference run in a worker thread,
so they do not block FastAPI's event loop. Clients are closed at shutdown.

| Variable                 | Default                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| `INDEX_SNAPSHOT`         | Unset: use the HTTP Chroma server                                    |
| `SQLITE_DB_PATH`         | `/data/sections.db`                                                  |
| `CHROMA_HOST`            | `localhost`                                                          |
| `CHROMA_PORT`            | `8000`                                                               |
| `CHROMA_SSL`             | `false`                                                              |
| `CHROMA_COLLECTION`      | `zalgorithm`                                                         |
| `SEARCH_RESULT_LIMIT`    | `5`                                                                  |
| `SEARCH_CANDIDATE_LIMIT` | `200`                                                                |
| `CORS_ORIGINS`           | `http://localhost:1313,http://127.0.0.1:1313,https://zalgorithm.com` |

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
HTML/form compatibility, keyword/Boolean syntax, whole-fragment filtering, RRF,
legacy snapshots, invalid queries and IDs, missing/deleted fragments,
backend errors, CORS, application cleanup, and both database modes.
