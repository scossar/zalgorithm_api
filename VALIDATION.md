# Validation — 2026-09-12

Keyword search, fragment-level inclusion/exclusion, and hybrid ranking were tested
with Python 3.13.15, SQLite FTS5, Chroma 1.5.9, and the real MiniLM encoder.

- All 45 API tests passed against a newly built site snapshot: 20 posts, 70 HTML
  fragments, 70 FTS rows, and 70 embedding chunks. Both local-snapshot and separate
  Chroma HTTP modes exercised keyword, filtered semantic, and hybrid requests.
- The same 45 tests passed against the older pre-FTS snapshot. Semantic requests
  remained functional; keyword-dependent requests returned the rebuild response.
- Real SQLite tests cover word/phrase matching, AND/OR/NOT, parentheses, prefixes,
  column filters, Unicode normalization, malformed expressions, and bound inputs.
- Filtering tests verify matches outside the semantic candidate window remain
  eligible, the complete keyword match set is used, full-fragment text is indexed,
  and empty/all-match cases are handled. RRF tests cover agreement, ties, missing
  results in one ranking, and duplicate chunks.
- HTTP tests cover all modes, filters, validation, HTML compatibility, and cache
  headers. Keyword-only queries bypass both encoding and Chroma retrieval.
- Snapshot hashes were unchanged after API use; temporary copies were cleaned up.
- The indexer's 28 tests passed, including real Hugo/Chroma rebuild, delete/restore,
  long-fragment FTS content, canonical post paths, and FTS integrity checks.
- The HTTP test invokes the Chroma launcher through the current interpreter because
  the existing launcher shebang still references the pre-rename project path.
- Chroma HTTP tests emitted ResourceWarnings for sockets during cleanup despite
  calling the client's close method. The temporary server was terminated normally.
- API Pyright reports the same four existing Chroma/startup typing diagnostics
  as the pre-change code; no new diagnostics were introduced. Indexer Pyright
  passed without errors or warnings.
- No Docker image build or website deployment was performed for this change.

# Validation — 2026-09-11

23 tests passed using Python 3.13.15, Chroma 1.5.9, FastAPI 0.128.0, and the actual
all-MiniLM-L6-v2 ONNX encoder.

- Existing form POST and fragment GET routes return the expected HTML.
- Both direct-snapshot mode and a separate Chroma HTTP server resolved search
  results to SQLite HTML. The HTTP test server used a temporary copy and loopback
  port, then shut down.
- Source snapshot database hashes were identical before and after API use.
- Tests verify distinct IDs with the same heading survive deduplication, duplicate
  embedding chunks trigger additional candidate retrieval, limits are enforced,
  missing fragments return 404, and backend failures return 503 without exposing
  internal details.
- Lifecycle tests verify resources close and temporary snapshot copies disappear.
- Both Docker Compose configurations passed `docker compose config --quiet`.
- A Docker image build and container deployment were not performed.

The original project environment was functional Python 3.11.13 with Chroma 1.4.0;
it was replaced to align the API with the new indexer's Python/Chroma versions.
The original files and environment were backed up before applying the update.
