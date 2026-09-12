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
