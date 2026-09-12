from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.backend import Backend, QueryTooLong, select_fragment_ids
from app.config import Settings
from app.main import create_app


class FakeCollection:
    def __init__(self, ids):
        self.ids = ids
        self.requested = []

    def count(self):
        return len(self.ids)

    def query(self, query_texts, n_results, include):
        self.requested.append(n_results)
        return {"metadatas": [[{"db_id": identifier, "section_heading": "Same heading"} for identifier in self.ids[:n_results]]]}


class RankingTests(unittest.TestCase):
    def test_distinct_fragments_with_same_heading_survive(self):
        self.assertEqual(select_fragment_ids(FakeCollection([1, 1, 2, 3]), "q", 3, 200), [1, 2, 3])

    def test_candidate_window_expands_after_duplicate_chunks(self):
        collection = FakeCollection([1] * 15 + [2, 3, 4])
        self.assertEqual(select_fragment_ids(collection, "q", 3, 200), [1, 2, 3])
        self.assertGreater(len(collection.requested), 1)

    def test_candidate_limit_caps_work(self):
        collection = FakeCollection([1] * 20 + [2])
        self.assertEqual(select_fragment_ids(collection, "q", 5, 15), [1])
        self.assertEqual(collection.requested[-1], 15)

    def test_empty_collection_does_not_query(self):
        collection = FakeCollection([])
        self.assertEqual(select_fragment_ids(collection, "q", 5, 200), [])
        self.assertEqual(collection.requested, [])

    def test_invalid_metadata_fails(self):
        for identifier in (None, "1", True, -1):
            with self.subTest(identifier=identifier), self.assertRaises(RuntimeError):
                select_fragment_ids(FakeCollection([identifier]), "q", 5, 200)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        database = Path(self.temporary.name) / "sections.db"
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("CREATE TABLE sections(id INTEGER PRIMARY KEY, html_heading TEXT, html_fragment TEXT)")
            db.executemany("INSERT INTO sections VALUES(?,?,?)", [(1, '<h2>Shared</h2>', '<p>One</p>'), (2, '<h2>Shared</h2>', '<p>Two</p>')])

        class FakeBackend:
            db_path = database
            closed = False
            error = None
            ids = [2, 1]
            queries = []

            def search(self, query, **options):
                self.queries.append(query)
                if self.error:
                    raise self.error
                return self.ids

            def close(self):
                self.closed = True

        self.backend = FakeBackend()
        self.app = create_app(Settings(), backend_factory=lambda settings: self.backend)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_form_query_returns_ranked_html_and_no_cache(self):
        response = self.client.post('/api/query', data={'query': '  example  '})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, '<h2>Shared</h2><p>Two</p><h2>Shared</h2><p>One</p>')
        self.assertEqual(self.backend.queries, ['example'])
        self.assertIn('text/html', response.headers['content-type'])
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_whitespace_query_returns_empty_html(self):
        response = self.client.post('/api/query', data={'query': '   '})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, '')
        self.assertEqual(self.backend.queries, [])

    def test_empty_results_return_empty_html(self):
        self.backend.ids = []
        self.assertEqual(self.client.post('/api/query', data={'query': 'example'}).text, '')

    def test_fragment_has_original_content_and_close_button(self):
        response = self.client.get('/api/fragment/1')
        self.assertEqual(response.status_code, 200)
        self.assertIn('<p>One</p>', response.text)
        self.assertIn('classList.toggle("hidden")', response.text)
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_missing_fragment_is_404(self):
        response = self.client.get('/api/fragment/999')
        self.assertEqual(response.status_code, 404)
        self.assertIn('no longer available', response.text)

    def test_invalid_fragment_ids_are_rejected(self):
        for identifier in ('0', '-1', 'abc', '1.2', '1%20OR%201=1'):
            self.assertEqual(self.client.get('/api/fragment/' + identifier).status_code, 422)

    def test_missing_and_oversized_form_input(self):
        self.assertEqual(self.client.post('/api/query', data={}).status_code, 422)
        self.assertEqual(self.client.post('/api/query', data={'query': 'a' * 4097}).status_code, 422)

    def test_query_token_limit(self):
        self.backend.error = QueryTooLong('Query exceeds the embedding model\'s 256-token limit')
        response = self.client.post('/api/query', data={'query': 'example'})
        self.assertEqual(response.status_code, 422)
        self.assertIn('256-token', response.text)

    def test_backend_exception_does_not_leak_details(self):
        self.backend.error = RuntimeError('/private/path/secret.db unavailable')
        with self.assertLogs('app.main', level='ERROR'):
            response = self.client.post('/api/query', data={'query': 'example'})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('secret', response.text)

    def test_missing_sqlite_reference_is_explicit_error(self):
        self.backend.ids = [999]
        with self.assertLogs('app.main', level='ERROR'):
            response = self.client.post('/api/query', data={'query': 'example'})
        self.assertEqual(response.status_code, 503)

    def test_database_unavailable_is_503_without_creating_a_file(self):
        self.backend.db_path = Path(self.temporary.name) / 'missing.db'
        with self.assertLogs('app.main', level='ERROR'):
            response = self.client.get('/api/fragment/1')
        self.assertEqual(response.status_code, 503)
        self.assertFalse(self.backend.db_path.exists())

    def test_cors_for_local_hugo(self):
        response = self.client.options('/api/query', headers={
            'Origin': 'http://localhost:1313', 'Access-Control-Request-Method': 'POST'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['access-control-allow-origin'], 'http://localhost:1313')

    def test_lifecycle_closes_backend(self):
        self.client.__exit__(None, None, None)
        self.assertTrue(self.backend.closed)


class SettingsTests(unittest.TestCase):
    def test_environment_configuration(self):
        with patch.dict(os.environ, {'INDEX_SNAPSHOT': '~/index/current', 'CHROMA_PORT': '8001', 'CORS_ORIGINS': 'http://localhost:1313'}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.snapshot, Path.home() / 'index/current')
        self.assertEqual(settings.chroma_port, 8001)
        self.assertEqual(settings.cors_origins, ('http://localhost:1313',))

    def test_invalid_limits_fail_early(self):
        for kwargs in ({'result_limit': 0}, {'result_limit': 201}, {'chroma_port': 0}):
            with self.assertRaises(ValueError):
                Settings(**kwargs)

    def test_missing_database_fails_startup_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'missing.db'
            with self.assertRaises(FileNotFoundError):
                Backend(Settings(sqlite_path=database))
            self.assertFalse(database.exists())


if __name__ == '__main__':
    unittest.main()
