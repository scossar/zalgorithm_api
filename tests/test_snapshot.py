"""Optional integration test against a completed fragment-indexer snapshot."""
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3
import unittest

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

SNAPSHOT = os.getenv('TEST_INDEX_SNAPSHOT')


@unittest.skipUnless(SNAPSHOT, 'Set TEST_INDEX_SNAPSHOT to test real SQLite/Chroma/ONNX integration')
class SnapshotTests(unittest.TestCase):
    def test_real_query_fragment_lookup_and_source_immutability(self):
        snapshot = Path(SNAPSHOT).expanduser().resolve()

        def hashes():
            paths = list((snapshot / 'chroma').rglob('*')) + list((snapshot / 'sqlite').rglob('*'))
            return {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

        before = hashes()
        with closing(sqlite3.connect(f'file:{snapshot}/sqlite/sections.db?mode=ro', uri=True)) as db:
            db_id, heading, body = db.execute('SELECT id, html_heading, html_fragment FROM sections ORDER BY id LIMIT 1').fetchone()
        app = create_app(Settings(snapshot=snapshot))
        with TestClient(app) as client:
            working_path = app.state.backend.db_path.parent
            self.assertNotEqual(working_path, snapshot)
            response = client.get(f'/api/fragment/{db_id}')
            self.assertEqual(response.status_code, 200)
            self.assertIn(heading + body, response.text)
            response = client.post('/api/query', data={'query': 'gradient descent'})
            self.assertEqual(response.status_code, 200)
            self.assertIn('article-fragment', response.text)
            self.assertNotIn('hx-get=', response.text)
            response = client.post('/api/query', data={'query': 'hello ' * 300})
            self.assertEqual(response.status_code, 422)
        self.assertFalse(working_path.exists())
        self.assertEqual(hashes(), before)


if __name__ == '__main__':
    unittest.main()
