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


def check_keyword_modes(test, backend, client):
    if not backend.has_keyword_index:
        response = client.post('/api/query', data={'query': 'gradient', 'mode': 'keyword'})
        test.assertEqual(response.status_code, 503)
        return
    with closing(sqlite3.connect(backend.db_path)) as db:
        title = db.execute('SELECT page_title FROM sections_fts ORDER BY rowid LIMIT 1').fetchone()[0]
        expression = 'page_title:"' + title.replace('"', '""') + '"'
        matches = {r[0] for r in db.execute('SELECT rowid FROM sections_fts WHERE sections_fts MATCH ?', (expression,))}
        test.assertTrue(matches)
        html_by_id = {i: h + b for i, h, b in db.execute('SELECT id, html_heading, html_fragment FROM sections')}
        # The canonical post path remains directly retrievable for every FTS hit.
        for identifier in matches:
            url = db.execute('SELECT page_url FROM sections WHERE id=?', (identifier,)).fetchone()[0]
            test.assertTrue(url.startswith('/'))
    for mode, query, options in (
        ('keyword', expression, {}),
        ('semantic', 'gradient descent', {'include': expression}),
        ('semantic', 'gradient descent', {'exclude': expression}),
        ('hybrid', 'gradient descent', {'keyword_query': expression}),
        ('hybrid', 'gradient descent', {'keyword_query': expression, 'include': expression}),
    ):
        ids = backend.search(query, mode=mode, keyword_syntax='fts5', **options)
        if mode == 'keyword' or 'include' in options:
            test.assertTrue(ids)
            test.assertTrue(set(ids) <= matches)
        if 'exclude' in options:
            test.assertTrue(set(ids).isdisjoint(matches))
        test.assertEqual(len(ids), len(set(ids)))
        response = client.post('/api/query', data={
            'mode': mode, 'query': query, 'keyword_syntax': 'fts5', **options,
        })
        test.assertEqual(response.status_code, 200, response.text)
        test.assertEqual(response.text, ''.join(html_by_id[i] for i in ids))


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
            check_keyword_modes(self, app.state.backend, client)
        self.assertFalse(working_path.exists())
        self.assertEqual(hashes(), before)


if __name__ == '__main__':
    unittest.main()
