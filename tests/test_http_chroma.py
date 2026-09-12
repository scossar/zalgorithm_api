"""Exercise the existing separate-Chroma-server architecture on loopback only."""
from contextlib import closing
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

SNAPSHOT = os.getenv('TEST_INDEX_SNAPSHOT')


@unittest.skipUnless(SNAPSHOT, 'Set TEST_INDEX_SNAPSHOT to test a local Chroma HTTP server')
class HttpChromaTests(unittest.TestCase):
    def test_query_and_fragment_with_separate_chroma_server(self):
        snapshot = Path(SNAPSHOT).expanduser().resolve()
        with tempfile.TemporaryDirectory() as directory:
            working = Path(directory)
            shutil.copytree(snapshot / 'chroma', working / 'chroma')
            shutil.copy2(snapshot / 'sqlite/sections.db', working / 'sections.db')
            with closing(sqlite3.connect(working / 'sections.db')) as db:
                fragment_id = db.execute('SELECT id FROM sections ORDER BY id LIMIT 1').fetchone()[0]
            with closing(socket.socket()) as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            with (working / 'server.log').open('w+') as logfile:
                server = subprocess.Popen([str(Path(sys.executable).parent / 'chroma'), 'run',
                                           '--path', str(working / 'chroma'), '--host', '127.0.0.1',
                                           '--port', str(port)], stdout=logfile, stderr=subprocess.STDOUT)
                try:
                    ready = False
                    with httpx.Client(timeout=0.5, trust_env=False) as client:
                        for _ in range(100):
                            if server.poll() is not None:
                                break
                            try:
                                if client.get(f'http://127.0.0.1:{port}/api/v2/heartbeat').status_code == 200:
                                    ready = True
                                    break
                            except httpx.HTTPError:
                                pass
                            time.sleep(0.1)
                    logfile.flush()
                    logfile.seek(0)
                    self.assertTrue(ready, logfile.read())
                    app = create_app(Settings(sqlite_path=working / 'sections.db', chroma_host='127.0.0.1', chroma_port=port))
                    with TestClient(app) as client:
                        response = client.post('/api/query', data={'query': 'gradient descent'})
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertIn('article-fragment', response.text)
                        self.assertEqual(client.get(f'/api/fragment/{fragment_id}').status_code, 200)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
