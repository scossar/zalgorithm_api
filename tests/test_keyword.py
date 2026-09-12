"""Real SQLite matching with controlled semantic ordering, plus local Chroma filters."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import chromadb
from fastapi.testclient import TestClient

from app.backend import Backend, QueryTooLong, select_fragment_ids
from app.config import Settings
from app.keyword import (
    KeywordIndexUnavailable, SearchInputError, keyword_expression,
    matching_ids, reciprocal_rank_fusion, validate_keyword_index,
)
from app.main import create_app


class OrderedCollection:
    def __init__(self, identifiers):
        self.identifiers = identifiers
        self.calls = []

    def count(self):
        return len(self.identifiers)

    def query(self, query_texts, n_results, include, where=None):
        self.calls.append((query_texts, n_results, where))
        identifiers = self.identifiers
        if where:
            condition = where['db_id']
            if '$in' in condition:
                identifiers = [i for i in identifiers if i in condition['$in']]
            if '$nin' in condition:
                identifiers = [i for i in identifiers if i not in condition['$nin']]
        return {'metadatas': [[{'db_id': i} for i in identifiers[:n_results]]]}


class KeywordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'sections.db'
        self.documents = [
            (1, 'Learning', 'Basics', 'gradient descent reinforcement'),
            (2, 'Learning', 'Basics', 'gradient stochastic descent'),
            (3, 'Trees', 'Branches', 'decision trees'),
            (4, 'Learning', 'Methods', 'stochastic gradient descent'),
            (5, 'Logic', '', 'AND OR NOT'),
            (6, 'Café', '', 'naïve approach'),
            (7, 'Gradient', 'Empty', ''),
            (8, 'Long document', '', 'padding ' * 500 + 'lastsentinel'),
        ]
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE sections(id INTEGER PRIMARY KEY, html_heading TEXT, html_fragment TEXT, page_url TEXT)')
            db.execute("CREATE VIRTUAL TABLE sections_fts USING fts5(page_title, headings, body, tokenize='unicode61')")
            db.executemany('INSERT INTO sections_fts(rowid,page_title,headings,body) VALUES(?,?,?,?)', self.documents)
            db.executemany('INSERT INTO sections VALUES(?,?,?,?)', [
                (i, f'<h2>Heading {i}</h2>', f'<p>Body {i}</p>', f'/posts/{i}/')
                for i, *_ in self.documents
            ])
        self.backend = Backend.__new__(Backend)
        self.backend.settings = Settings(result_limit=3, candidate_limit=20)
        self.backend.db_path = self.path
        self.backend.has_keyword_index = True
        self.backend.client = None
        self.backend.temporary = None
        self.backend.tokenizer = Mock()
        self.backend.tokenizer.encode.return_value = SimpleNamespace(ids=[1])
        self.backend.collection = OrderedCollection([3] * 8 + [1, 2, 4, 7, 5, 6, 8])

    def matches(self, text, syntax='simple'):
        with closing(sqlite3.connect(self.path)) as db:
            return matching_ids(db, keyword_expression(text, syntax))

    def test_simple_and_phrases_and_case(self):
        self.assertEqual(self.matches('GRADIENT descent'), {1, 2, 4})
        self.assertEqual(self.matches('"gradient descent"'), {1, 4})
        self.assertEqual(self.matches('stochastic "gradient descent"'), {4})

    def test_simple_operators_are_literal_and_unicode_is_normalized(self):
        self.assertEqual(self.matches('OR NOT'), {5})
        self.assertEqual(self.matches('cafe naive'), {6})
        self.assertEqual(self.matches('---'), set())

    def test_fts5_boolean_parentheses_prefixes_and_columns(self):
        self.assertEqual(self.matches('(gradient OR trees) NOT reinforcement', 'fts5'), {2, 3, 4, 7})
        self.assertEqual(self.matches('grad* AND descent', 'fts5'), {1, 2, 4})
        self.assertEqual(self.matches('page_title:gradient', 'fts5'), {7})
        self.assertEqual(self.matches('"gradient descent"', 'fts5'), {1, 4})

    def test_invalid_syntax_is_a_query_error(self):
        for query, syntax in [('"open', 'simple'), ('""', 'simple'), ('a\x00b', 'simple'),
                              ('gradient OR', 'fts5'), ('NOT gradient', 'fts5'),
                              ('missing:gradient', 'fts5'), ('"open', 'fts5')]:
            with self.subTest(query=query), self.assertRaises(SearchInputError):
                keyword_expression(query, syntax)

    def test_queries_are_not_sql(self):
        self.assertEqual(self.matches("x'; DROP TABLE sections; --", 'simple'), set())
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM sections').fetchone()[0], 8)

    def test_full_fragment_and_empty_heading_are_searchable(self):
        self.assertEqual(self.matches('lastsentinel'), {8})
        self.assertIn(7, self.matches('gradient'))

    def test_keyword_does_not_encode_or_query_chroma(self):
        self.backend.tokenizer.encode.side_effect = AssertionError('Should not encode')
        self.assertEqual(self.backend.search('trees', mode='keyword'), [3])
        self.assertEqual(self.backend.collection.calls, [])
        self.assertEqual(self.backend.search('padding ' * 300, mode='keyword'), [8])

    def test_semantic_default_order_and_token_limit(self):
        self.assertEqual(self.backend.search('learning'), [3, 1, 2])
        self.backend.tokenizer.encode.return_value = SimpleNamespace(ids=list(range(257)))
        for mode in ('semantic', 'hybrid'):
            with self.assertRaises(QueryTooLong):
                self.backend.search('learning', mode=mode)

    def test_inclusion_filters_before_semantic_candidate_cap(self):
        self.backend.settings = Settings(result_limit=3, candidate_limit=3)
        self.assertEqual(self.backend.search('learning', include='gradient descent'), [1, 2, 4])
        self.assertEqual(self.backend.collection.calls[-1][2], {'db_id': {'$in': [1, 2, 4]}})

    def test_exclusion_filters_before_limit_and_combines_with_inclusion(self):
        self.assertEqual(self.backend.search('learning', exclude='trees'), [1, 2, 4])
        self.assertEqual(self.backend.collection.calls[-1][2], {'db_id': {'$nin': [3]}})
        self.assertEqual(self.backend.search('learning', include='gradient', exclude='reinforcement'), [2, 4, 7])

    def test_no_included_matches_short_circuits_chroma(self):
        self.assertEqual(self.backend.search('learning', include='absentword'), [])
        self.assertEqual(self.backend.search('learning', include='trees', exclude='trees'), [])
        self.assertEqual(self.backend.collection.calls, [])

    def test_excluding_everything_and_excluding_nothing(self):
        all_terms = 'gradient OR trees OR logic OR cafe OR padding'
        self.assertEqual(self.backend.search('learning', exclude=all_terms, keyword_syntax='fts5'), [])
        self.assertEqual(self.backend.search('learning', exclude='absentword'), [3, 1, 2])

    def test_filter_matching_is_not_truncated_to_ranked_limit(self):
        self.backend.settings = Settings(result_limit=1, candidate_limit=1)
        self.backend.collection = OrderedCollection([4, 2, 1, 7])
        self.assertEqual(self.backend.search('learning', include='gradient'), [4])
        self.assertEqual(self.backend.collection.calls[-1][2], {'db_id': {'$in': [1, 2, 4, 7]}})

    def test_keyword_filters_before_limit(self):
        self.backend.settings = Settings(result_limit=1)
        self.assertEqual(self.backend.search('gradient', mode='keyword', include='stochastic', exclude='"gradient descent"'), [2])

    def test_hybrid_fuses_fragments_and_allows_separate_keyword_query(self):
        self.backend.collection = OrderedCollection([3, 3, 1])
        self.assertEqual(self.backend.search('learning', mode='hybrid', keyword_query='reinforcement'), [1, 3])
        self.assertEqual(self.backend.collection.calls[-1][0], ['learning'])
        self.assertEqual(self.backend.search('learning', mode='hybrid', keyword_query='trees'), [3, 1])

    def test_hybrid_filters_both_rankings(self):
        self.assertEqual(set(self.backend.search('gradient', mode='hybrid', include='stochastic', exclude='reinforcement')), {2, 4})
        self.assertEqual(self.backend.search('learning', mode='hybrid', keyword_query='absentword'), [3, 1, 2])

    def test_bad_keyword_override_is_rejected(self):
        for mode, override in [('semantic', 'trees'), ('keyword', 'trees'), ('hybrid', ' ' )]:
            with self.subTest(mode=mode), self.assertRaises(SearchInputError):
                self.backend.search('learning', mode=mode, keyword_query=override)

    def test_old_snapshots_still_allow_semantic_queries(self):
        self.backend.has_keyword_index = False
        self.assertEqual(self.backend.search('learning'), [3, 1, 2])
        for options in ({'mode': 'keyword'}, {'mode': 'hybrid'}, {'include': 'trees'}, {'exclude': 'trees'}):
            with self.subTest(options=options), self.assertRaises(KeywordIndexUnavailable):
                self.backend.search('learning', **options)

    def test_snapshot_manifest_and_ids_are_checked(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            manifest = {'version': 1, 'tokenizer': 'unicode61', 'fragments': 8}
            self.assertTrue(validate_keyword_index(db, manifest))
            for overrides in ({'version': 2}, {'fragments': 7}, {'tokenizer': 'porter'}):
                with self.assertRaises(ValueError):
                    validate_keyword_index(db, manifest | overrides)
            db.execute('DELETE FROM sections_fts WHERE rowid=8')
            with self.assertRaisesRegex(ValueError, 'IDs differ'):
                validate_keyword_index(db, manifest)
            db.execute('DROP TABLE sections_fts')
            self.assertFalse(validate_keyword_index(db))
            with self.assertRaisesRegex(ValueError, 'missing keyword index'):
                validate_keyword_index(db, manifest)

    def test_rrf_rewards_agreement_and_deduplicates(self):
        self.assertEqual(reciprocal_rank_fusion([1, 1, 2], [2, 3], limit=3), [2, 1, 3])
        self.assertEqual(reciprocal_rank_fusion([], [3, 2], limit=3), [3, 2])
        self.assertEqual(reciprocal_rank_fusion([3], [2], limit=3), [2, 3])

    def test_http_modes_filters_errors_and_html(self):
        with TestClient(create_app(self.backend.settings, backend_factory=lambda _: self.backend)) as client:
            for data in (
                {'query': 'trees', 'mode': 'keyword'},
                {'query': 'learning', 'include': 'trees'},
                {'query': 'trees', 'mode': 'hybrid', 'include': 'trees'},
                {'query': 'learning', 'include': 'trees OR absentword', 'keyword_syntax': 'fts5'},
            ):
                response = client.post('/api/query', data=data)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.text, '<h2>Heading 3</h2><p>Body 3</p>')
                self.assertEqual(response.headers['cache-control'], 'no-store')
            for extra in (
                {'mode': 'bad'}, {'keyword_syntax': 'bad'}, {'keyword_query': 'trees'},
                {'mode': 'keyword', 'query': '"open'},
                {'mode': 'keyword', 'keyword_syntax': 'fts5', 'query': 'gradient OR'},
                {'include': 'a' * 4097}, {'exclude': 'a' * 4097},
                {'mode': 'hybrid', 'keyword_query': 'a' * 4097},
            ):
                response = client.post('/api/query', data={'query': 'learning'} | extra)
                self.assertEqual(response.status_code, 422, response.text)
            response = client.post('/api/query', data={'query': 'padding ' * 300, 'mode': 'keyword'})
            self.assertEqual(response.status_code, 200)
            self.backend.has_keyword_index = False
            response = client.post('/api/query', data={'query': 'trees', 'mode': 'keyword'})
            self.assertEqual(response.status_code, 503)
            self.assertIn('rebuilt', response.text)


class RealChromaFilterTests(unittest.TestCase):
    def test_metadata_filters_and_empty_results_with_real_chroma(self):
        with tempfile.TemporaryDirectory() as directory:
            client = chromadb.PersistentClient(path=directory)
            try:
                collection = client.create_collection('keyword-filter-test', embedding_function=None)
                collection.add(ids=['a', 'b', 'c', 'd'], embeddings=[[1., 0.], [.9, .1], [.8, .2], [.7, .3]], metadatas=[{'db_id': 1}, {'db_id': 1}, {'db_id': 2}, {'db_id': 3}])

                class EncodedCollection:
                    def count(self):
                        return collection.count()

                    def query(self, **kwargs):
                        kwargs.pop('query_texts')
                        return collection.query(query_embeddings=[[1., 0.]], **kwargs)

                adapter = EncodedCollection()
                self.assertEqual(select_fragment_ids(adapter, 'q', 3, 20, {'db_id': {'$in': [2, 3]}}), [2, 3])
                self.assertEqual(select_fragment_ids(adapter, 'q', 3, 20, {'db_id': {'$nin': [1, 2, 3]}}), [])
                self.assertEqual(select_fragment_ids(adapter, 'q', 3, 20, {'db_id': {'$in': [999]}}), [])
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
