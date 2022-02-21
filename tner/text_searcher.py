""" Text searcher for contextualised prediction. """
import json
import signal
import logging
import os
from datetime import datetime
from typing import List

import pandas as pd
from tqdm import tqdm

from whoosh import query
from whoosh.fields import Schema, TEXT, DATETIME, ID, NUMERIC
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.collectors import TimeLimitCollector, TimeLimit
from sentence_transformers import SentenceTransformer


class SentenceEmbedding:

    def __init__(self, model: str = None):
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, list_of_sentence, batch_size: int = 16, show_progress_bar: bool = False):
        return self.model.encode(list_of_sentence, batch_size=batch_size, show_progress_bar=False)


def handler(signum, frame):
    raise Exception("TIMEOUT")


signal.signal(signal.SIGALRM, handler)


class WhooshSearcher:

    def __init__(self,
                 index_path: str = None,
                 embedding_path: str = None,
                 column_text: str = 'text',
                 column_id: str = 'id',
                 column_datetime: str = 'datetime',
                 embedding_model: str = None):

        self.column_id = column_id
        self.column_datetime = column_datetime
        self.column_text = column_text
        self.index_path = index_path
        schema_config = {
            self.column_text: TEXT(stored=True),
            self.column_id: ID(stored=True),
            self.column_datetime: DATETIME(stored=True),
        }
        if embedding_model is not None:
            self.embedding_model = SentenceEmbedding(embedding_model)
        else:
            self.embedding_model = None

        self.indexer = self.instantiate_indexer(index_path, schema_config)

        self.embedding_path = embedding_path
        if self.embedding_path is not None and os.path.exists(self.embedding_path):
            with open(self.embedding_path) as f:
                self.embedding_cache = json.load(f)
        else:
            self.embedding_cache = {}

    @staticmethod
    def instantiate_indexer(index_dir, schema_config):
        schema = Schema(**schema_config)
        if os.path.exists(index_dir):
            logging.info('load index from {}'.format(index_dir))
            ix = open_dir(index_dir, schema=schema)
            logging.info('{} documents'.format(ix.doc_count()))
            assert all(i in ix.schema.names() for i in schema.names())
            return ix
        else:
            logging.info('create new index at {}'.format(index_dir))
            os.makedirs(index_dir, exist_ok=True)
            return create_in(index_dir, schema)

    def whoosh_indexing(self,
                        csv_file: str,
                        column_text: str = None,
                        column_id: str = None,
                        column_datetime: str = None,
                        datetime_format: str = '%Y-%m-%d',
                        batch_size: int = 16,
                        chunk_size: int = 4):
        column_text = column_text if column_text is not None else self.column_text
        column_id = column_id if column_id is not None else self.column_id
        column_datetime = column_datetime if column_datetime is not None else self.column_datetime
        df = pd.read_csv(csv_file, lineterminator='\n')

        # compute embeddings
        if self.embedding_model is not None:
            text = df[column_text].tolist()
            _id = df[column_id].tolist()
            # filter out existing embeddings
            text = [t for i, t in zip(_id, text) if i not in self.embedding_cache]
            _id = [i for i in _id if i not in self.embedding_cache]
            if len(text) > 0:
                embedding = self.__precompute_embedding(text, batch_size=batch_size, chunk_size=chunk_size)
                self.embedding_cache.update({i: _e for i, _e in zip(_id, embedding)})
                os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
                with open(self.embedding_path, 'w') as f:
                    json.dump(self.embedding_cache, f)
        writer = self.indexer.writer()
        logging.info('indexing data')
        try:
            for n, (_, _df) in tqdm(list(enumerate(df.iterrows()))):
                _dict = _df.to_dict()
                data = {self.column_text: str(_dict[column_text])}
                if column_id is not None and self.column_text is not None:
                    data[self.column_id] = str(_dict[column_id])
                if column_datetime is not None and self.column_datetime is not None:
                    data[self.column_datetime] = datetime.strptime(_dict[column_datetime], datetime_format),
                writer.add_document(**data)
            writer.commit()
        except Exception as e:
            logging.exception('Error occurred while indexing.')
            writer.cancel()
            raise e

    def __precompute_embedding(self, list_text, batch_size: int = 16, chunk_size: int = 4):
        tmp_file = '.tmp.embedding.txt'
        bucket = []
        logging.info('computing sentence embedding')
        with open(tmp_file, 'w') as f:
            for i in tqdm(list_text):
                bucket.append(i)
                if len(bucket) == chunk_size * batch_size:
                    embedding = self.embedding_model.embed(bucket)
                    for v in embedding.tolist():
                        f.write(','.join([str(_v) for _v in v]) + '\n')
                    bucket = []
            if len(bucket) != 0:
                embedding = self.embedding_model.embed(bucket)
                for v in embedding.tolist():
                    f.write(','.join([str(_v) for _v in v]) + '\n')
        with open(tmp_file) as f:
            embeddings = [[float(v) for v in i.split(',')] for i in f.read().split('\n') if len(i) > 0]
        os.remove(tmp_file)
        return embeddings

    def search(self,
               query_string: str,
               limit: int = 10,
               target_field: str = None,
               return_field: List or str = None,
               timeout: int = 1000,
               date_range_start=None,
               date_range_end=None):
        target_field = self.column_text if target_field is None else target_field
        return_field = self.column_text if return_field is None else return_field
        if type(return_field) is str:
            return_field = [return_field]
        return_field += ['score', self.column_id]
        q = QueryParser(target_field, self.indexer.schema).parse(str(query_string))
        if date_range_start is None and date_range_end is None:
            q_date_range = None
        else:
            assert self.column_datetime is not None
            q_date_range = query.DateRange(self.column_datetime, start=date_range_start, end=date_range_end)
        with self.indexer.searcher() as searcher:
            if q_date_range is not None:
                # Currently, the TimeLimitCollector doesn't work with the DateRange
                # date range does not work with the collector
                signal.alarm(timeout)
                try:
                    results = searcher.search(q, limit=limit, filter=q_date_range)
                except Exception as exc:
                    logging.warning("Search took too long, aborting!: {}".format(query_string))
                    results = []
                signal.alarm(0)
            else:
                # Get a collector object
                c = searcher.collector(limit=limit, filter=q_date_range)
                # Wrap it in a TimeLimitedCollector and set the time limit to 10 seconds
                tlc = TimeLimitCollector(c, timelimit=timeout)
                try:
                    searcher.search_with_collector(q, tlc)
                except TimeLimit:
                    logging.warning("Search took too long, aborting!: {}".format(query_string))
                results = tlc.results()

            results = [{field: r[field] if field != 'score' else r.score for field in return_field}
                       for n, r in enumerate(results)]
            if self.embedding_cache is not None or self.embedding_model is not None:
                for result in results:
                    _id = result[self.column_id]
                    try:
                        v = self.embedding_cache[_id]
                    except KeyError:
                        if self.embedding_model is not None:
                            v = self.embedding_model.embed(result[self.column_text])
                        else:
                            raise ValueError('no embedding found')
                    result['embedding'] = v
            return results


