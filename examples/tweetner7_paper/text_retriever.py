""" Text searcher for contextualised prediction """
import argparse
import logging
import json
import signal
import logging
import os
from datetime import datetime
from typing import List
from tqdm import tqdm
from time import time
from os.path import join as pj

import pandas as pd
from whoosh import query
from whoosh.fields import Schema, TEXT, DATETIME, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.collectors import TimeLimitCollector, TimeLimit
from sentence_transformers import SentenceTransformer

from tner import TransformersNER

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class SentenceEmbedding:

    def __init__(self, model: str = None):
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, list_of_sentence, batch_size: int = 16, show_progress_bar: bool = False):
        return self.model.encode(list_of_sentence, batch_size=batch_size, show_progress_bar=show_progress_bar)


def handler(signum, frame):
    raise Exception("TIMEOUT")


signal.signal(signal.SIGALRM, handler)


class Retriever:
    """ Text search engine with sentence-embedding/ner-prediction """

    def __init__(self,
                 index_dir: str,
                 ner_model: str = None,
                 embedding_model: str = None,
                 max_length: int = 128,
                 column_text: str = 'text',
                 column_id: str = 'id',
                 column_datetime: str = 'datetime'):
        # config
        self.ner_model_name = ner_model
        self.embedding_model_name = embedding_model
        self.max_length = max_length
        self.index_dir = index_dir
        self.column_id = column_id
        self.column_datetime = column_datetime
        self.column_text = column_text
        self.embedding_model = None
        self.ner_model = None
        self.indexer = None
        if index_dir is not None:
            # search engine
            schema = Schema(**{
                self.column_text: TEXT(stored=True),
                self.column_id: ID(stored=True),
                self.column_datetime: DATETIME(stored=True)
            })
            index_search = pj(index_dir, 'search_index')
            if os.path.exists(index_search):
                logging.info(f'load index from {index_search}')
                self.indexer = open_dir(index_search, schema=schema)
                logging.info(f'{self.indexer.doc_count()} documents')
                assert all(i in self.indexer.schema.names() for i in schema.names())
            else:
                os.makedirs(index_search, exist_ok=True)
                logging.info(f'create new index at {index_search}')
                self.indexer = create_in(index_search, schema)

        # embedding
        self.embedding_cache = {}
        self.embedding_path = pj(index_dir, 'embedding.json')
        if os.path.exists(self.embedding_path):
            with open(self.embedding_path) as f:
                self.embedding_cache = json.load(f)

        # ner prediction
        self.ner_prediction_cache = {}
        self.ner_prediction_path = pj(index_dir, 'ner_prediction.json')
        if os.path.exists(self.ner_prediction_path):
            with open(self.ner_prediction_path) as f:
                self.ner_prediction_cache = json.load(f)

    @property
    def ids_embedding(self):
        return list(self.embedding_cache.keys())

    @property
    def ids_ner(self):
        return list(self.ner_prediction_cache.keys())

    def get_ner(self, list_tokenized_sentence, batch_size: int = None):
        if self.ner_model is None:
            assert self.ner_model_name is not None
            self.ner_model = TransformersNER(self.ner_model_name, max_length=self.max_length)
        output = self.ner_model.predict(list_tokenized_sentence, batch_size=batch_size, return_loader=False)
        return output['entity_prediction']

    def get_embedding(self, list_sentence, batch_size: int = 128):
        list_sentence = [' '.join(i) if type(i) is not list else i for i in list_sentence]
        if self.embedding_model is None:
            assert self.embedding_model_name is not None
            self.embedding_model = SentenceEmbedding(self.embedding_model_name)
        return self.embedding_model.embed(list_sentence, batch_size=batch_size)

    def indexing(self,
                 csv_file: str,
                 column_text: str = None,
                 column_id: str = None,
                 column_datetime: str = None,
                 datetime_format: str = '%Y-%m-%d',
                 batch_size: int = 16,
                 chunk_size: int = 4):
        # setup
        column_text = column_text if column_text is not None else self.column_text
        column_id = column_id if column_id is not None else self.column_id
        column_datetime = column_datetime if column_datetime is not None else self.column_datetime
        df = pd.read_csv(csv_file, lineterminator='\n')
        df = df.dropna()

        # compute ner
        tmp = [(t, i) for i, t in zip(df[column_id].tolist(), df[column_text].tolist()) if
               str(i) not in self.ner_prediction_cache]
        if len(tmp) != 0:
            text, _id = list(zip(*tmp))
            if len(text) > 0:
                logging.info('computing NER')
                ner = self.get_ner(text, batch_size=batch_size)
                self.ner_prediction_cache.update({str(i): _ner for i, _ner in zip(_id, ner)})
                os.makedirs(os.path.dirname(self.ner_prediction_path), exist_ok=True)
                with open(self.ner_prediction_path, 'w') as f:
                    json.dump(self.ner_prediction_cache, f)

        # compute embeddings
        tmp = [(t, i) for i, t in zip(df[column_id].tolist(), df[column_text].tolist()) if
               str(i) not in self.embedding_cache]
        if len(tmp) != 0:
            text, _id = list(zip(*tmp))
            if len(text) > 0:
                tmp_file = f'tmp.embedding.{time()}.txt'
                logging.info(f'computing sentence embedding: cache file at {tmp_file}')
                bucket = []
                with open(tmp_file, 'w') as f:
                    for i in tqdm(text):
                        bucket.append(i)
                        if len(bucket) == chunk_size * batch_size:
                            embedding = self.get_embedding(bucket)
                            f.write('\n'.join([','.join([str(_v) for _v in v]) for v in embedding.tolist()]) + '\n')
                            bucket = []
                    if len(bucket) != 0:
                        embedding = self.get_embedding(bucket)
                        f.write('\n'.join([','.join([str(_v) for _v in v]) for v in embedding.tolist()]) + '\n')
                with open(tmp_file) as f:
                    embeddings = [[float(v) for v in i.split(',')] for i in f.read().split('\n') if len(i) > 0]
                os.remove(tmp_file)
                self.embedding_cache.update({str(i): _e for i, _e in zip(_id, embeddings)})
                os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
                with open(self.embedding_path, 'w') as f:
                    json.dump(self.embedding_cache, f)

        # index data
        assert self.indexer is not None
        writer = self.indexer.writer()
        try:
            for n, (_, _df) in tqdm(list(enumerate(df.iterrows()))):
                _dict = _df.to_dict()
                data = {self.column_text: str(_dict[column_text])}
                if column_id is not None and self.column_text is not None:
                    data[self.column_id] = str(_dict[column_id])
                if column_datetime is not None and self.column_datetime is not None:
                    data[self.column_datetime] = datetime.strptime(_dict[column_datetime], datetime_format)
                writer.add_document(**data)
            writer.commit()
        except Exception as e:
            logging.exception('Error occurred while indexing.')
            writer.cancel()
            raise e

    def search(self,
               query_string: str,
               limit: int = 10,
               target_field: str = None,
               return_field: List or str = None,
               timeout: int = 1000,
               date_range_start=None,
               date_range_end=None,
               no_inference: bool = True):
        assert self.indexer is not None
        target_field = self.column_text if target_field is None else target_field
        return_field = ['score', self.column_id, self.column_text, self.column_datetime] if return_field is None else return_field
        if type(return_field) is str:
            return_field = [return_field]
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
                    logging.warning(f"Search took too long, aborting!: {query_string}")
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
                    logging.warning(f"Search took too long, aborting!: {query_string}")
                results = tlc.results()

            results = [{field: r[field] if field != 'score' else r.score for field in return_field}
                       for n, r in enumerate(results)]
            id_list = []
            new_result = []
            for result in results:
                _id = result[self.column_id]
                if _id in id_list:
                    continue
                id_list.append(_id)
                try:
                    result['embedding'] = self.embedding_cache[str(_id)]
                except KeyError:
                    assert not no_inference, f'{str(_id)} not found in embedding cache'
                    logging.warning('running embedding model on demand: better to cache beforehand')
                    result['embedding'] = self.get_embedding(result[self.column_text])
                try:
                    result['ner'] = self.ner_prediction_cache[str(_id)]
                except KeyError:
                    assert not no_inference, f'{str(_id)} not found in ner cache'
                    logging.warning('running NER model on demand: better to cache beforehand')
                    result['ner'] = self.get_ner(result[self.column_text])
                new_result.append(result)
            return new_result




def arguments(parser):
    parser.add_argument('-i', '--index-dir', help='path to index directory', required=True, type=str)
    return parser


def main_indexing():
    parser = argparse.ArgumentParser(description='Index document for contextualized prediction.')
    parser = arguments(parser)
    parser.add_argument('-n', '--ner', help='ner model', required=True, type=str)
    parser.add_argument('-s', '--sentence', help='sentence transformer model',
                        default='sentence-transformers/all-mpnet-base-v1', type=str)
    parser.add_argument('-f', '--csv-file', help='csv file to index', default=None, type=str)
    parser.add_argument('--column-text', help='column of text to index', default='tweet', type=str)
    parser.add_argument('--column-id', help='column of text to index', default='id', type=str)
    parser.add_argument('--column-datetime', help='column of text to index', default='created_at', type=str)
    parser.add_argument('--datetime-format', help='datetime format', default='%Y-%m-%d', type=str)
    parser.add_argument('-b', '--batch-size', help='batch size of embedding/ner', default=128, type=int)
    parser.add_argument('-c', '--chunk-size', help='batch size of embedding', default=100, type=int)
    args = parser.parse_args()
    retriever = Retriever(index_dir=args.index_dir, ner_model=args.ner, embedding_model=args.sentence)
    retriever.indexing(
        csv_file=args.csv_file,
        column_text=args.column_text,
        column_id=args.column_id,
        column_datetime=args.column_datetime,
        datetime_format=args.datetime_format,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )


def main_interactive():
    parser = argparse.ArgumentParser(description='Interactive mode.')
    parser = arguments(parser)
    parser.add_argument('--datetime-range-start', help='', default=None, type=str)
    parser.add_argument('--datetime-range-end', help='', default=None, type=str)
    parser.add_argument('--datetime-format', help='datetime format', default='%Y-%m-%d', type=str)
    parser.add_argument('-q', '--query', help='test query', default='Peaky Blinders', type=str)
    args = parser.parse_args()
    retriever = Retriever(index_dir=args.index_dir)
    datetime_range_start = None
    datetime_range_end = None
    if args.datetime_range_start is not None:
        datetime_range_start = datetime.strptime(args.datetime_range_start, args.datetime_format)
    if args.datetime_range_end is not None:
        datetime_range_end = datetime.strptime(args.datetime_range_end, args.datetime_format)
    out = retriever.search(args.query, date_range_end=datetime_range_end, date_range_start=datetime_range_start)
    print(out)
    while True:
        _inp = input('query >>>')
        if _inp == 'q':
            break
        elif _inp == '':
            continue
        else:
            out = retriever.search(_inp, date_range_end=datetime_range_end, date_range_start=datetime_range_start)
            print('# {} documents #'.format(len(out)))
            print('#' * 100)
            for n, i in enumerate(out):
                print(' *** {} *** \n{}'.format(n, i))
                print('#' * 100)
                print(i.keys())


if __name__ == '__main__':
    main_indexing()