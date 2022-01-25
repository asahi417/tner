""" Text searcher for contextualised prediction. """
import signal
import logging
import os
from datetime import datetime
from typing import List

import pandas as pd
from tqdm import tqdm

from whoosh import query
from whoosh.fields import Schema, TEXT, DATETIME, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.collectors import TimeLimitCollector, TimeLimit


def handler(signum, frame):
    raise Exception("TIMEOUT")


signal.signal(signal.SIGALRM, handler)


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


class WhooshSearcher:

    def __init__(self,
                 index_path: str,
                 column_text: str = 'text',
                 column_id: str = 'id',
                 column_datetime: str = 'datetime'):

        self.column_id = column_id
        self.column_datetime = column_datetime
        self.column_text = column_text

        schema_config = {
            self.column_text: TEXT(stored=True),
            self.column_id: ID(stored=True),
            self.column_datetime: DATETIME(stored=True)
        }
        self.indexer = instantiate_indexer(index_path, schema_config)

    def whoosh_indexing(self,
                        csv_file: str,
                        column_text: str = None,
                        column_id: str = None,
                        column_datetime: str = None,
                        datetime_format: str = '%Y-%m-%d'):
        writer = self.indexer.writer()
        column_text = column_text if column_text is not None else self.column_text
        column_id = column_id if column_id is not None else self.column_id
        column_datetime = column_datetime if column_datetime is not None else self.column_datetime
        df = pd.read_csv(csv_file, lineterminator='\n', index_col=0)
        try:
            for _, _df in tqdm(df.iterrows()):
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

    def search(self, query_string: str, limit: int = 10, target_field: str = None, return_field: List or str = None,
               timeout: int = None, date_range_start=None, date_range_end=None):
        target_field = self.column_text if target_field is None else target_field
        return_field = self.column_text if return_field is None else return_field
        if type(return_field) is str:
            return_field = [return_field]
        return_field += ['score']
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
            return results


