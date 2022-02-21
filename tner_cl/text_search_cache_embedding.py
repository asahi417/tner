""" Cache embedding for contextualized prediction """
import argparse
import json
import logging
import os
import pandas as pd
from tner.text_searcher import SentenceEmbedding


def get_options():
    parser = argparse.ArgumentParser(description='Cache embedding for contextualized prediction')
    parser.add_argument('-f', '--csv-file', help='csv file to index', required=True, type=str)
    parser.add_argument('-t', '--column-text', help='column of text to index', required=True, type=str)
    parser.add_argument('-i', '--column-id', help='column of text to index', required=True, type=str)
    parser.add_argument('-m', '--model', help='model', default='sentence-transformers/all-mpnet-base-v1', type=str)
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('-e', '--export-file', help='path to export the metric', required=True, type=str)
    return parser.parse_args()


def main():
    opt = get_options()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    embedding_model = SentenceEmbedding(opt.model)
    # run inference
    df = pd.read_csv(opt.csv_file, lineterminator='\n')
    text = df[opt.column_text].tolist()
    ids = df[opt.column_id].tolist()
    embedding = embedding_model.embed(text, batch_size=opt.batch_size, show_progress_bar=True)
    embedding = embedding.tolist()

    # save the result
    os.makedirs(os.path.dirname(opt.export_file), exist_ok=True)
    with open(opt.export_file, 'w') as f:
        for _id, v in zip(ids, embedding):
            f.write(json.dumps({'id': _id, 'embedding': v}) + '\n')


if __name__ == '__main__':
    main()
