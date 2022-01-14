""" Cache prediction for contextualized prediction """
import argparse
import json
import logging
import os
import pandas as pd

from tner import TransformersNER


def get_options():
    parser = argparse.ArgumentParser(description='Cache prediction for contextualized prediction')
    parser.add_argument('-f', '--csv-file', help='csv file to index', required=True, type=str)
    parser.add_argument('-t', '--column-text', help='column of text to index', required=True, type=str)
    parser.add_argument('-i', '--column-id', help='column of text to index', required=True, type=str)
    parser.add_argument('--base-model', help='base model', default=None, type=str)
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('-e', '--export-file', help='path to export the metric', required=True, type=str)
    parser.add_argument('--adapter', help='', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    # setup model
    if opt.adapter:
        assert opt.base_model is not None, 'adapter needs base model'
        classifier = TransformersNER(
            opt.base_model, max_length=opt.max_length, adapter=opt.adapter,
            adapter_model=opt.model)
    else:
        classifier = TransformersNER(opt.model, max_length=opt.max_length)
    classifier.eval()

    # run inference
    df = pd.read_csv(opt.csv_file, lineterminator='\n', index_col=0)
    text = df[opt.column_text].tolist()[:20]
    ids = df[opt.column_id].tolist()
    text = [i.split(' ') for i in text]
    out = classifier.predict(text, batch_size=opt.batch_size, decode_bio=True)

    # save the result
    os.makedirs(os.path.dirname(opt.export_file), exist_ok=True)
    with open(opt.export_file, 'w') as f:
        for _id, _out in zip(ids, out):
            f.write(json.dumps({'id': _id, 'entity': out}) + '\n')


if __name__ == '__main__':
    main()
