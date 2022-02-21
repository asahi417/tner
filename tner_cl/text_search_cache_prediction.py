""" Cache prediction for contextualized prediction """
import argparse
import json
import logging
import pandas as pd
from tner import TransformersNER


def get_options():
    parser = argparse.ArgumentParser(description='Cache prediction for contextualized prediction')
    parser.add_argument('-f', '--file', help='csv file to index', required=True, type=str)
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

    df = pd.read_csv(opt.file, lineterminator='\n')
    text = df[opt.column_text].tolist()
    _id = df[opt.column_id].tolist()
    text = [i.split(' ') for i in text]
    input(len(text))

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
    out = classifier.predict(text, batch_size=opt.batch_size, decode_bio=True)
    _, tmp_decode = out[0]
    with open(opt.export_file, 'w') as f:
        json.dump({str(_i): o for o, _i in zip(tmp_decode, _id)}, f)


if __name__ == '__main__':
    main()
