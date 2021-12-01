""" Fine-tune transformers on NER dataset """
import argparse
import logging
import json

from tner.grid_searcher import evaluate


def get_options():
    parser = argparse.ArgumentParser(description='Fine-tune transformers on NER dataset')
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('-d', '--data', help='dataset to evaluate', default='wnut2017', type=str)
    parser.add_argument('-e', '--export-dir', help='path to export the metric', default=None, type=str)
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    metric = evaluate(
        model=opt.model,
        export_dir=opt.export_dir,
        batch_size=opt.batch_size,
        max_length=opt.max_length,
        data=opt.data,
        lower_case=opt.lower_case
    )

    print(json.dumps(metric, indent=4))


if __name__ == '__main__':
    main()
