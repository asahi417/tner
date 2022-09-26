import logging
import argparse
import os

from datasets import load_dataset
from tner import TransformersNER


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_options():
    parser = argparse.ArgumentParser(description='Self labeling on dataset with finetuned NER model',)
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    parser.add_argument('--split', help='dataset to evaluate', default='2021.unlabelled', type=str)
    parser.add_argument('-e', '--export-file', help='path to export the metric', required=True, type=str)
    return parser.parse_args()


def main():
    opt = get_options()
    data = load_dataset('tweetner7', split=opt.split)['tokens']
    classifier = TransformersNER(opt.model, max_length=opt.max_length)
    classifier.eval()
    output = classifier.predict(data, batch_size=opt.batch_size)
    pred_list = output['prediction']
    os.makedirs(os.path.dirname(opt.export_file), exist_ok=True)
    with open(opt.export_file, 'w') as f:
        for _i, _l in zip(data, pred_list):
            for __i, __l in zip(_i, _l):
                __l = __l.replace(' ', '-')
                f.write(f'{__i} {__l}' + '\n')
            f.write('\n')


if __name__ == '__main__':
    main()
