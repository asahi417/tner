""" Fine-tune transformers on NER dataset """
import argparse
import logging
import os
import json

from tner import TransformersNER, get_dataset


def get_options():
    parser = argparse.ArgumentParser(description='Fine-tune transformers on NER dataset')
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-d', '--dataset-eval', help='dataset to evaluate', default='wnut2017', type=str)
    parser.add_argument('-e', '--export', help='path to export the metric', default=None, type=str)
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    # train model
    lm = TransformersNER(model=opt.model, max_length=opt.max_length)
    get_dataset(opt.dataset_eval)
    dataset_split, _, _, _ = get_dataset(
        opt.dataset_eval, lower_case=opt.lower_case, label_to_id=lm.label2id, fix_label_dict=True)
    metrics_dict = {}
    for split in dataset_split.keys():
        if split == 'train':
            continue
        metrics_dict[split] = lm.span_f1(
            inputs=dataset_split[split]['data'],
            labels=dataset_split[split]['label'],
            batch_size=opt.batch_size,
        )
    print(json.dumps(metrics_dict, indent=4))
    if opt.export is not None:
        os.makedirs(os.path.dirname(opt.export), exist_ok=True)
        with open(opt.export, 'w') as f:
            json.dump(metrics_dict, f)


if __name__ == '__main__':
    main()
