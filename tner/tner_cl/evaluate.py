""" Evaluate NER model """
import argparse
import json
import logging
import os

from tner import TransformersNER

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER model")
    parser.add_argument('-m', '--model', help='model alias of huggingface or local checkpoint', required=True, type=str)
    parser.add_argument('-e', '--export', help='file to export the result', required=True, type=str)
    parser.add_argument('-d', '--dataset',
                        help="dataset name (or a list of it) on huggingface tner organization "
                             "eg. 'tner/conll2003' ['tner/conll2003', 'tner/ontonotes5']] "
                             "see https://huggingface.co/datasets?search=tner for full dataset list",
                        nargs='+', default=None, type=str)
    parser.add_argument('-l', '--local-dataset',
                        help='a dictionary (or a list) of paths to local BIO files eg.'
                             '{"train": "examples/local_dataset_sample/train.txt",'
                             ' "test": "examples/local_dataset_sample/test.txt"}',
                        nargs='+', default=None, type=json.loads)
    parser.add_argument('--dataset-name',
                        help='[optional] data name of huggingface dataset (should be same length as the `dataset`)',
                        nargs='+', default=None, type=str)
    parser.add_argument('--dataset-split', help="dataset split to be used for test ('test' as default)",
                        default='test', type=str)
    parser.add_argument('--span-detection-mode', action='store_true',
                        help='return F1 of entity span detection (ignoring entity type error and cast '
                             'as binary sequence classification as below)'
                             '- NER                  : ["O", "B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG"]'
                             '- Entity-span detection: ["O", "B-ENT", "I-ENT", "O", "B-ENT", "O", "B-ENT"]')
    parser.add_argument('--return-ci', action='store_true', help='return confidence interval by bootstrap')
    parser.add_argument('-b', '--batch-size', help='batch size', default=32, type=int)
    opt = parser.parse_args()

    # train model
    model = TransformersNER(opt.model)
    metric = model.evaluate(
        dataset=opt.dataset,
        dataset_name=opt.dataset_name,
        local_dataset=opt.local_dataset,
        batch_size=opt.batch_size,
        dataset_split=opt.dataset_split,
        span_detection_mode=opt.span_detection_mode,
        return_ci=opt.return_ci,
        unification_by_shared_label=True,
        separator=" "
    )
    logging.info(json.dumps(metric, indent=4))
    os.makedirs(os.path.dirname(opt.export), exist_ok=True)
    with open(opt.export, 'w') as f:
        json.dump(metric, f)
