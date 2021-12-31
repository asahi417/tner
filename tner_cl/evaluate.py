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
    parser.add_argument('-d', '--dataset', help='dataset to evaluate', default=None, type=str)
    parser.add_argument('--custom-dataset', help='custom data set', default=None, type=str)
    parser.add_argument('--custom-dataset-name', help='custom data set', default='test', type=str)
    parser.add_argument('-e', '--export-dir', help='path to export the metric', default=None, type=str)
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    parser.add_argument('--span-detection', help='', action='store_true')
    return parser.parse_args()


def format_data(opt):
    assert opt.dataset is not None or opt.custom_dataset is not None
    if opt.dataset is not None:
        return opt.dataset.split(','), None
    custom_data = {}
    assert opt.custom_dataset_name is not None, 'please specify the name of the evaluation set'
    custom_data[opt.custom_dataset_name] = opt.custom_dataset
    return None, custom_data


def main():
    opt = get_options()
    dataset, custom_dataset = format_data(opt)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    metric = evaluate(
        model=opt.model,
        export_dir=opt.export_dir,
        batch_size=opt.batch_size,
        max_length=opt.max_length,
        data=dataset,
        custom_dataset=custom_dataset,
        lower_case=opt.lower_case,
        span_detection_mode=opt.span_detection,
        force_update=True
    )

    print(json.dumps(metric, indent=4))


if __name__ == '__main__':
    main()
