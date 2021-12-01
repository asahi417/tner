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
    parser.add_argument('-d', '--data', help='dataset to evaluate', default=None, type=str)
    parser.add_argument('--custom-dataset-train', help='custom data set', default=None, type=str)
    parser.add_argument('--custom-dataset-valid', help='custom data set', default=None, type=str)
    parser.add_argument('--custom-dataset-test', help='custom data set', default=None, type=str)
    parser.add_argument('-e', '--export-dir', help='path to export the metric', default=None, type=str)
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    return parser.parse_args()


def format_data(opt):
    assert opt.dataset is not None or opt.custom_dataset_train is not None
    if opt.dataset is not None:
        return opt.dataset.split(','), None
    custom_data = {'train': opt.custom_dataset_train}
    if opt.custom_dataset_valid is not None:
        custom_data['valid'] = opt.custom_dataset_valid
    if opt.custom_dataset_test is not None:
        custom_data['test'] = opt.custom_dataset_test
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
        lower_case=opt.lower_case
    )

    print(json.dumps(metric, indent=4))


if __name__ == '__main__':
    main()
