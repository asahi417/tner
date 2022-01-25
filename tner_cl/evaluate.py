""" Fine-tune transformers on NER dataset """
import argparse
import logging
import json

from tner.grid_searcher import evaluate


def get_options():
    parser = argparse.ArgumentParser(description='Fine-tune transformers on NER dataset')
    parser.add_argument('--base-model', help='base model', default=None, type=str)
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('-d', '--dataset', help='dataset to evaluate', default=None, type=str)
    parser.add_argument('--custom-dataset', help='custom data set', default=None, type=str)
    parser.add_argument('--custom-dataset-name', help='custom data set', default='test', type=str)
    parser.add_argument('-e', '--export-dir', help='path to export the metric', default=None, type=str)
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    parser.add_argument('--span-detection', help='', action='store_true')
    parser.add_argument('--entity-list', help='', action='store_true')
    parser.add_argument('--adapter', help='', action='store_true')
    parser.add_argument('--max-retrieval-size', help='', default=10, type=int)
    parser.add_argument('--timeout', help='', default=None, type=int)
    parser.add_argument('--index-data-path', help='index for retrieval at prediction phase', default=None, type=str)
    parser.add_argument('--index-prediction-path', help='index for retrieval at prediction phase', default=None, type=str)
    parser.add_argument('--contextualisation-cache-prefix', help='index for retrieval at prediction phase', default=None, type=str)
    parser.add_argument('--timedelta-hour-after', help='', default=None, type=float)
    parser.add_argument('--timedelta-hour-before', help='', default=None, type=float)
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
        base_model=opt.base_model,
        export_dir=opt.export_dir,
        batch_size=opt.batch_size,
        max_length=opt.max_length,
        data=dataset,
        custom_dataset=custom_dataset,
        lower_case=opt.lower_case,
        span_detection_mode=opt.span_detection,
        adapter=opt.adapter,
        entity_list=opt.entity_list,
        force_update=True,
        index_data_path=opt.index_data_path,
        index_prediction_path=opt.index_prediction_path,
        max_retrieval_size=opt.max_retrieval_size,
        contextualisation_cache_prefix=opt.contextualisation_cache_prefix,
        timeout=opt.timeout,
        timedelta_hour_after=opt.timedelta_hour_after,
        timedelta_hour_before=opt.timedelta_hour_before
    )

    print(json.dumps(metric, indent=4))


if __name__ == '__main__':
    main()
