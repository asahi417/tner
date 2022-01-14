""" Cache prediction for contextualized prediction """
import argparse
import logging
import json

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
    parser.add_argument('-e', '--export-file', help='path to export the metric', default=None, type=str)
    parser.add_argument('--adapter', help='', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    if opt.adapter:
        assert opt.base_model is not None
        classifier = TransformersNER(opt.base_model, max_length=opt.max_length)
    else:
        classifier = TransformersNER(opt.model, max_length=opt.max_length)

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
        index_path=opt.index_path,
        max_retrieval_size=opt.max_retrieval_size
    )

    print(json.dumps(metric, indent=4))


if __name__ == '__main__':
    main()
