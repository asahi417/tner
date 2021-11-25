""" Fine-tune transformers on NER dataset """
import argparse
import logging
from tner import TransformersNER, get_dataset


def get_options():
    parser = argparse.ArgumentParser(description='Fine-tune transformers on NER dataset')
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    parser.add_argument('-d', '--dataset-eval', help='dataset to evaluate', default='wnut2017', type=str)
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    parser.add_argument('--test-lower-case', help='lower case all the test data', action='store_true')
    parser.add_argument('--test-entity-span', help='evaluate entity span', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    # train model
    lm = TransformersNER(model=opt.model, max_length=opt.max_length)
    get_dataset
    lm
    test_data = [None] if opt.test_data is None else opt.test_data.split(',')
    for i in test_data:
        trainer.test(test_dataset=i, entity_span_prediction=opt.test_entity_span, lower_case=opt.test_lower_case)


if __name__ == '__main__':
    main()
