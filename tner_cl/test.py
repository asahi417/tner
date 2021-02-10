""" Fine-tune transformers on NER dataset """
import argparse
from tner import TrainTransformersNER


def get_options():
    parser = argparse.ArgumentParser(description='Fine-tune transformers on NER dataset')
    parser.add_argument('-c', '--checkpoint_dir', help='checkpoint directory', required=True, type=str)
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    parser.add_argument('--test-data', help='test dataset (if not specified, use trained set)', default=None, type=str)
    parser.add_argument('--test-lower-case', help='lower case all the test data', action='store_true')
    parser.add_argument('--test-entity-span', help='evaluate entity span', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    # train model
    trainer = TrainTransformersNER(
        checkpoint_dir=opt.checkpoint_dir
    )
    test_data = [None] if opt.test_data is None else opt.test_data.split(',')
    for i in test_data:
        trainer.test(test_dataset=i, entity_span_prediction=opt.test_entity_span, lower_case=opt.test_lower_case)


if __name__ == '__main__':
    main()
