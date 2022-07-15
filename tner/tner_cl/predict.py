""" command line tool to test finetuned NER model """
import logging
import argparse
from pprint import pprint

from tner import TransformersNER


def get_options():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('-c', '--checkpoint', help='checkpoint to load', default=None, type=str)
    parser.add_argument('--debug', help='show debug log', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    classifier = TransformersNER(opt.checkpoint)
    test_sentences = [
        'I live in United States.',
        'I have an Apple computer.',
        'I like to eat an apple.'
    ]
    test_result = classifier.predict(test_sentences)
    pprint('-- DEMO --')
    pprint(test_result)
    pprint('----------')
    while True:
        _inp = input('input sentence >>>')
        if _inp == 'q':
            break
        elif _inp == '':
            continue
        else:
            pprint(classifier.predict([_inp]))


if __name__ == '__main__':
    main()
