""" command line tool to test finetuned NER model """
import logging
import argparse
from pprint import pprint

from tner import TransformersNER

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_options():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('-m', '--model', help='model alias of huggingface or local checkpoint', required=True, type=str)
    return parser.parse_args()


def main():
    opt = get_options()
    classifier = TransformersNER(opt.model)
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

