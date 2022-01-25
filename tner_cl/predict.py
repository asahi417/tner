""" command line tool to test finetuned NER model
tner-predict -m tner_output/twitter_ner_baseline/roberta_base/model_bzracx/epoch_9
tner-predict -m tner_output/twitter_ner_baseline/adapter_roberta_base/model_ncould/epoch_24 --base-model roberta-base --adapter
"""
import logging
import argparse
from pprint import pprint

from tner import TransformersNER


def get_options():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('--adapter', help='', action='store_true')
    parser.add_argument('--base-model', help='base model', default=None, type=str)
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    return parser.parse_args()


def main():
    opt = get_options()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    if opt.adapter:
        assert opt.base_model is not None, 'adapter needs base model'
        classifier = TransformersNER(
            opt.base_model, max_length=opt.max_length, adapter=opt.adapter,
            adapter_model=opt.model)
    else:
        classifier = TransformersNER(opt.model, max_length=opt.max_length)
    classifier.eval()

    test_sentences = [
        ['I', 'live', 'in', 'United', 'States', '.'],
        ['I', 'have', 'an', 'Apple', 'computer', '.']
    ]
    _, test_result = classifier.predict(test_sentences, decode_bio=True)
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
            pprint(classifier.predict([_inp.split(' ')], decode_bio=True)[1])


if __name__ == '__main__':
    main()
