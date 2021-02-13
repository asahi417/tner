""" UnitTest for dataset """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner
from pprint import pprint

transformers_model = 'asahi417/tner-xlm-roberta-large-ontonotes5'


class Test(unittest.TestCase):
    """ Test TrainTransformersNER """

    def test_1(self):
        model = tner.TransformersNER(transformers_model)
        test_sentences = [
            'I live in United States.',
            'I have an Apple computer.',
            'I like to eat an apple.'
        ]
        test_result = model.predict(test_sentences)
        pprint(list(zip(test_sentences, test_result)))


if __name__ == "__main__":
    unittest.main()
