""" UnitTest """
import unittest
import logging
from tner.ner_model import TransformersNER

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
sample_sentence = ["I live in London", "Jacob Collier is a Grammy awarded English artist."]


class Test(unittest.TestCase):
    """ Test get_benchmark_dataset """

    def test_load_trained_model(self):
        model = TransformersNER('tner/roberta-base-tweetner-2021')
        pred = model.predict(sample_sentence)
        print(pred)

    def test_load_not_trained_model(self):
        try:
            TransformersNER('roberta-base')
        except AssertionError:
            pass
        TransformersNER('roberta-base', crf=True, label2id={'O': 0})
        model = TransformersNER('roberta-base', crf=False, label2id={'O': 0})
        pred = model.predict(sample_sentence)
        print(pred)


if __name__ == "__main__":
    unittest.main()
