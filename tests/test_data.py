""" UnitTest """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner


class Test(unittest.TestCase):
    """ Test get_benchmark_dataset """

    def test_data(self):
        unified_data, label_to_id, language = tner.get_dataset_ner(['conll2003', 'ontonote5'])
        assert language == 'en'

    def test_custom_data(self):
        unified_data, label_to_id, language = tner.get_dataset_ner(['./examples/sample_data'])
        assert language == 'en'
        unified_data, label_to_id, language = tner.get_dataset_ner(['wnut2017', './examples/sample_data'])
        assert language == 'en'


if __name__ == "__main__":
    unittest.main()