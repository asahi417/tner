""" UnitTest """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner


class Test(unittest.TestCase):
    """ Test get_benchmark_dataset """

    def test_data(self):
        unified_data, label_to_id, language, unseen_entity_set = tner.get_dataset_ner(['conll2003', 'ontonote5'])
        unified_data, label_to_id, language, unseen_entity_set = tner.get_dataset_ner(['fin'])
        unified_data, label_to_id, language, unseen_entity_set = tner.get_dataset_ner(['bionlp2004'])
        unified_data, label_to_id, language, unseen_entity_set = tner.get_dataset_ner(['wiki_ja'])
        unified_data, label_to_id, language, unseen_entity_set = tner.get_dataset_ner(['wiki_news_ja'])

    def test_custom_data(self):
        unified_data, label_to_id, language, unseen_entity_set = tner.get_dataset_ner(['./tests/sample_data'])


if __name__ == "__main__":
    unittest.main()
