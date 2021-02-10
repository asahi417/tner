""" UnitTest """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner

path_to_custom_data = './examples/custom_dataset_sample'


class Test(unittest.TestCase):
    """ Test get_benchmark_dataset """

    def test_data(self):
        for i in tner.VALID_DATASET:
            if 'panx' in i:
                continue
            unified_data, label_to_id, language, unseen_entity_set = tner.get_dataset_ner([i])
            tags = list(filter(lambda x: len(x) > 0, list(set([k[2:] for k in label_to_id.keys()]))))
            logging.info('\n- data: {}, tag: {} ({})\n'.format(i, tags, len(tags)))

    def test_custom_data(self):
        tner.get_dataset_ner([path_to_custom_data])

    def test_multiple_data(self):
        tner.get_dataset_ner(['conll2003', 'fin', 'ontonotes5', path_to_custom_data])


if __name__ == "__main__":
    unittest.main()
