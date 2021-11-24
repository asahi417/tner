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
            if 'panx' in i and i not in ['panx_dataset_en', 'panx_dataset_ja']:
                continue
            logging.info('######## {} ########'.format(i))
            unified_data, label_to_id, language, unseen_entity_set = tner.get_dataset([i])
            print(unified_data, label_to_id)
            input()
            tags = list(filter(lambda x: len(x) > 0, list(set([k[2:] for k in label_to_id.keys()]))))
            logging.info('- data: {}'.format(tags, len(tags)))
            logging.info('- tag: {} ({})'.format(tags, len(tags)))
            logging.info('- sample sentences:')
            for n in range(3):
                logging.info(' '.join(unified_data['valid']['data'][n]))


if __name__ == "__main__":
    unittest.main()
