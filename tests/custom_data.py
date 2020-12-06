""" UnitTest for dataset """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner


class Test(unittest.TestCase):
    """Test get_benchmark_dataset """

    def test_custom_data(self):
        path = './examples/custom_data'
        tner.get_dataset_ner(path, )
        for i in VALID_DATASET_LIST:
            LOGGER.info('** {} **'.format(i))
            tmp, language = get_benchmark_dataset(i, keep_only_valid_label=True)
            LOGGER.info(language)
            for n, v in enumerate(tmp):
                LOGGER.info('\n - {0}: \n * source: {1} \n * keywords: {2}'.format(v['id'], v['source'], v['keywords']))
                if n > 5:
                    break

    def test_statistics(self):
        tmp, language = get_benchmark_dataset('Inspec', keep_only_valid_label=False)
        for n, i in enumerate(tmp):
            logging.info(i['source'])
            out = get_statistics(i['keywords'], i['source'])
            logging.info('- n_label: {}'.format(out['n_label']))
            logging.info('- n_label_in_candidates: {}'.format(out['n_label_in_candidates']))
            logging.info('- n_label_out_candidates: {}'.format(out['n_label_out_candidates']))
            logging.info('- n_label_intractable: {}'.format(out['n_label_intractable']))
            if n > 10:
                break


if __name__ == "__main__":
    unittest.main()