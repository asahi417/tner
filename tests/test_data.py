""" UnitTest """
import unittest
import logging
from tner.get_dataset import get_dataset, concat_dataset, get_shared_label

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
path_to_custom_data = './examples/custom_dataset_sample'
test_dataset = ['tner/conll2003', 'tner/ontonotes5']


class Test(unittest.TestCase):
    """ Test get_benchmark_dataset """

    def test_shared_label(self):
        label = get_shared_label()
        print(label)

    def test_get_data(self):
        for d in test_dataset:
            data = get_dataset(d)
            print(data)

    def test_custom_data(self):
        data = get_dataset(data_path={"train": f"{path_to_custom_data}/train.txt",
                                      "valid": f"{path_to_custom_data}/valid.txt"})
        print(data)

    def test_concat(self):
        all_data = []
        for d in test_dataset:
            data = get_dataset(d)
            all_data.append(data)
        data = get_dataset(data_path={"train": f"{path_to_custom_data}/train.txt",
                                      "valid": f"{path_to_custom_data}/valid.txt"})
        all_data.append(data)
        data = concat_dataset(all_data)
        print(data)


if __name__ == "__main__":
    unittest.main()
