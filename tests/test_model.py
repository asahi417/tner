""" UnitTest for dataset """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner

data = './tests/sample_data'
transformers_model = 'albert-base-v1'  # the smallest bert only 11M parameters


class Test(unittest.TestCase):
    """ Test TrainTransformersNER """

    def test_1(self):

        # test training
        model = tner.TrainTransformersNER(dataset=[data],
                                          total_step=100,
                                          warmup_step=10,
                                          transformers_model=transformers_model,
                                          checkpoint_dir='./tests/ckpt_1')
        model.train()
        model.test(test_dataset=data)
        model.test(test_dataset=data, ignore_entity_type=True)
        model.test(test_dataset='wnut2017')
        model.test(test_dataset='wnut2017', ignore_entity_type=True)
        load_model(model.checkpoint)


def load_model(checkpoint):
    # test testing
    model = tner.TrainTransformersNER(checkpoint=checkpoint)
    model.test(test_dataset=data)
    model.test(test_dataset=data, ignore_entity_type=True)
    model.test(test_dataset='wnut2017')
    model.test(test_dataset='wnut2017', ignore_entity_type=True)


if __name__ == "__main__":
    unittest.main()
