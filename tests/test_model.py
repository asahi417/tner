""" UnitTest for dataset """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner

data = './examples/custom_dataset_sample'
transformers_model = 'albert-base-v1'  # the smallest bert only 11M parameters


class Test(unittest.TestCase):
    """ Test TrainTransformersNER """

    def test_1(self):

        # test training
        model = tner.TrainTransformersNER(dataset=[data],
                                          total_step=2,
                                          warmup_step=1,
                                          batch_size=1,
                                          transformers_model=transformers_model,
                                          checkpoint_dir='./tests/ckpt_1')
        assert not model.is_trained
        model.train()
        model.test(test_dataset=data)
        model.test(test_dataset=data, ignore_entity_type=True)
        checkpoint = model.checkpoint
        logging.info(checkpoint)

        model = tner.TrainTransformersNER(checkpoint=checkpoint)
        assert model.is_trained
        model.test(test_dataset=data)
        model.test(test_dataset=data, ignore_entity_type=True)


if __name__ == "__main__":
    unittest.main()
