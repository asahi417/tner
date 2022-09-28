import json
import unittest
import logging
import tner

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

lm = 'albert-base-v1'  # the smallest bert only 11M parameters
data = 'tner/fin'
ckpt = 'tests/test_ckpt'
path_to_custom_data = './examples/local_dataset_sample'
test_local_dataset = {"train": f"{path_to_custom_data}/train.txt", "valid": f"{path_to_custom_data}/valid.txt"}


class Test(unittest.TestCase):
    """ Test TrainTransformersNER """

    def test_trainer_1(self):
        model = tner.Trainer(
            f'{ckpt}_1', local_dataset=test_local_dataset, epoch=2, batch_size=4, model=lm, crf=True, max_length=32)
        # train for one epoch
        model.train(epoch_partial=1)
        # resume training
        model.train()
        # evaluation
        model = tner.TransformersNER(f'{ckpt}_1/epoch_2')
        out = model.evaluate(local_dataset=test_local_dataset,
                             dataset_split='valid',
                             batch_size=4,
                             cache_file_feature=f'{ckpt}_1/epoch_2/cache_feature',
                             cache_file_prediction=f'{ckpt}_1/epoch_2/cache_prediction')
        print(json.dumps(out, indent=4))
        out = model.evaluate(local_dataset=test_local_dataset,
                             dataset_split='valid',
                             batch_size=4,
                             span_detection_mode=True,
                             cache_file_feature=f'{ckpt}_1/epoch_2/cache_feature',
                             cache_file_prediction=f'{ckpt}_1/epoch_2/cache_prediction')
        print(json.dumps(out, indent=4))

    def test_trainer_2(self):
        model = tner.Trainer(f'{ckpt}_2', local_dataset=test_local_dataset, epoch=2, batch_size=4, model=lm, crf=True, max_length=32)
        # train for one epoch
        model.train(epoch_partial=1)

        # resume training
        model = tner.Trainer(f'{ckpt}_2')
        model.train()

    def test_trainer_3(self):
        model = tner.Trainer(f'{ckpt}_3', dataset=["tner/wnut2017", "tner/fin"], epoch=2, batch_size=4, model=lm, max_length=32)
        # train for one epoch
        model.train(epoch_partial=1)

        # resume training
        model = tner.Trainer(f'{ckpt}_2')
        model.train()


if __name__ == "__main__":
    unittest.main()
