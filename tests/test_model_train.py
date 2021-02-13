""" UnitTest for dataset """
import unittest
import logging
import shutil
import os
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
import tner

data = './examples/custom_dataset_sample'
transformers_model = 'albert-base-v1'  # the smallest bert only 11M parameters


class Test(unittest.TestCase):
    """ Test TrainTransformersNER """

    def test_train_finetuned_model(self):
        model = tner.TrainTransformersNER(
            dataset=[data],
            total_step=2,
            warmup_step=1,
            batch_size=1,
            checkpoint_dir='./tests/ckpt_test/finetuned',
            transformers_model='asahi417/tner-xlm-roberta-large-ontonotes5'
            )
        model.train()
        model.test(test_dataset=data)
        # clean up dir
        shutil.rmtree('./tests/ckpt_test')

    def test_test_finetuned_model(self):
        model = tner.TrainTransformersNER(
            transformers_model='asahi417/tner-xlm-roberta-large-ontonotes5',
            checkpoint_dir='./tests/ckpt_test/finetuned')
        model.test(test_dataset=data)
        model.test(test_dataset=data, entity_span_prediction=True)
        assert os.path.exists(model.args.checkpoint_dir)
        # clean up dir
        shutil.rmtree('./tests/ckpt_test')

    def test_train(self):
        # model training/test
        model = tner.TrainTransformersNER(
            './tests/ckpt_test/finetune_1',
            dataset=data, total_step=2, warmup_step=1, batch_size=1, transformers_model=transformers_model)
        model.train()
        model.test(test_dataset=data)
        assert os.path.exists(model.args.checkpoint_dir)

        # model additional training
        model = tner.TrainTransformersNER(
            './tests/ckpt_test/finetune_2',
            dataset=data, total_step=2, warmup_step=1, batch_size=1, transformers_model='./tests/ckpt_test/finetune_1')
        model.train()
        model.test(test_dataset=data)
        assert os.path.exists(model.args.checkpoint_dir)

        # model training/test: duplicate name but different config
        model = tner.TrainTransformersNER(
            './tests/ckpt_test/finetune_1',
            dataset=data, total_step=3, warmup_step=1, batch_size=1, transformers_model=transformers_model)
        model.train()
        model.test(test_dataset=data)
        assert os.path.exists(model.args.checkpoint_dir)

        # model additional training: duplicate name but different config
        model = tner.TrainTransformersNER(
            './tests/ckpt_test/finetune_1',
            dataset=data, total_step=3, warmup_step=1, batch_size=1, transformers_model='./tests/ckpt_test/finetune_1')
        model.train()
        model.test(test_dataset=data)
        assert os.path.exists(model.args.checkpoint_dir)

        # model additional training: wrong name
        try:
            model = tner.TrainTransformersNER(
                './tests/ckpt_test/finetune_wrong',
                dataset=data, total_step=3, warmup_step=1, batch_size=1, transformers_model='./tests/ckpt_test/finetune_10')
            model.train()
            model.test(test_dataset=data)
        except OSError:
            pass

        # model loading and test
        model = tner.TrainTransformersNER('./tests/ckpt_test/test_1', transformers_model='./tests/ckpt_test/finetune_1')
        model.test(test_dataset=data)

        # clean up dir
        shutil.rmtree('./tests/ckpt_test')


if __name__ == "__main__":
    unittest.main()
