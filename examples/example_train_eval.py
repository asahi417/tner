""" Model training/evaluation script

eg) To train a model
```
python ./example_train_eval.py \
    -t xlm-roberta-base \
    -d ontonote5 \
    --max-seq-length 128
```

eg) To evaluate a model
```
python ./example_train_eval.py -c <path-to-checkpoint> --test \
    --test-data conll_2003 \
    --test-ignore-entity \
    --test-greedy-baseline
```

"""
import argparse

from tner import TrainTransformersNER

VALID_DATASET = [
    'panx_dataset/*', 'conll_2003', 'wnut_17', 'ontonote5', 'mit_movie_trivia', 'mit_restaurant',
    'all_english', 'all_english_no_lower_case'
]


def get_options():
    parser = argparse.ArgumentParser(description='Fine-tune transformers on NER dataset')
    parser.add_argument('-c', '--checkpoint', help='checkpoint to load', default=None, type=str)
    parser.add_argument('--checkpoint-dir', help='checkpoint directory', default=None, type=str)
    parser.add_argument('-d', '--data', help='dataset: {}'.format(VALID_DATASET), default='wnut_17', type=str)
    parser.add_argument('-t', '--transformer', help='pretrained language model', default='xlm-roberta-base', type=str)
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--max-seq-length',
                        help='max sequence length (use same length as used in pre-training if not provided)',
                        default=128,
                        type=int)
    parser.add_argument('-b', '--batch-size', help='batch size', default=16, type=int)
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('--total-step', help='total training step', default=13000, type=int)
    parser.add_argument('--batch-size-validation',
                        help='batch size for validation (smaller size to save memory)',
                        default=2,
                        type=int)
    parser.add_argument('--warmup-step', help='warmup step (6 percent of total is recommended)', default=700, type=int)
    parser.add_argument('--weight-decay', help='weight decay', default=1e-7, type=float)
    parser.add_argument('--early-stop', help='value of accuracy drop for early stop', default=0.1, type=float)
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument('--skip-validation', help='train without validation', action='store_true')
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    parser.add_argument('--test-lower-case', help='lower case all the test data', action='store_true')
    parser.add_argument('--test', help='test mode', action='store_true')
    parser.add_argument('--test-data', help='test dataset (if not specified, use trained set)', default=None, type=str)
    parser.add_argument('--test-ignore-entity', help='test with ignoring entity type', action='store_true')
    parser.add_argument('--test-greedy-baseline',
                        help='test with greedy entity selection, the most frequent entity in the training set',
                        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    # train model
    trainer = TrainTransformersNER(
        checkpoint=opt.checkpoint,
        batch_size_validation=opt.batch_size_validation,
        checkpoint_dir=opt.checkpoint_dir,
        dataset=opt.data,
        transformer=opt.transformer,
        random_seed=opt.random_seed,
        lr=opt.lr,
        total_step=opt.total_step,
        warmup_step=opt.warmup_step,
        weight_decay=opt.weight_decay,
        batch_size=opt.batch_size,
        max_seq_length=opt.max_seq_length,
        early_stop=opt.early_stop,
        fp16=opt.fp16,
        max_grad_norm=opt.max_grad_norm,
        lower_case=opt.lower_case
    )
    if opt.test:
        trainer.test(
            test_dataset=opt.test_data,
            ignore_entity_type=opt.test_ignore_entity,
            greedy_baseline=opt.test_greedy_baseline,
            lower_case=opt.test_lower_case
        )
    else:
        trainer.train(skip_validation=opt.skip_validation)