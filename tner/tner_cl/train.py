""" Fine-tune transformers on NER dataset """
import argparse
import json
import logging
from tner import GridSearcher, Trainer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def arguments(parser):
    parser.add_argument('-c', '--checkpoint-dir', help='checkpoint directory', required=True, type=str)
    parser.add_argument('-d', '--dataset',
                        help="dataset name (or a list of it) on huggingface tner organization "
                             "eg. 'tner/conll2003' ['tner/conll2003', 'tner/ontonotes5']] "
                             "see https://huggingface.co/datasets?search=tner for full dataset list",
                        nargs='+', default=None, type=str)
    parser.add_argument('-l', '--local-dataset',
                        help='a dictionary (or a list) of paths to local BIO files eg.'
                             '{"train": "examples/local_dataset_sample/train.txt",'
                             ' "test": "examples/local_dataset_sample/test.txt"}',
                        nargs='+', default=None, type=json.loads)
    parser.add_argument('--dataset-name',
                        help='[optional] data name of huggingface dataset (should be same length as the `dataset`)',
                        nargs='+', default=None, type=str)
    parser.add_argument('-m', '--model', help='model name of underlying language model (huggingface model)',
                        default='roberta-base', type=str)
    parser.add_argument('-b', '--batch-size', help='batch size', default=32, type=int)
    parser.add_argument('-e', '--epoch', help='the number of epoch', default=15, type=int)
    parser.add_argument('--max-length', default=128, type=int, help='max length of language model')
    parser.add_argument('--use-auth-token', help='Huggingface transformers argument of `use_auth_token`',
                        action='store_true')
    return parser


def arguments_trainer(parser):
    parser.add_argument('--dataset-split', help="dataset split to be used ('train' as default)",
                        default='train', type=str)
    parser.add_argument('--lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('--random-seed', help='random seed', default=42, type=int)
    parser.add_argument('-g', "--gradient-accumulation-steps", default=4, type=int,
                        help="the number of gradient accumulation")
    parser.add_argument('--weight-decay', help='coefficient of weight decay', default=1e-7, type=float)
    parser.add_argument('--lr-warmup-step-ratio',
                        help="linear warmup ratio of learning rate. eg) if it's 0.3, the learning rate will warmup "
                             "linearly till 30%% of the total step (no decay after all)",
                        default=0.1, type=float)
    parser.add_argument("--max-grad-norm", default=None, type=float, help="norm for gradient clipping")
    parser.add_argument('--crf', help='use CRF on top of output embedding (0 or 1)', default=True,
                        type=lambda x: bool(int(x)))
    parser.add_argument('--epoch-save',
                        help='interval of epoch to save intermediate checkpoint (every single epoch as default)',
                        default=1, type=int)
    parser.add_argument('--optimizer-on-cpu', help='put optimizer on CPU to save memory of GPU', action='store_true')
    return parser


def main_trainer():
    parser = argparse.ArgumentParser(description='Fine-tune transformers on NER dataset')
    parser = arguments(parser)
    parser = arguments_trainer(parser)
    opt = parser.parse_args()

    # train model
    trainer = Trainer(
        checkpoint_dir=opt.checkpoint_dir,
        dataset=opt.dataset,
        local_dataset=opt.local_dataset,
        dataset_split=opt.dataset_split,
        model=opt.model,
        crf=opt.crf,
        max_length=opt.max_length,
        epoch=opt.epoch,
        batch_size=opt.batch_size,
        lr=opt.lr,
        random_seed=opt.random_seed,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        weight_decay=opt.weight_decay,
        lr_warmup_step_ratio=opt.lr_warmup_step_ratio,
        max_grad_norm=opt.max_grad_norm,
        use_auth_token=opt.use_auth_token
    )
    trainer.train(epoch_save=opt.epoch_save, optimizer_on_cpu=opt.optimizer_on_cpu)


def arguments_trainer_with_search(parser):
    parser.add_argument('--dataset-split-train', help="dataset split to be used for training ('train' as default)",
                        default='train', type=str)
    parser.add_argument('--dataset-split-valid', help="dataset split to be used for validation ('validation' as default)",
                        default='validation', type=str)
    parser.add_argument('--lr', help='learning rate', default=[1e-4, 1e-5], type=float, nargs='+')
    parser.add_argument('--random-seed', help='random seed', default=[42], type=int, nargs='+')
    parser.add_argument('-g', "--gradient-accumulation-steps", default=[2, 4], type=int,
                        help="the number of gradient accumulation", nargs='+')
    parser.add_argument('--weight-decay', help='coefficient of weight decay (set 0 for None)',
                        default=[None, 1e-7], type=float, nargs='+')
    parser.add_argument('--lr-warmup-step-ratio',
                        help="linear warmup ratio of learning rate (no decay)."
                             "eg) if it's 0.3, the learning rate will warmup "
                             "linearly till 30%% of the total step (set 0 for None)",
                        default=[0.1], type=float, nargs='+')
    parser.add_argument("--max-grad-norm", default=[None, 10], type=float,
                        help="norm for gradient clipping (set 0 for None)", nargs='+')
    parser.add_argument('--crf', help='use CRF on top of output embedding (0 or 1)', default=[True, False],
                        type=lambda x: bool(int(x)), nargs='+')
    parser.add_argument('--optimizer-on-cpu', help='put optimizer on CPU to save memory of GPU', action='store_true')
    parser.add_argument('--n-max-config', default=3, help="the number of configs to run 2nd phase search", type=int)
    parser.add_argument('--epoch-partial', default=5, help="the number of epoch for 1st phase search", type=int)
    parser.add_argument('--max-length-eval', default=128, type=int, help='max length of language model at evaluation')
    return parser


def main_trainer_with_search():
    parser = argparse.ArgumentParser(description='Fine-tune transformers on NER dataset with Robust Parameter Search')
    parser = arguments(parser)
    parser = arguments_trainer_with_search(parser)
    opt = parser.parse_args()
    # train model
    trainer = GridSearcher(
        checkpoint_dir=opt.checkpoint_dir,
        dataset=opt.dataset,
        local_dataset=opt.local_dataset,
        n_max_config=opt.n_max_config,
        epoch_partial=opt.epoch_partial,
        max_length_eval=opt.max_length_eval,
        dataset_split_train=opt.dataset_split_train,
        dataset_split_valid=opt.dataset_split_valid,
        model=opt.model,
        crf=opt.crf,
        max_length=opt.max_length,
        epoch=opt.epoch,
        batch_size=opt.batch_size,
        lr=opt.lr,
        random_seed=opt.random_seed,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        weight_decay=[i if i != 0 else None for i in opt.weight_decay],
        lr_warmup_step_ratio=[i if i != 0 else None for i in opt.lr_warmup_step_ratio],
        max_grad_norm=[i if i != 0 else None for i in opt.max_grad_norm],
        use_auth_token=opt.use_auth_token
    )
    trainer.train(optimizer_on_cpu=opt.optimizer_on_cpu)
