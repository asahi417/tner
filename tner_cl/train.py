""" Fine-tune transformers on NER dataset """
import argparse
import logging
from tner import VALID_DATASET
from tner import Trainer, GridSearcher

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def arguments(parser):
    parser.add_argument('-c', '--checkpoint-dir', help='directory to save checkpoint', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='dataset: {}'.format(VALID_DATASET), default='wnut2017', type=str)
    parser.add_argument('-m', '--model', help='pretrained language model', default='xlm-roberta-base', type=str)
    parser.add_argument('-e', '--epoch', help='epoch', default=15, type=int)
    parser.add_argument('-g', '--gradient-accumulation-steps', help='', default=4, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=128, type=int)
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--interval', default=50, type=int)
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    return parser


def arguments_training(parser):
    # training config
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--crf', action='store_true')
    parser.add_argument('--weight-decay', help='weight decay', default=None, type=float)
    parser.add_argument('--max-grad-norm', default=None, type=float)
    parser.add_argument('--lr-warmup-epoch', default=None, type=int)
    # monitoring parameter
    parser.add_argument('--epoch-save', default=1, type=int)
    return parser


def arguments_parameter_search(parser):
    parser.add_argument('--batch-eval', default=8, type=int)
    parser.add_argument('--n-max-config', default=5, type=int)
    parser.add_argument('--epoch-partial', help='epoch', default=5, type=int)
    parser.add_argument('--max-length-eval', default=256, type=int)
    parser.add_argument('--weight-decay', help='weight decay', default='1e-7', type=str)
    parser.add_argument('-l', '--lr', help='learning rate', default='5e-7,1e-6', type=str)
    parser.add_argument('--random-seed', help='random seed', default='0', type=str)
    parser.add_argument('--crf', default='0,1', type=str)
    parser.add_argument('--max-grad-norm', default='-1,1', type=str)
    parser.add_argument('--lr-warmup-epoch', default='-1,1', type=str)
    return parser


def main_train():
    parser = argparse.ArgumentParser(description='Fine-tuning on NER.')
    parser = arguments(parser)
    parser = arguments_training(parser)
    opt = parser.parse_args()

    # train model
    trainer = Trainer(
        checkpoint_dir=opt.checkpoint_dir,
        dataset=opt.dataset,
        random_seed=opt.random_seed,
        model=opt.model,
        lower_case=opt.lower_case,
        crf=opt.crf,
        weight_decay=opt.weight_decay,
        epoch=opt.epoch,
        lr=opt.lr,
        batch=opt.batch,
        max_length=opt.max_length,
        fp16=opt.fp16,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        max_grad_norm=opt.max_grad_norm,
        lr_warmup_epoch=opt.lr_warmup_epoch
    )
    trainer.train(
        epoch_save=opt.epoch_save,
        interval=opt.interval,
        num_workers=opt.num_workers)


def main_train_search():
    parser = argparse.ArgumentParser(description='Finetuning on NER with Grid Search.')
    parser = arguments(parser)
    parser = arguments_parameter_search(parser)
    opt = parser.parse_args()

    # train model
    trainer = GridSearcher(
        checkpoint_dir=opt.checkpoint_dir,
        dataset=opt.dataset,
        model=opt.model,
        lower_case=opt.lower_case,
        fp16=opt.fp16,
        epoch=opt.epoch,
        epoch_partial=opt.epoch_partial,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        batch=opt.batch,
        max_length=opt.max_length,
        n_max_config=opt.n_max_config,
        lr=[float(i) for i in opt.lr.split(',')],
        crf=[bool(int(i)) for i in opt.crf.split(',')],
        random_seed=[int(i) for i in opt.random_seed.split(',')],
        weight_decay=[float(i) for i in opt.weight_decay.split(',')],
        lr_warmup_epoch=[int(i) if int(i) != -1 else None for i in opt.lr_warmup_epoch.split(',')],
        max_grad_norm=[float(i) if float(i) != -1 else None for i in opt.max_grad_norm.split(',')],
        batch_eval=opt.batch_eval,
        max_length_eval=opt.max_length_eval
    )
    trainer.run(interval=opt.interval, num_workers=opt.num_workers)

