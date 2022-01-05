""" Fine-tune transformers on NER dataset """
import argparse
import logging
from tner import VALID_DATASET
from tner import Trainer, GridSearcher

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def arguments(parser):
    parser.add_argument('-c', '--checkpoint-dir', help='directory to save checkpoint', required=True, type=str)
    parser.add_argument('-d', '--dataset', help='dataset: {}'.format(VALID_DATASET), default=None, type=str)
    parser.add_argument('--custom-dataset-train', help='custom data set', default=None, type=str)
    parser.add_argument('--custom-dataset-valid', help='custom data set', default=None, type=str)
    parser.add_argument('--custom-dataset-test', help='custom data set', default=None, type=str)
    parser.add_argument('-m', '--model', help='pretrained language model', default='roberta-base', type=str)
    parser.add_argument('-e', '--epoch', help='epoch', default=15, type=int)
    parser.add_argument('-b', '--batch-size', help='batch size', default=128, type=int)
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--interval', default=10, type=int)
    parser.add_argument('--lower-case', help='lower case all the data', action='store_true')
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    parser.add_argument('--additional-special-tokens', help='', default=None, type=str)
    parser.add_argument('--inherit-tner-checkpoint', action='store_true')
    # adapter
    parser.add_argument('--adapter-task-name', default='ner', type=str)
    parser.add_argument('--adapter-non-linearity', default=None, type=str)
    parser.add_argument('--adapter-config', default='pfeiffer', type=str)
    parser.add_argument('--adapter-language', default='en', type=str)
    parser.add_argument('--adapter-reduction-factor', default=None, type=int)
    return parser


def arguments_training(parser):
    # training config
    parser.add_argument('--random-seed', help='random seed', default=1234, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--crf', action='store_true')
    parser.add_argument('--weight-decay', help='weight decay', default=None, type=float)
    parser.add_argument('--max-grad-norm', default=None, type=float)
    parser.add_argument('--lr-warmup-step-ratio', default=None, type=float)
    parser.add_argument('-g', '--gradient-accumulation-steps', help='', default=1, type=int)
    parser.add_argument('--adapter', action='store_true')
    # monitoring parameter
    parser.add_argument('--epoch-save', default=1, type=int)
    return parser


def arguments_parameter_search(parser):
    parser.add_argument('--batch-size-eval', default=8, type=int)
    parser.add_argument('--n-max-config', default=5, type=int)
    parser.add_argument('--epoch-partial', help='epoch', default=5, type=int)
    parser.add_argument('--max-length-eval', default=128, type=int)
    parser.add_argument('--weight-decay', help='weight decay', default='1e-7', type=str)
    parser.add_argument('-l', '--lr', help='learning rate', default='1e-6,1e-5,1e-4', type=str)
    parser.add_argument('--random-seed', help='random seed', default='0', type=str)
    parser.add_argument('--crf', default='0,1', type=str)
    parser.add_argument('--adapter', default='0', type=str)
    parser.add_argument('--max-grad-norm', default='-1,1', type=str)
    parser.add_argument('--lr-warmup-step-ratio', default='-1,0.3', type=str)
    parser.add_argument('-g', '--gradient-accumulation-steps', help='', default='1,4', type=str)
    return parser


def format_data(opt):
    assert opt.dataset is not None or opt.custom_dataset_train is not None
    if opt.dataset is not None:
        return opt.dataset.split(','), None
    custom_data = {'train': opt.custom_dataset_train}
    if opt.custom_dataset_valid is not None:
        custom_data['valid'] = opt.custom_dataset_valid
    if opt.custom_dataset_test is not None:
        custom_data['test'] = opt.custom_dataset_test
    return None, custom_data


def main_train():
    parser = argparse.ArgumentParser(description='Fine-tuning on NER.')
    parser = arguments(parser)
    parser = arguments_training(parser)
    opt = parser.parse_args()
    dataset, custom_dataset = format_data(opt)

    # train model
    trainer = Trainer(
        dataset=dataset,
        custom_dataset=custom_dataset,
        checkpoint_dir=opt.checkpoint_dir,
        random_seed=opt.random_seed,
        model=opt.model,
        lower_case=opt.lower_case,
        crf=opt.crf,
        weight_decay=opt.weight_decay,
        epoch=opt.epoch,
        lr=opt.lr,
        batch_size=opt.batch_size,
        max_length=opt.max_length,
        fp16=opt.fp16,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        max_grad_norm=opt.max_grad_norm,
        lr_warmup_step_ratio=opt.lr_warmup_step_ratio,
        inherit_tner_checkpoint=opt.inherit_tner_checkpoint,
        adapter=opt.adapter,
        adapter_config={
            "adapter_task_name": opt.adapter_task_name,
            "adapter_non_linearity": opt.adapter_non_linearity,
            "adapter_config": opt.adapter_config,
            "adapter_reduction_factor": opt.adapter_reduction_factor,
            "adapter_language": opt.adapter_language
        }
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

    dataset, custom_dataset = format_data(opt)
    # train model
    trainer = GridSearcher(
        dataset=dataset,
        custom_dataset=custom_dataset,
        checkpoint_dir=opt.checkpoint_dir,
        model=opt.model,
        lower_case=opt.lower_case,
        fp16=opt.fp16,
        epoch=opt.epoch,
        epoch_partial=opt.epoch_partial,
        batch_size=opt.batch_size,
        max_length=opt.max_length,
        n_max_config=opt.n_max_config,
        gradient_accumulation_steps=[int(i) for i in opt.gradient_accumulation_steps.split(',')],
        lr=[float(i) for i in opt.lr.split(',')],
        crf=[bool(int(i)) for i in opt.crf.split(',')],
        random_seed=[int(i) for i in opt.random_seed.split(',')],
        weight_decay=[float(i) for i in opt.weight_decay.split(',')],
        lr_warmup_step_ratio=[float(i) if float(i) != -1 else None for i in opt.lr_warmup_step_ratio.split(',')],
        max_grad_norm=[float(i) if float(i) != -1 else None for i in opt.max_grad_norm.split(',')],
        batch_size_eval=opt.batch_size_eval,
        max_length_eval=opt.max_length_eval,
        inherit_tner_checkpoint=opt.inherit_tner_checkpoint,
        adapter=[bool(int(i)) for i in opt.adapter.split(',')],
        adapter_config={
            "adapter_task_name": opt.adapter_task_name,
            "adapter_non_linearity": opt.adapter_non_linearity,
            "adapter_config": opt.adapter_config,
            "adapter_reduction_factor": opt.adapter_reduction_factor,
            "adapter_language": opt.adapter_language
        }
    )
    trainer.run(interval=opt.interval, num_workers=opt.num_workers)

