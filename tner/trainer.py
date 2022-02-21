""" Training model. """
import os
import json
import logging
import shutil
import random
import gc

from glob import glob
from typing import List, Dict

import torch
import transformers

from .language_model import TransformersNER
from .data import get_dataset, CACHE_DIR

OPTIMIZER_ON_CPU = int(os.getenv('OPTIMIZER_ON_CPU', 0))


def get_model_instance(
        base_model=None, max_length=None, model_path: str = None,
        label_to_id: Dict = None, adapter_config: Dict = None,
        adapter: bool = False, crf: bool = False,
        index_data_path: str = None, index_prediction_path: str = None):
    adapter_config = {} if adapter_config is None else adapter_config
    if adapter:
        assert base_model is not None, 'adapter needs base model'
        model = base_model
        adapter_model = model_path
    else:
        model = base_model if model_path is None else model_path
        adapter_model = None
    return TransformersNER(
        model=model,
        crf=crf,
        max_length=max_length,
        adapter=adapter,
        adapter_model=adapter_model,
        label2id=label_to_id,
        index_data_path=index_data_path,
        index_prediction_path=index_prediction_path,
        **adapter_config
    )


class Config:
    """ Model checkpoint managing class. """

    def __init__(self, checkpoint_dir: str, **kwargs):
        self.checkpoint_dir = checkpoint_dir
        if os.path.exists(self.checkpoint_dir):
            logging.info('load config from existing checkpoint at {}'.format(self.checkpoint_dir))
            self.config = self.safe_open('{}/trainer_config.json'.format(self.checkpoint_dir))
        else:
            logging.info('initialize checkpoint at {}'.format(self.checkpoint_dir))
            self.config = kwargs
            configs = {i: self.safe_open(i) for i in glob(
                '{}/*/trainer_config.json'.format(os.path.dirname(self.checkpoint_dir)))}
            configs = list(filter(lambda x: x[1] == self.config, configs.items()))
            if len(configs) != 0:
                input('\ncheckpoint with same config already exists: {}\n enter to overwrite >>>'.format(configs[0]))
                for _p, _ in configs:
                    shutil.rmtree(os.path.dirname(_p))
            self.__initialize_checkpoint()

        self.__dict__.update(self.config)
        logging.info('hyperparameters')
        for k, v in self.config.items():
            logging.info('\t * {}: {}'.format(k, str(v)[:min(100, len(str(v)))]))

    def __initialize_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if not os.path.exists('{}/trainer_config.json'.format(self.checkpoint_dir)):
            with open('{}/trainer_config.json'.format(self.checkpoint_dir), 'w') as f:
                json.dump(self.config, f)

    @staticmethod
    def safe_open(_file):
        with open(_file, 'r') as f:
            return json.load(f)


class Trainer:

    def __init__(self,
                 checkpoint_dir: str,
                 dataset: (str, List) = None,
                 custom_dataset: Dict = None,
                 model: str = 'xlm-roberta-large',
                 crf: bool = False,
                 max_length: int = 128,
                 epoch: int = 10,
                 batch_size: int = 128,
                 lr: float = 1e-4,
                 fp16: bool = False,
                 lower_case: bool = False,
                 random_seed: int = 42,
                 gradient_accumulation_steps: int = 4,
                 weight_decay: float = 1e-7,
                 lr_warmup_step_ratio: int = None,
                 max_grad_norm: float = None,
                 disable_log: bool = False,
                 inherit_tner_checkpoint: bool = False,
                 adapter: bool = False,
                 adapter_config: Dict = None):

        logging.info('initialize model trainer')

        # config
        if dataset is not None:
            dataset = [dataset] if type(dataset) is str else dataset
        self.config = Config(
            checkpoint_dir=checkpoint_dir,
            dataset=dataset,
            custom_dataset=custom_dataset,
            model=model,
            max_length=max_length,
            epoch=epoch,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            fp16=fp16,
            lower_case=lower_case,
            random_seed=random_seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            crf=crf,
            lr_warmup_step_ratio=lr_warmup_step_ratio,
            max_grad_norm=max_grad_norm,
            adapter=adapter,
            adapter_config=adapter_config
        )

        random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if not disable_log:
            # add file handler
            logger = logging.getLogger()
            file_handler = logging.FileHandler('{}/training.log'.format(self.config.checkpoint_dir))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
            logger.addHandler(file_handler)

        # load model
        ckpts = glob('{}/epoch_*'.format(self.config.checkpoint_dir))
        ckpts = [i for i in ckpts if os.path.exists('{}/optimizers/optimizer.{}.pt'.format(
            self.config.checkpoint_dir, i.split('epoch_')[-1]
        ))]
        flag = False
        if len(ckpts) > 0:
            try:
                epoch = sorted([int(i.split('epoch_')[-1]) for i in ckpts], reverse=True)[0]
                path = '{}/epoch_{}'.format(self.config.checkpoint_dir, epoch)
                logging.info('load checkpoint from {}'.format(path))
                self.model = self.get_model_instance(path)
                self.current_epoch = epoch
                assert self.current_epoch <= self.config.epoch, 'model training is done'
                self.dataset_split, label_to_id, self.language, self.unseen_entity_set = get_dataset(
                    data=self.config.dataset,
                    custom_data=self.config.custom_dataset,
                    lower_case=lower_case,
                    label_to_id=self.model.label2id,
                    fix_label_dict=True)
                step_per_epoch = int(
                    len(self.dataset_split['train']['data'])/self.config.batch_size/self.config.gradient_accumulation_steps
                )
                self.optimizer, self.scheduler = self.setup_optimizer(epoch, step_per_epoch=step_per_epoch)
                flag = True
            except Exception:
                logging.exception('error at loading checkpoint {}'.format(ckpts))
        if not flag:
            if inherit_tner_checkpoint:
                logging.info('inherit checkpoint from {}'.format(self.config.model))
                self.model = self.get_model_instance()
                self.dataset_split, _, self.language, self.unseen_entity_set = get_dataset(
                    data=self.config.dataset,
                    custom_data=self.config.custom_dataset,
                    lower_case=lower_case,
                    label_to_id=self.model.label2id,
                    fix_label_dict=True)
            else:
                self.dataset_split, label_to_id, self.language, self.unseen_entity_set = get_dataset(
                    data=self.config.dataset,
                    custom_data=self.config.custom_dataset,
                    lower_case=lower_case)
                logging.info('initialize checkpoint with {}'.format(self.config.model))
                self.model = self.get_model_instance(label_to_id=label_to_id)
            step_per_epoch = int(
                len(self.dataset_split['train'][
                        'data']) / self.config.batch_size / self.config.gradient_accumulation_steps
            )
            self.current_epoch = 0
            self.optimizer, self.scheduler = self.setup_optimizer(step_per_epoch=step_per_epoch)
        # GPU mixture precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.fp16)

        # cached data folder
        if self.config.dataset is not None:
            os.makedirs(CACHE_DIR, exist_ok=True)
            self.cache_data_path = '{}/data_encoded/{}.{}.{}{}{}.train.pkl'.format(
                CACHE_DIR,
                '_'.join(sorted(self.config.dataset)),
                self.config.model,
                self.config.max_length,
                '.lower' if self.config.lower_case else '',
                '.crf' if self.config.crf else ''
            )
        else:
            self.cache_data_path = None

    def get_model_instance(self, model_path=None, label_to_id=None):
        return get_model_instance(self.config.model, self.config.max_length, model_path,
                                  label_to_id, self.config.adapter_config, self.config.adapter, self.config.crf)

    def setup_optimizer(self, epoch: int = None, step_per_epoch: int = None):
        # optimizer
        if self.config.weight_decay is not None and self.config.weight_decay != 0:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": self.config.weight_decay},
                {"params": [p for n, p in self.model.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0}]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.lr)
        else:
            optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=self.config.lr)
        if self.config.lr_warmup_step_ratio is not None:
            assert step_per_epoch is not None
            total_step = step_per_epoch * self.config.epoch
            num_warmup_steps = int(total_step * self.config.lr_warmup_step_ratio)
            logging.info('optimizer with scheduler:\n\t num_warmup_steps: {}\n\t num_training_steps:{}'.format(
                num_warmup_steps, total_step
            ))
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_step)
        else:
            scheduler = None

        # resume fine-tuning
        if epoch is not None:
            path = '{}/optimizers/optimizer.{}.pt'.format(self.config.checkpoint_dir, epoch)
            logging.info('load optimizer from {}'.format(path))
            self.release_cache()
            gc.collect()
            device = 'cpu' if OPTIMIZER_ON_CPU == 1 else self.model.device
            logging.info('optimizer is loading on {}'.format(device))
            stats = torch.load(path, map_location=torch.device(device))
            optimizer.load_state_dict(stats['optimizer_state_dict'])
            if scheduler is not None:
                logging.info('load scheduler from {}'.format(path))
                self.release_cache()
                gc.collect()
                scheduler.load_state_dict(stats['scheduler_state_dict'])
            del stats
            self.release_cache()
            gc.collect()
        return optimizer, scheduler

    def save(self, current_epoch):
        self.release_cache()
        gc.collect()
        # save model
        save_dir = '{}/epoch_{}'.format(self.config.checkpoint_dir, current_epoch + 1)
        os.makedirs(save_dir, exist_ok=True)
        logging.info('model saving at {}'.format(save_dir))
        self.model.save(save_dir)
        with open('{}/trainer_config.json'.format(save_dir), 'w') as f:
            json.dump(self.config, f)
        # save optimizer
        save_dir_opt = '{}/optimizers/optimizer.{}.pt'.format(self.config.checkpoint_dir, current_epoch + 1)
        os.makedirs(os.path.dirname(save_dir_opt), exist_ok=True)
        # Fix the memory error
        logging.info('optimizer saving at {}'.format(save_dir_opt))
        if self.scheduler is not None:
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_dir_opt)
        else:
            torch.save({'optimizer_state_dict': self.optimizer.state_dict()}, save_dir_opt)
        logging.info('remove old optimizer files')
        if os.path.exists('{}/optimizers/optimizer.{}.pt'.format(self.config.checkpoint_dir, current_epoch)):
            os.remove('{}/optimizers/optimizer.{}.pt'.format(self.config.checkpoint_dir, current_epoch))

    def train(self,
              num_workers: int = 0,
              epoch_save: None or int = 1,
              interval: int = 25,
              epoch_partial: int = None):

        logging.info('dataset preprocessing')
        if self.current_epoch == self.config.epoch:
            raise ValueError('training is completed')
        loader = self.model.get_data_loader(
            inputs=self.dataset_split['train']['data'],
            labels=self.dataset_split['train']['label'],
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            cache_data_path=self.cache_data_path)
        self.model.train()

        logging.info('start model training')
        global_step = 0

        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            for e in range(self.current_epoch, self.config.epoch):  # loop over the epoch
                mean_loss, global_step = self.train_single_epoch(loader, global_step, interval)
                logging.info('[epoch {}/{}] average loss: {}, lr: {}'.format(
                    e, self.config.epoch, round(mean_loss, 3), self.optimizer.param_groups[0]['lr']))
                if epoch_save is not None and (e + 1) % epoch_save == 0 and (e + 1) != 0:
                    self.save(e)
                if epoch_partial is not None and (e + 1) == epoch_partial:
                    break

        self.save(e)
        logging.info('complete training: model ckpt was saved at {}'.format(self.config.checkpoint_dir))

    def train_single_epoch(self, data_loader, global_step: int, interval):
        total_loss = []
        self.optimizer.zero_grad()
        for n, encode in enumerate(data_loader):
            loss = self.model.encode_to_loss(encode)

            self.scaler.scale(loss).backward()

            if self.config.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)

            total_loss.append(loss.cpu().item())
            if (n + 1) % self.config.gradient_accumulation_steps != 0:
                continue

            global_step += 1
            _total_loss = total_loss[-self.config.gradient_accumulation_steps:]
            inst_loss = sum(_total_loss)/len(_total_loss)

            # optimizer update
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()
            if global_step % interval == 0:
                logging.info('\t * (global step {}: loss: {}, lr: {}'.format(global_step, inst_loss, self.optimizer.param_groups[0]['lr']))

        self.optimizer.zero_grad()
        return sum(total_loss)/len(total_loss), global_step

    def release_cache(self):
        if self.model.device == "cuda":
            torch.cuda.empty_cache()
