""" hugginface.transformers based NER model """
import os
import random
import json
import logging
from time import time
from typing import Dict, List
from itertools import groupby, chain

import transformers
import torch
from torch import nn
from torch.autograd import detect_anomaly
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from torch.utils.tensorboard import SummaryWriter

from .get_dataset import get_dataset_ner
from .checkpoint_versioning import Argument
from .tokenizer import Transforms
from .util import get_logger

LOGGER = get_logger()
NUM_WORKER = 4
PROGRESS_INTERVAL = 100
CACHE_DIR = './cache'
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning
os.makedirs(CACHE_DIR, exist_ok=True)


class Dataset(torch.utils.data.Dataset):
    """ simple torch.utils.data.Dataset wrapper converting into tensor"""
    float_tensors = ['attention_mask']

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class TrainTransformersNER:
    """ finetune transformers NER """

    def __init__(self,
                 dataset: str = None,
                 batch_size_validation: int = None,
                 checkpoint: str = None,
                 checkpoint_dir: str = None,
                 transformer: str = 'xlm-roberta-base',
                 random_seed: int = 1234,
                 lr: float = 1e-5,
                 total_step: int = 13000,
                 warmup_step: int = 700,
                 weight_decay: float = 1e-7,
                 batch_size: int = 16,
                 max_seq_length: int = 128,
                 early_stop: float = None,
                 fp16: bool = False,
                 max_grad_norm: float = 1.0,
                 lower_case: bool = False):
        LOGGER.info('*** initialize network ***')

        # checkpoint version
        self.args = Argument(
            dataset=dataset,
            checkpoint_dir=checkpoint_dir,
            checkpoint=checkpoint,
            transformer=transformer,
            random_seed=random_seed,
            lr=lr,
            total_step=total_step,
            warmup_step=warmup_step,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            early_stop=early_stop,
            fp16=fp16,
            max_grad_norm=max_grad_norm,
            lower_case=lower_case)

        self.batch_size_validation = batch_size_validation if batch_size_validation else self.args.batch_size

        # fix random seed
        random.seed(self.args.random_seed)
        transformers.set_seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

        # dataset
        if self.args.model_statistics is None:
            self.dataset_split, self.label_to_id, self.language, _ = get_dataset_ner(
                self.args.dataset, lower_case=self.args.lower_case)
            with open(os.path.join(self.args.checkpoint_dir, 'label_to_id.json'), 'w') as f:
                json.dump(self.label_to_id, f)
        else:
            self.dataset_split, self.label_to_id, self.language, _ = get_dataset_ner(
                self.args.dataset, label_to_id=self.args.label_to_id, fix_label_dict=True, lower_case=self.args.lower_case)
        self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}

        # model setup
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(
            self.args.transformer,
            config=transformers.AutoConfig.from_pretrained(
                self.args.transformer,
                num_labels=len(self.id_to_label),
                id2label=self.id_to_label,
                label2id=self.label_to_id,
                cache_dir=CACHE_DIR)
        )
        self.transforms = Transforms(self.args.transformer)

        # optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=1e-8)

        # scheduler
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_step, num_training_steps=self.args.total_step)

        # apply checkpoint statistics to optimizer/scheduler
        self.__step = 0
        self.__epoch = 0
        self.__best_val_score = None
        if self.args.model_statistics is not None:
            self.__step = self.args.model_statistics['step']
            self.__epoch = self.args.model_statistics['epoch']
            self.__best_val_score = self.args.model_statistics['best_val_score']
            self.model.load_state_dict(self.args.model_statistics['model_state_dict'])
            if self.optimizer is not None and self.scheduler is not None:
                self.optimizer.load_state_dict(self.args.model_statistics['optimizer_state_dict'])
                self.scheduler.load_state_dict(self.args.model_statistics['scheduler_state_dict'])

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model.to(self.device)

        # GPU mixture precision
        self.scale_loss = None
        if self.args.fp16:
            try:
                from apex import amp  # noqa: F401
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level='O1', max_loss_scale=2 ** 13, min_loss_scale=1e-5)
                self.master_params = amp.master_params
                self.scale_loss = amp.scale_loss
                LOGGER.info('using `apex.amp`')
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        # multi-gpus
        if self.n_gpu > 1:
            # multi-gpu training (should be after apex fp16 initialization)
            self.model = torch.nn.DataParallel(self.model.cuda())
            LOGGER.info('using `torch.nn.DataParallel`')
        LOGGER.info('running on %i GPUs' % self.n_gpu)

    def __setup_loader(self, data_type: str, dataset_split: Dict, language: str):
        if data_type not in dataset_split.keys():
            return None
        is_train = data_type == 'train'
        features = self.transforms.encode_plus_all(
            tokens=dataset_split[data_type]['data'],
            labels=dataset_split[data_type]['label'],
            language=language,
            max_length=self.args.max_seq_length if is_train else None)
        data_obj = Dataset(features)
        _batch_size = self.args.batch_size if is_train else self.batch_size_validation
        return torch.utils.data.DataLoader(
            data_obj, num_workers=NUM_WORKER, batch_size=_batch_size, shuffle=is_train, drop_last=is_train)

    def test(self,
             test_dataset: str = None,
             ignore_entity_type: bool = False,
             greedy_baseline: bool = False,
             lower_case: bool = False):
        """
        ignore_entity_type: entity position detection
        greedy_baseline: rely on model's prediction but use most frequent entity type in training set as the entity
        """
        if test_dataset is not None:
            LOGGER.addHandler(logging.FileHandler(
                os.path.join(self.args.checkpoint_dir, 'logger_test.{}.log'.format(test_dataset.replace('/', '_')))))
            LOGGER.info('cross-transfer testing on {}...'.format(test_dataset))
            dataset_split, self.label_to_id, language, unseen_entity_set = get_dataset_ner(
                test_dataset, label_to_id=self.label_to_id, lower_case=lower_case)
            self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}
        else:
            LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_test.log')))
            dataset_split = self.dataset_split
            language = self.language
            unseen_entity_set = None
        if greedy_baseline:
            b_labels = {k: v for k, v in self.label_to_id.items() if 'B' in k}
            labels_flatten = list(chain(*dataset_split['train']['label']))
            entity_n = sorted([
                ('-'.join(self.id_to_label[v].split('-')[1:]), sum(i == v for i in labels_flatten))
                for k, v in b_labels.items()],
                              key=lambda x: x[1], reverse=True)
            top_entity, top_n = entity_n[0]
        else:
            top_entity, top_n = None, None

        data_loader = {k: self.__setup_loader(k, dataset_split, language) for k in dataset_split.keys() if k != 'train'}
        LOGGER.info('data_loader: {}'.format(str(list(data_loader.keys()))))
        LOGGER.info('ignore_entity_type: {}'.format(ignore_entity_type))
        LOGGER.info('greedy_baseline: {}'.format(greedy_baseline))
        start_time = time()
        for k, v in data_loader.items():
            assert v is not None, '{} data split is not found'.format(k)
            self.__epoch_valid(
                v,
                prefix=k,
                ignore_entity_type=ignore_entity_type,
                unseen_entity_set=unseen_entity_set,
                greedy_entity_type=top_entity)
            self.release_cache()
        LOGGER.info('[test completed, %0.2f sec in total]' % (time() - start_time))

    def train(self, skip_validation: bool = False):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_train.log')))
        writer = SummaryWriter(log_dir=self.args.checkpoint_dir)
        start_time = time()

        # setup dataset/data loader
        data_loader = {k: self.__setup_loader(k, self.dataset_split, self.language) for k in ['train', 'valid']}
        LOGGER.info('data_loader: %s' % str(list(data_loader.keys())))
        LOGGER.info('*** start training from step %i, epoch %i ***' % (self.__step, self.__epoch))
        if_early_stop = False
        try:
            with detect_anomaly():
                while True:
                    if_training_finish = self.__epoch_train(data_loader['train'], writer=writer)
                    self.release_cache()
                    if not skip_validation and data_loader['valid']:
                        if_early_stop = self.__epoch_valid(data_loader['valid'], writer=writer, prefix='valid')
                        self.release_cache()
                    if if_training_finish or if_early_stop:
                        break
                    self.__epoch += 1
        except RuntimeError:
            LOGGER.exception('*** RuntimeError (NaN found, see above log in detail) ***')

        except KeyboardInterrupt:
            LOGGER.info('*** KeyboardInterrupt ***')

        LOGGER.info('[training completed, %0.2f sec in total]' % (time() - start_time))
        if self.n_gpu > 1:
            model_wts = self.model.module.state_dict()
        else:
            model_wts = self.model.state_dict()
        torch.save({
            'step': self.__step,
            'epoch': self.__epoch,
            'model_state_dict': model_wts,
            'best_val_score': self.__best_val_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, os.path.join(self.args.checkpoint_dir, 'model.pt'))
        writer.close()
        LOGGER.info('ckpt saved at %s' % self.args.checkpoint_dir)

    def __epoch_train(self, data_loader, writer):
        """ train on single epoch, returning flag which is True if training has been completed """
        self.model.train()
        for i, encode in enumerate(data_loader, 1):
            # update model
            encode = {k: v.to(self.device) for k, v in encode.items()}
            self.optimizer.zero_grad()
            loss = self.model(**encode)[0]
            if self.n_gpu > 1:
                loss = loss.mean()
            if self.args.fp16:
                with self.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.master_params(self.optimizer), self.args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            # optimizer and scheduler step
            self.optimizer.step()
            self.scheduler.step()
            # log instantaneous accuracy, loss, and learning rate
            inst_loss = loss.cpu().detach().item()
            inst_lr = self.optimizer.param_groups[0]['lr']
            if writer:
                writer.add_scalar('train/loss', inst_loss, self.__step)
                writer.add_scalar('train/learning_rate', inst_lr, self.__step)
            if self.__step % PROGRESS_INTERVAL == 0:
                LOGGER.info('[epoch %i] * (training step %i) loss: %.3f, lr: %0.8f'
                            % (self.__epoch, self.__step, inst_loss, inst_lr))
            self.__step += 1
            # break
            if self.__step >= self.args.total_step:
                LOGGER.info('reached maximum step')
                return True
        return False

    def __epoch_valid(self,
                      data_loader,
                      writer=None,
                      prefix: str = 'valid',
                      unseen_entity_set: set = None,
                      ignore_entity_type: bool = False,
                      greedy_entity_type: int = None):
        """ validation/test, returning flag which is True if early stop condition was applied """
        self.model.eval()
        seq_pred, seq_true = [], []
        for encode in data_loader:
            encode = {k: v.to(self.device) for k, v in encode.items()}
            labels_tensor = encode.pop('labels')
            model_outputs = self.model(**encode)
            logit = model_outputs[0]
            _true = labels_tensor.cpu().detach().int().tolist()
            _pred = torch.max(logit, 2)[1].cpu().detach().int().tolist()
            for b in range(len(_true)):
                _pred_list, _true_list = [], []
                for s in range(len(_true[b])):
                    if _true[b][s] != PAD_TOKEN_LABEL_ID:
                        _true_list.append(self.id_to_label[_true[b][s]])
                        if unseen_entity_set is None:
                            _pred_list.append(self.id_to_label[_pred[b][s]])
                        else:
                            __pred = self.id_to_label[_pred[b][s]]
                            if __pred in unseen_entity_set:
                                _pred_list.append('O')
                            else:
                                _pred_list.append(__pred)
                assert len(_pred_list) == len(_true_list)
                if len(_true_list) > 0:
                    if ignore_entity_type:
                        # ignore entity type and focus on entity position
                        _true_list = [i if i == 'O' else '-'.join([i.split('-')[0], 'entity']) for i in _true_list]
                        _pred_list = [i if i == 'O' else '-'.join([i.split('-')[0], 'entity']) for i in _pred_list]
                    elif greedy_entity_type is not None:
                        _pred_list = [i if i == 'O' else '-'.join([i.split('-')[0], greedy_entity_type])
                                      for i in _pred_list]
                    seq_true.append(_true_list)
                    seq_pred.append(_pred_list)
        try:
            LOGGER.info('[epoch %i] (%s) \n %s' % (self.__epoch, prefix, classification_report(seq_true, seq_pred)))
        except ZeroDivisionError:
            LOGGER.info('[epoch %i] (%s) * classification_report raises `ZeroDivisionError`' % (self.__epoch, prefix))
        if writer:
            writer.add_scalar('%s/f1' % prefix, f1_score(seq_true, seq_pred), self.__epoch)
            writer.add_scalar('%s/recall' % prefix, recall_score(seq_true, seq_pred), self.__epoch)
            writer.add_scalar('%s/precision' % prefix, precision_score(seq_true, seq_pred), self.__epoch)
            writer.add_scalar('%s/accuracy' % prefix, accuracy_score(seq_true, seq_pred), self.__epoch)
        if prefix == 'valid':
            score = f1_score(seq_true, seq_pred)
            if self.__best_val_score is None or score > self.__best_val_score:
                self.__best_val_score = score
            if self.args.early_stop and self.__best_val_score - score > self.args.early_stop:
                return True
        return False

    def release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()


class TransformersNER:
    """ transformers NER, interface to get prediction from pre-trained checkpoint """

    def __init__(self, checkpoint: str):
        LOGGER.info('*** initialize network ***')

        # checkpoint version
        self.args = Argument(checkpoint=checkpoint)
        if self.args.model_statistics is None:
            raise ValueError('model is not trained')
        self.label_to_id = self.args.label_to_id
        self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(
            self.args.transformer,
            config=transformers.AutoConfig.from_pretrained(
                self.args.transformer,
                num_labels=len(self.id_to_label),
                id2label=self.id_to_label,
                label2id=self.label_to_id,
                cache_dir=CACHE_DIR)
        )
        self.transforms = Transforms(self.args.transformer)
        self.model.load_state_dict(self.args.model_statistics['model_state_dict'])

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model.to(self.device)

    @staticmethod
    def decode_ner_tags(tag_sequence, tag_probability, non_entity: str = 'O'):
        """ take tag sequence, return list of entity
        input:  ["B-LOC", "O", "O", "B-ORG", "I-ORG", "O"]
        return: [['LOC', [0, 1]], ['ORG', [3, 5]]]
        """
        assert len(tag_sequence) == len(tag_probability)
        unique_type = list(set(i.split('-')[-1] for i in tag_sequence if i != non_entity))
        result = []
        for i in unique_type:
            mask = [t.split('-')[-1] == i for t, p in zip(tag_sequence, tag_probability)]

            # find blocks of True in a boolean list
            group = [list(g) for _, g in groupby(mask)]
            length = [len(g) for g in group]
            group_length = [[sum(length[:n]), sum(length[:n]) + len(g)] for n, g in enumerate(group) if all(g)]

            # get entity
            for g in group_length:
                result.append([i, g])
        result = sorted(result, key=lambda x: x[1][0])
        return result

    def predict(self, x: List, max_seq_length=128):
        """ return a list of dictionary consisting of 'type', 'position', 'mention' and """
        self.model.eval()
        encode_list = self.transforms.encode_plus_all(x, max_length=max_seq_length)
        data_loader = torch.utils.data.DataLoader(Dataset(encode_list), batch_size=len(encode_list))
        encode = list(data_loader)[0]
        logit = self.model(**{k: v.to(self.device) for k, v in encode.items()})[0]
        entities = []
        for n, e in enumerate(encode['input_ids'].cpu().tolist()):
            sentence = self.transforms.tokenizer.decode(e, skip_special_tokens=True)

            pred = torch.max(logit[n], dim=-1)[1].cpu().tolist()
            activated = nn.Softmax(dim=-1)(logit[n])
            prob = torch.max(activated, dim=-1)[0].cpu().tolist()
            pred = [self.id_to_label[_p] for _p in pred]
            tag_lists = self.decode_ner_tags(pred, prob)

            _entities = []
            for tag, (start, end) in tag_lists:
                mention = self.transforms.tokenizer.decode(e[start:end], skip_special_tokens=True)
                start_char = len(self.transforms.tokenizer.decode(e[:start], skip_special_tokens=True))
                if sentence[start_char] == ' ':
                    start_char += 1
                end_char = start_char + len(mention)
                if mention != sentence[start_char:end_char]:
                    LOGGER.warning('entity mismatch: {} vs {}'.format(mention, sentence[start_char:end_char]))
                result = {'type': tag, 'position': [start_char, end_char], 'mention': mention,
                          'probability': sum(prob[start: end])/(end - start)}
                _entities.append(result)

            entities.append({'entity': _entities, 'sentence': sentence})
        return entities
