""" NER finetuning on hugginface.transformers """
import argparse
import os
import random
import json
import logging
import re
from time import time
from logging.config import dictConfig
from itertools import chain, groupby
from typing import List, Dict
from pprint import pprint

import transformers
import torch
from torch import nn
from torch.autograd import detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score

from get_dataset import Dataset, get_dataset_ner
from checkpoint_versioning import Argument


dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
NUM_WORKER = int(os.getenv("NUM_WORKER", '4'))
PROGRESS_INTERVAL = int(os.getenv("PROGRESS_INTERVAL", '100'))
CACHE_DIR = os.getenv("CACHE_DIR", './cache')
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning
os.makedirs(CACHE_DIR, exist_ok=True)


def additional_special_tokens(tokenizer):
    """ get additional special token for beginning/separate/ending, {'input_ids': [0], 'attention_mask': [1]} """
    encode_first = tokenizer.encode_plus('sent1', 'sent2')
    # group by block boolean
    sp_token_mask = [i in tokenizer.all_special_ids for i in encode_first['input_ids']]
    group = [list(g) for _, g in groupby(sp_token_mask)]
    length = [len(g) for g in group]
    group_length = [[sum(length[:n]), sum(length[:n]) + len(g)] for n, g in enumerate(group) if all(g)]
    assert len(group_length) == 3, 'more than 3 special tokens group: {}'.format(group)
    sp_token_start = {k: v[group_length[0][0]:group_length[0][1]] for k, v in encode_first.items()}
    sp_token_sep = {k: v[group_length[1][0]:group_length[1][1]] for k, v in encode_first.items()}
    sp_token_end = {k: v[group_length[2][0]:group_length[2][1]] for k, v in encode_first.items()}
    return sp_token_start, sp_token_sep, sp_token_end


class Transforms:
    """ NER specific transform pipeline"""

    def __init__(self, transformer_tokenizer: str):
        """ NER specific transform pipeline"""
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_tokenizer, cache_dir=CACHE_DIR)
        self.pad_ids = {"labels": PAD_TOKEN_LABEL_ID, "input_ids": self.tokenizer.pad_token_id, "__default__": 0}
        # find tokenizer-depend prefix
        self.prefix = self.__sp_token_prefix()
        # find special tokens to be added
        self.sp_token_start, _, self.sp_token_end = additional_special_tokens(self.tokenizer)
        self.sp_token_end['labels'] = [self.pad_ids['labels']] * len(self.sp_token_end['input_ids'])
        self.sp_token_start['labels'] = [self.pad_ids['labels']] * len(self.sp_token_start['input_ids'])

    def __sp_token_prefix(self):
        sentence_go_around = ''.join(self.tokenizer.tokenize('get tokenizer specific prefix'))
        return sentence_go_around[:list(re.finditer('get', sentence_go_around))[0].span()[0]]

    def fixed_encode_en(self, tokens, labels: List = None, max_seq_length: int = 128):
        """ fixed encoding for language with halfspace in between words """
        encode = self.tokenizer.encode_plus(
            ' '.join(tokens), max_length=max_seq_length, pad_to_max_length=True, truncation=True)
        if labels:
            assert len(tokens) == len(labels)
            fixed_labels = list(chain(*[
                [label] + [self.pad_ids['labels']] * (len(self.tokenizer.tokenize(word)) - 1)
                for label, word in zip(labels, tokens)]))
            fixed_labels = [self.pad_ids['labels']] * len(self.sp_token_start['labels']) + fixed_labels
            fixed_labels = fixed_labels[:min(len(fixed_labels), max_seq_length - len(self.sp_token_end['labels']))]
            fixed_labels = fixed_labels + [self.pad_ids['labels']] * (max_seq_length - len(fixed_labels))
            encode['labels'] = fixed_labels
        return encode

    def fixed_encode_ja(self, tokens, labels: List = None, max_seq_length: int = 128):
        """ fixed encoding for language without halfspace in between words """
        dummy = '@'
        # get special tokens at start/end of sentence based on first token
        encode_all = self.tokenizer.batch_encode_plus(tokens)
        # token_ids without prefix/special tokens
        # `wifi` will be treated as `_wifi` and change the tokenize result, so add dummy on top of the sentence to fix
        token_ids_all = [[self.tokenizer.convert_tokens_to_ids(_t.replace(self.prefix, '').replace(dummy, ''))
                          for _t in self.tokenizer.tokenize(dummy+t)
                          if len(_t.replace(self.prefix, '').replace(dummy, '')) > 0]
                         for t in tokens]

        for n in range(len(tokens)):
            if n == 0:
                encode = {k: v[n][:-len(self.sp_token_end[k])] for k, v in encode_all.items()}
                if labels:
                    encode['labels'] = [self.pad_ids['labels']] * len(self.sp_token_start['labels']) + [labels[n]]
                    encode['labels'] += [self.pad_ids['labels']] * (len(encode['input_ids']) - len(encode['labels']))
            else:
                encode['input_ids'] += token_ids_all[n]
                # other attribution without prefix/special tokens
                tmp_encode = {k: v[n] for k, v in encode_all.items()}
                s, e = len(self.sp_token_start['input_ids']), -len(self.sp_token_end['input_ids'])
                input_ids_with_prefix = tmp_encode.pop('input_ids')[s:e]
                prefix_length = len(input_ids_with_prefix) - len(token_ids_all[n])
                for k, v in tmp_encode.items():
                    s, e = len(self.sp_token_start['input_ids']) + prefix_length, -len(self.sp_token_end['input_ids'])
                    encode[k] += v[s:e]
                if labels:
                    encode['labels'] += [labels[n]] + [self.pad_ids['labels']] * (len((token_ids_all[n])) - 1)

        # add special token at the end and padding/truncate accordingly
        for k in encode.keys():
            encode[k] = encode[k][:min(len(encode[k]), max_seq_length - len(self.sp_token_end[k]))]
            encode[k] += self.sp_token_end[k]
            pad_id = self.pad_ids[k] if k in self.pad_ids.keys() else self.pad_ids['__default__']
            encode[k] += [pad_id] * (max_seq_length - len(encode[k]))
        return encode

    def encode_plus_all(self,
                        tokens: List,
                        labels: List = None,
                        language: str = 'en',
                        max_length: int = None):
        max_length = self.tokenizer.max_len if max_length is None else max_length
        # TODO: no padding for prediction
        shared_param = {'language': language, 'pad_to_max_length': True, 'max_length': max_length}
        if labels:
            return [self.encode_plus(*i, **shared_param) for i in zip(tokens, labels)]
        else:
            return [self.encode_plus(i, **shared_param) for i in tokens]

    def encode_plus(self, tokens, labels: List = None, language: str = 'en', max_length: int = None,
                    pad_to_max_length: bool = False):
        if labels is None:
            return self.tokenizer.encode_plus(
                tokens, max_length=max_length, pad_to_max_length=pad_to_max_length, truncation=pad_to_max_length)
        if language == 'en':
            return self.fixed_encode_en(tokens, labels, max_length)
        elif language == 'ja':
            return self.fixed_encode_ja(tokens, labels, max_length)
        else:
            raise ValueError('unknown language: {}'.format(language))

    @property
    def all_special_ids(self):
        return self.tokenizer.all_special_ids

    @property
    def all_special_tokens(self):
        return self.tokenizer.all_special_tokens

    def tokenize(self, *args, **kwargs):
        return self.tokenizer.tokenize(*args, **kwargs)


class TrainTransformerNER:
    """ finetune transformers NER """

    def __init__(self,
                 batch_size_validation: int = None,
                 checkpoint: str = None,
                 checkpoint_dir: str = './ckpt',
                 **kwargs):
        LOGGER.info('*** initialize network ***')

        # checkpoint version
        self.args = Argument(checkpoint=checkpoint, checkpoint_dir=checkpoint_dir, **kwargs)
        self.batch_size_validation = batch_size_validation if batch_size_validation else self.args.batch_size

        # fix random seed
        random.seed(self.args.random_seed)
        transformers.set_seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

        # dataset
        if self.args.model_statistics:
            self.label_to_id = self.args.label_to_id
            self.dataset_split, _ = get_dataset_ner(
                self.args.dataset, cache_dir=CACHE_DIR, label_to_id=self.label_to_id)
        else:
            self.dataset_split, self.label_to_id = get_dataset_ner(self.args.dataset, cache_dir=CACHE_DIR)
            with open(os.path.join(self.args.checkpoint_dir, 'label_to_id.json'), 'w') as f:
                json.dump(self.label_to_id, f)
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

    def test(self):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_test.log')))
        data_loader = {k: self.__setup_loader(k, self.dataset_split) for k in self.dataset_split.keys() if k != 'train'}
        LOGGER.info('data_loader: %s' % str(list(data_loader.keys())))
        start_time = time()
        for k, v in data_loader.items():
            self.__epoch_valid(v, prefix=k)
            self.release_cache()
        LOGGER.info('[test completed, %0.2f sec in total]' % (time() - start_time))

    def __setup_loader(self, data_type: str, dataset_split: Dict):
        assert data_type in dataset_split.keys()
        is_train = data_type == 'train'
        features = self.transforms.encode_plus_all(
            tokens=dataset_split[data_type]['data'],
            labels=dataset_split[data_type]['label'],
            language=self.args.language,
            max_length=self.args.max_seq_length if is_train else None)
        data_obj = Dataset(features)
        _batch_size = self.args.batch_size if is_train else self.batch_size_validation
        return torch.utils.data.DataLoader(
            data_obj, num_workers=NUM_WORKER, batch_size=_batch_size, shuffle=is_train, drop_last=is_train)

    def train(self):
        LOGGER.addHandler(logging.FileHandler(os.path.join(self.args.checkpoint_dir, 'logger_train.log')))
        writer = SummaryWriter(log_dir=self.args.checkpoint_dir)
        start_time = time()

        # setup dataset/data loader
        data_loader = {k: self.__setup_loader(k, self.dataset_split) for k in ['train', 'valid']}
        LOGGER.info('data_loader: %s' % str(list(data_loader.keys())))
        LOGGER.info('*** start training from step %i, epoch %i ***' % (self.__step, self.__epoch))
        try:
            with detect_anomaly():
                while True:
                    if_training_finish = self.__epoch_train(data_loader['train'], writer=writer)
                    self.release_cache()
                    if_early_stop = self.__epoch_valid(data_loader['valid'], writer=writer, prefix='valid')
                    self.release_cache()
                    if if_training_finish or if_early_stop:
                        break
                    self.__epoch += 1
        except RuntimeError:
            LOGGER.exception('*** RuntimeError (NaN found, see above log in detail) ***')

        except KeyboardInterrupt:
            LOGGER.info('*** KeyboardInterrupt ***')

        if self.__best_val_score is None:
            self.args.remove_ckpt()
            exit('nothing to be saved')

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

    def __epoch_valid(self, data_loader, writer=None, prefix: str='valid'):
        """ validation/test, returning flag which is True if early stop condition was applied """
        self.model.eval()
        list_loss, seq_pred, seq_true = [], [], []
        for encode in data_loader:
            encode = {k: v.to(self.device) for k, v in encode.items()}
            model_outputs = self.model(**encode)
            loss, logit = model_outputs[0:2]
            if self.n_gpu > 1:
                loss = torch.sum(loss)
            list_loss.append(loss.cpu().detach().item())
            _true = encode['labels'].cpu().detach().int().tolist()
            _pred = torch.max(logit, 2)[1].cpu().detach().int().tolist()
            for b in range(len(_true)):
                _pred_list, _true_list = [], []
                for s in range(len(_true[b])):
                    if _true[b][s] != PAD_TOKEN_LABEL_ID:
                        _true_list.append(self.id_to_label[_pred[b][s]])
                        _pred_list.append(self.id_to_label[_true[b][s]])
                assert len(_pred_list) == len(_true_list)
                if len(_true_list) > 0:
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
            writer.add_scalar('%s/loss' % prefix, float(sum(list_loss) / len(list_loss)), self.__epoch)
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


class TransformerNER:
    """ transformers NER, interface to get prediction from pre-trained checkpoint """

    def __init__(self, checkpoint: str):
        LOGGER.info('*** initialize network ***')

        # checkpoint version
        self.args = Argument(checkpoint=checkpoint)
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
    def decode_ner_tags(tag_sequence, tag_probability, min_probability: float, non_entity: str = 'O'):
        """ take tag sequence, return list of entity
        input:  ["B-LOC", "O", "O", "B-ORG", "I-ORG", "O"]
        return: [['LOC', [0, 1]], ['ORG', [3, 5]]]
        """
        assert len(tag_sequence) == len(tag_probability)
        mask = [t != non_entity and p > min_probability for t, p in zip(tag_sequence, tag_probability)]

        # find blocks of True in a boolean list
        group = [list(g) for _, g in groupby(mask)]
        length = [len(g) for g in group]
        group_length = [[sum(length[:n]), sum(length[:n]) + len(g)] for n, g in enumerate(group) if all(g)]

        # get entity
        result = []
        for i in group_length:
            entity = tag_sequence[i[0]].split('-')[1]
            result.append([entity, i])
        return result

    def predict(self, x: List, max_seq_length=128, min_probability: float = 0.8):
        """ return a list of dictionary consisting of 'type', 'position', 'mention' and
        'probability' if return_prob is True"""
        self.model.eval()
        encode_list = self.transforms.encode_plus_all(x, max_length=max_seq_length)
        data_loader = torch.utils.data.DataLoader(Dataset(encode_list), batch_size=len(encode_list))
        encode = list(data_loader)[0]
        logit = self.model(**{k: v.to(self.device) for k, v in encode.items()})[0]
        entities = []
        for n, e in enumerate(encode['input_ids'].cpu().tolist()):
            pred = torch.max(logit[n], dim=-1)[1].cpu().tolist()
            activated = nn.Softmax(dim=-1)(logit[n])
            prob = torch.max(activated, dim=-1)[0].cpu().tolist()
            _entities = []
            pred = [self.id_to_label[_p] for _p in pred]
            tag_lists = self.decode_ner_tags(pred, prob, min_probability=min_probability)
            for tag, (start, end) in tag_lists:
                mention = self.transforms.tokenizer.decode(e[start:end], skip_special_tokens=True)
                start_char = len(self.transforms.tokenizer.decode(e[:start], skip_special_tokens=True)) + 1
                end_char = start_char + len(mention)
                result = {'type': tag, 'position': [start_char, end_char], 'mention': mention,
                          'probability': prob[start: end]}
                _entities.append(result)
                entities.append(
                    {'entity': _entities, 'sentence': self.transforms.tokenizer.decode(e, skip_special_tokens=True)})
        return entities


def get_options():
    parser = argparse.ArgumentParser(
        description='finetune transformers to sentiment analysis',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--checkpoint', help='checkpoint to load', default=None, type=str)
    parser.add_argument('--checkpoint-dir', help='checkpoint directory', default='./ckpt', type=str)
    parser.add_argument('-d', '--data', help='data conll_2003/wnut_17', default='wnut_17', type=str)
    parser.add_argument('-t', '--transformer', help='pretrained language model', default='xlm-roberta-base', type=str)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
    parser.add_argument('--test', help='run over testdataset', action='store_true')
    parser.add_argument('--test-data', help='dataset for test', default=None, type=str)
    parser.add_argument('--fp16', help='fp16', action='store_true')
    parser.add_argument('-l', '--language', help='language', default='en', type=str)
    parser.add_argument('--inference-mode', help='inference mode', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    if not opt.inference_mode:
        # train model
        trainer = TrainTransformerNER(
            batch_size_validation=opt.batch_size_validation,
            checkpoint=opt.checkpoint,
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
            language=opt.language
        )
        if opt.test:
            trainer.test()
        else:
            trainer.train()
    else:
        # play around with trained model
        classifier = TransformerNER(checkpoint=opt.checkpoint)
        test_sentences = [
            'I live in United States, but Microsoft asks me to move to Japan.',
            'I have an Apple computer.',
            'I like to eat an apple.'
        ]
        test_result = classifier.predict(test_sentences)
        pprint('-- DEMO --')
        pprint(test_result)
        pprint('----------')
        while True:
            _inp = input('input sentence >>>')
            if _inp == 'q':
                break
            elif _inp == '':
                continue
            else:
                pprint(classifier.predict([_inp]))
