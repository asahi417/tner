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


class TransformerNER:
    """ transformers NER, interface to get prediction from pre-trained checkpoint """

    def __init__(self, checkpoint: str, ):
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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.transformer, cache_dir=CACHE_DIR)
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
        encode_list = self.tokenizer.encode_plus(x, max_length=max_seq_length)
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
