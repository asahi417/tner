import os
import logging
import pickle
import re
import gc
from typing import List, Dict
from itertools import groupby

import transformers
import torch

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from .tokenizer import TokenizerFixed, Dataset, PAD_TOKEN_LABEL_ID

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message

__all__ = 'TransformersNER'


def pickle_save(obj, path: str):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    with open(path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


def load_hf(model_name, cache_dir, label2id, local_files_only=False):
    """ load huggingface checkpoints """
    logging.info('initialize language model with `{}`'.format(model_name))
    if label2id is not None:
        config = transformers.AutoConfig.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label={v: k for k, v in label2id.items()},
            label2id=label2id,
            cache_dir=cache_dir,
            local_files_only=local_files_only)
    else:
        config = transformers.AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only)
    gc.collect()
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_name, config=config, cache_dir=cache_dir, local_files_only=local_files_only)
    return model


class TransformersNER:

    def __init__(self,
                 model: str,
                 max_length: int = 128,
                 crf: bool = False,
                 label2id: Dict = None,
                 cache_dir: str = None):
        self.model_name = model
        self.max_length = max_length

        # load model
        try:
            self.model = load_hf(self.model_name, cache_dir, label2id)
        except Exception:
            self.model = load_hf(self.model_name, cache_dir, label2id, local_files_only=True)

        # load crf layer
        if 'crf_state_dict' in self.model.config.to_dict().keys() or crf:
            self.crf_layer = ConditionalRandomField(
                num_tags=len(self.model.config.id2label),
                constraints=allowed_transitions(constraint_type="BIO", labels=self.model.config.id2label)
            )
            if 'crf_state_dict' in self.model.config.to_dict().keys():
                state = {k: torch.FloatTensor(v) for k, v in self.model.config.crf_state_dict.items()}
                self.crf_layer.load_state_dict(state)
        else:
            self.crf_layer = None

        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
            if self.crf_layer is not None:
                self.crf_layer = torch.nn.DataParallel(self.crf_layer)
        self.model.to(self.device)
        if self.crf_layer is not None:
            self.crf_layer.to(self.device)
        logging.info('{} GPUs are in use'.format(torch.cuda.device_count()))

        # load pre processor
        if self.crf_layer is not None:
            self.tokenizer = TokenizerFixed(
                self.model_name, cache_dir=cache_dir, id2label=self.id2label, padding_id=self.label2id['O']
            )
        else:
            self.tokenizer = TokenizerFixed(self.model_name, cache_dir=cache_dir, id2label=self.id2label)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, save_dir):

        if self.parallel:
            if self.crf_layer is not None:
                self.model.module.config.update(
                    {'crf_state_dict': {k: v.tolist() for k, v in self.crf_layer.module.state_dict().items()}})
            self.model.module.save_pretrained(save_dir)
        else:
            if self.crf_layer is not None:
                self.model.config.update(
                    {'crf_state_dict': {k: v.tolist() for k, v in self.crf_layer.state_dict().items()}})
            self.model.save_pretrained(save_dir)
        self.tokenizer.tokenizer.save_pretrained(save_dir)

    def encode_to_loss(self, encode: Dict):
        assert 'labels' in encode
        encode = {k: v.to(self.device) for k, v in encode.items()}
        output = self.model(**encode)
        if self.crf_layer is not None:
            loss = - self.crf_layer(output['logits'], encode['labels'], encode['attention_mask'])
        else:
            loss = output['loss']
        return loss.mean() if self.parallel else loss

    def encode_to_prediction(self, encode: Dict):
        encode = {k: v.to(self.device) for k, v in encode.items()}
        output = self.model(**encode)
        if self.crf_layer is not None:
            if self.parallel:
                best_path = self.crf_layer.module.viterbi_tags(output['logits'])
            else:
                best_path = self.crf_layer.viterbi_tags(output['logits'])
            pred_results = []
            for tag_seq, prob in best_path:
                pred_results.append(tag_seq)
            return pred_results
        else:
            return torch.max(output['logits'], dim=-1)[1].cpu().detach().int().tolist()

    def get_data_loader(self,
                        inputs,
                        labels: List = None,
                        batch_size: int = None,
                        num_workers: int = 0,
                        shuffle: bool = False,
                        drop_last: bool = False,
                        mask_by_padding_token: bool = False,
                        cache_path: str = None):
        """ Transform features (produced by BERTClassifier.preprocess method) to data loader. """
        if cache_path is not None and os.path.exists(cache_path):
            logging.info('loading preprocessed feature from {}'.format(cache_path))
            out = pickle_load(cache_path)
        else:
            out = self.tokenizer.encode_plus_all(
                tokens=inputs,
                labels=labels,
                max_length=self.max_length,
                mask_by_padding_token=mask_by_padding_token)

            # remove overflow text
            logging.info('encode all the data: {}'.format(len(out)))

            # cache the encoded data
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                pickle_save(out, cache_path)
                logging.info('preprocessed feature is saved at {}'.format(cache_path))

        batch_size = len(out) if batch_size is None else batch_size
        return torch.utils.data.DataLoader(
            Dataset(out), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    def span_f1(self,
                inputs: List,
                labels: List,
                batch_size: int = None,
                num_workers: int = 0,
                cache_path: str = None):
        self.model.eval()
        loader = self.get_data_loader(
            inputs,
            labels=labels,
            batch_size=batch_size,
            num_workers=num_workers,
            mask_by_padding_token=True,
            cache_path=cache_path)
        label_list = []
        pred_list = []
        ind = 0
        for i in loader:
            label = i.pop('labels').cpu().tolist()
            pred = self.encode_to_prediction(i)
            assert len(label) == len(pred)
            for _l, _p in zip(label, pred):
                assert len(_l) == len(_p)
                tmp = [(__p, __l) for __p, __l in zip(_p, _l) if __l != PAD_TOKEN_LABEL_ID]
                tmp_pred = list(list(zip(*tmp))[0])
                tmp_label = list(list(zip(*tmp))[1])
                if len(tmp_label) != len(labels[ind]):
                    if len(tmp_label) < len(labels[ind]):
                        logging.info('found sequence possibly more than max_length')
                        logging.info('{}: \n\t - model loader: {}\n\t - label: {}'.format(ind, tmp_label, labels[ind]))
                        tmp_pred = tmp_pred + [self.label2id['O']] * (len(labels[ind]) - len(tmp_label))
                    else:
                        raise ValueError('{}: \n\t - model loader: {}\n\t - label: {}'.format(ind, tmp_label, labels[ind]))
                assert len(tmp_pred) == len(labels[ind])
                pred_list.append(tmp_pred)
                label_list.append(labels[ind])
                ind += 1

        label_list = [[self.id2label[__l] for __l in _l] for _l in label_list]
        pred_list = [[self.id2label[__p] for __p in _p] for _p in pred_list]
        # compute metrics
        logging.info('\n{}'.format(classification_report(label_list, pred_list)))
        metric = {
            "micro/f1": f1_score(label_list, pred_list, average='micro'),
            "micro/recall": recall_score(label_list, pred_list, average='micro'),
            "micro/precision": precision_score(label_list, pred_list, average='micro'),
            "macro/f1": f1_score(label_list, pred_list, average='macro'),
            "macro/recall": recall_score(label_list, pred_list, average='macro'),
            "macro/precision": precision_score(label_list, pred_list, average='macro'),
        }
        return metric

    def predict(self,
                inputs: List,
                batch_size: int = None,
                num_workers: int = 0,
                decode_bio: bool = True):
        self.eval()
        dummy_labels = [[0] * len(i) for i in inputs]
        loader = self.get_data_loader(
            inputs,
            labels=dummy_labels,
            mask_by_padding_token=True,
            batch_size=batch_size,
            num_workers=num_workers)
        pred_list = []
        inputs_list = []
        pointer = 0
        for i in loader:
            input_ids = i['input_ids'].cpu().tolist()
            labels = i['labels'].cpu().tolist()
            pred = self.encode_to_prediction(i)
            assert len(input_ids) == len(pred) == len(labels)
            for _i, _p, _l in zip(input_ids, pred, labels):
                assert len(_i) == len(_p) == len(_l)
                label = [self.id2label[__p] for __p, __l in zip(_p, _l) if __l != PAD_TOKEN_LABEL_ID]

                if len(label) != len(inputs[pointer]):
                    size = min(len(label), len(inputs[pointer]))
                    label = label[:size]
                    _inputs = inputs[pointer][:size]
                    logging.warning('sequence mismatch found: \n{}'.format(list(zip(_inputs, label))))
                else:
                    _inputs = inputs[pointer]
                pred_list.append(label)
                inputs_list.append(_inputs)
                pointer += 1
        if decode_bio:
            return [self.decode_ner_tags(_p, _i) for _p, _i in zip(pred_list, inputs_list)]
        return pred_list

    @staticmethod
    def decode_ner_tags(tag_sequence, input_sequence):
        assert len(tag_sequence) == len(input_sequence)

        def update_collection(_tmp_entity, _tmp_entity_type, _out):
            if len(_tmp_entity) != 0 and _tmp_entity_type is not None:
                _out.append({'type': _tmp_entity_type, 'entity': _tmp_entity})
                _tmp_entity = []
                _tmp_entity_type = None
            return _tmp_entity, _tmp_entity_type, _out

        out = []
        tmp_entity = []
        tmp_entity_type = None
        for _l, _i in zip(tag_sequence, input_sequence):
            if _l.startswith('B-'):
                _, _, out = update_collection(tmp_entity, tmp_entity_type, out)
                tmp_entity_type = '-'.join(_l.split('-')[1:])
                tmp_entity = [_i]
            elif _l.startswith('I-'):
                tmp_tmp_entity_type = '-'.join(_l.split('-')[1:])
                if len(tmp_entity) == 0 and tmp_entity_type is None:
                    tmp_entity, tmp_entity_type, out = update_collection(tmp_entity, tmp_entity_type, out)
                elif tmp_tmp_entity_type != tmp_entity_type:
                    tmp_entity, tmp_entity_type, out = update_collection(tmp_entity, tmp_entity_type, out)
                else:
                    tmp_entity.append(_i)
            elif _l == 'O':
                tmp_entity, tmp_entity_type, out = update_collection(tmp_entity, tmp_entity_type, out)
            else:
                raise ValueError('unknown tag: {}'.format(_l))
        _, _, out = update_collection(tmp_entity, tmp_entity_type, out)
        return out

