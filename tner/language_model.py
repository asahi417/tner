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
                 max_length: int = 512,
                 crf: bool = False,
                 label2id: Dict = None,
                 cache_dir: str = None):
        self.model_name = model
        self.max_length = max_length

        # load model
        try:
            self.model = load_hf(self.model_name, cache_dir, label2id)
        except ValueError:
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
            return torch.max(output['logits'], dim=-1)[1].cpu().tolist()

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
                tokens=inputs, labels=labels, max_length=self.max_length, mask_by_padding_token=mask_by_padding_token)

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
        loader = self.get_data_loader(
            inputs, labels, batch_size=batch_size, num_workers=num_workers, mask_by_padding_token=True, cache_path=cache_path)
        label_list = []
        pred_list = []
        for i in loader:
            labels = i['labels'].cpu().tolist()
            pred = self.encode_to_prediction(i)
            assert len(labels) == len(pred)
            for _l, _p in zip(labels, pred):
                assert len(_l) == len(_p)
                tmp = [(__p, __l) for __p, __l in zip(_p, _l) if __l != PAD_TOKEN_LABEL_ID]
                pred_list.append(list(list(zip(*tmp))[0]))
                label_list.append(list(list(zip(*tmp))[1]))

        label_list = [[self.id2label[__p] for __p in _p] for _p in label_list]
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
                decode_bio: bool = False,
                cache_path: str = None):
        self.eval()
        loader = self.get_data_loader(inputs, batch_size=batch_size, num_workers=num_workers, cache_path=cache_path)
        pred_list = []
        inputs_list = []
        for i in loader:
            input_ids = i['input_ids'].cpu().tolist()
            pred = self.encode_to_prediction(i)
            for _input_ids, _pred in zip(input_ids, pred):
                _tmp = [(_i, self.id2label[_p])
                        for _i, _p in zip(_input_ids, _pred) if _i != self.tokenizer.pad_ids['input_ids']]
                pred_list.append([_p for _i, _p in _tmp])
                inputs_list.append([_i for _i, _p in _tmp])
        if decode_bio:
            return [self.decode_ner_tags(_p, _i) for _p, _i in zip(pred_list, inputs_list)]
        return pred_list

    def decode_ner_tags(self, tag_sequence, input_sequence):
        assert len(tag_sequence) == len(input_sequence)
        unique_type = list(set(i.split('-')[-1] for i in tag_sequence if i != 'O'))
        result = []
        for i in unique_type:
            mask = [t.split('-')[-1] == i for t in tag_sequence]

            # find blocks of True in a boolean list
            group = list(map(lambda x: list(x[1]), groupby(mask)))
            length = list(map(lambda x: len(x), group))
            group_length = [[sum(length[:n]), sum(length[:n]) + len(g) - 1] for n, g in enumerate(group) if all(g)]

            # get entity
            for g in group_length:
                surface = self.tokenizer.tokenizer.decode(input_sequence[g[0]:g[1]])
                surface = self.cleanup(surface)
                result.append({'type': i, 'entity': surface})
        result = sorted(result, key=lambda x: x['type'])
        return result

    @staticmethod
    def cleanup(_string):
        _string = re.sub(r'\A\s*', '', _string)
        _string = re.sub(r'[\s(\[{]*\Z', '', _string)
        return _string



