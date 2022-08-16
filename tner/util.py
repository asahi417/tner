import logging
import pickle
import json
import string
import random
from typing import List, Dict
from tqdm import tqdm
from itertools import chain

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForTokenClassification

# For evaluation (span-F1 score)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from scipy.stats import bootstrap

from .get_dataset import get_shared_label


def pickle_save(obj, path: str):
    """ save as pickle object

    @param obj: object
    @param path: path to save
    """
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    """ load pickle object

    @param path: path to load
    @return: object
    """
    with open(path, "rb") as fp:
        return pickle.load(fp)


def json_load(path: str):
    """ load json object

    @param path: path to load
    @return: object
    """
    with open(path, 'r') as f:
        return json.load(f)


def json_save(obj, path: str):
    """ save as json object

    @param obj: object
    @param path: path to save
    """
    with open(path, 'w') as f:
        json.dump(obj, f)


def get_random_string(length: int = 6, exclude: List = None):
    """ get random string

    @param length: length of string to generate
    @param exclude: a list of strings where the generated string shouldn't match
    @return: string
    """
    tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    if exclude:
        while tmp in exclude:
            tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    return tmp


def span_f1(pred_list: List,
            label_list: List,
            span_detection_mode: bool = False,
            return_ci: bool = False,
            unification_by_shared_label: bool = True):
    """ calculate span F1 score

    @param pred_list: a list of predicted tag sequences
    @param label_list: a list of gold tag sequences
    @param return_ci: [optional] return confidence interval by bootstrap
    @param span_detection_mode: [optional] return F1 of entity span detection (ignoring entity type error and cast
        as binary sequence classification as below)
        - NER                  : ["O", "B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG"]
        - Entity-span detection: ["O", "B-ENT", "I-ENT", "O", "B-ENT", "O", "B-ENT"]
    @param unification_by_shared_label: [optional] map entities into a shared form
    @return: a dictionary containing span f1 scores
    """
    if unification_by_shared_label:
        unified_label_set = get_shared_label()
        logging.info(f'map entity into shared label set {unified_label_set}')

        def convert_to_shared_entity(entity_label):
            if entity_label == 'O':
                return entity_label
            prefix = entity_label.split('-')[0]  # B or I
            entity = '-'.join(entity_label.split('-')[1:])
            normalized_entity = [k for k, v in unified_label_set.items() if entity in v]
            assert len(
                normalized_entity) <= 1, f'duplicated entity found in the shared label set\n {normalized_entity} \n {entity}'
            if len(normalized_entity) == 0:
                logging.warning(f'Entity `{entity}` is not found in the shared label set {unified_label_set}. '
                                f'Original entity (`{entity}`) will be used as label.')
                return f'{prefix}-{entity}'
            return f'{prefix}-{normalized_entity[0]}'


        label_list = [[convert_to_shared_entity(_i) for _i in i] for i in label_list]
        pred_list = [[convert_to_shared_entity(_i) for _i in i] for i in pred_list]

    if span_detection_mode:
        logging.info(f'span_detection_mode: map entity into binary label set')

        def convert_to_binary_mask(entity_label):
            if entity_label == 'O':
                return entity_label
            prefix = entity_label.split('-')[0]  # B or I
            return f'{prefix}-entity'

        label_list = [[convert_to_binary_mask(_i) for _i in i] for i in label_list]
        pred_list = [[convert_to_binary_mask(_i) for _i in i] for i in pred_list]

    # compute metrics
    logging.info(f'\n{classification_report(label_list, pred_list)}')
    m_micro, ci_micro = span_f1_single(label_list, pred_list, average='micro', return_ci=return_ci)
    m_macro, ci_macro = span_f1_single(label_list, pred_list, average='macro', return_ci=return_ci)
    metric = {
        "micro/f1": m_micro,
        "micro/f1_ci": ci_micro,
        "micro/recall": recall_score(label_list, pred_list, average='micro'),
        "micro/precision": precision_score(label_list, pred_list, average='micro'),
        "macro/f1": m_macro,
        "macro/f1_ci": ci_macro,
        "macro/recall": recall_score(label_list, pred_list, average='macro'),
        "macro/precision": precision_score(label_list, pred_list, average='macro'),
    }
    target_names = sorted(list(set([k.replace('B-', '') for k in list(chain(*label_list)) if k.startswith('B-')])))

    if not span_detection_mode:
        metric["per_entity_metric"] = {}
        for t in target_names:
            _label_list = [[_i if _i.endswith(t) else 'O' for _i in i] for i in label_list]
            _pred_list = [[_i if _i.endswith(t) else 'O' for _i in i] for i in pred_list]
            m, ci = span_f1_single(_label_list, _pred_list, return_ci=return_ci)
            metric["per_entity_metric"][t] = {
                "f1": m,
                "f1_ci": ci,
                "precision": precision_score(_label_list, _pred_list),
                "recall": recall_score(_label_list, _pred_list)}
    return metric


def span_f1_single(label_list,
                   pred_list,
                   random_seed: int = 0,
                   n_resamples: int = 1000,
                   confidence_level: List = None,
                   return_ci: bool = False,
                   average: str = 'macro'):
    """ span-F1 score with bootstrap CI (data.shape == (n_sample, 2)) """
    data = np.array(list(zip(pred_list, label_list)), dtype=object)

    def get_f1(xy, axis=None):
        assert len(xy.shape) in [2, 3], xy.shape
        prediction = xy[0]
        label = xy[1]
        if axis == -1 and len(xy.shape) == 3:
            assert average is not None
            tmp = []
            for i in tqdm(list(range(len(label)))):
                tmp.append(f1_score(label[i, :], prediction[i, :], average=average))
            return np.array(tmp)
        assert average is not None
        return f1_score(label, prediction, average=average)

    confidence_level = confidence_level if confidence_level is not None else [90, 95]
    mean_score = get_f1(data.T)
    ci = {}
    if return_ci:
        for c in confidence_level:
            logging.info(f'computing confidence interval: {c}')
            res = bootstrap(
                (data,),
                get_f1,
                confidence_level=c * 0.01,
                method='percentile',
                n_resamples=n_resamples,
                random_state=np.random.default_rng(random_seed)
            )
            ci[str(c)] = [res.confidence_interval.low, res.confidence_interval.high]
    return mean_score, ci


def decode_ner_tags(tag_sequence: List, input_sequence: List, probability_sequence: List = None):
    """ decode ner tag sequence

    @param tag_sequence: a list of tag sequence
    @param input_sequence: a list of input token sequence
    @param probability_sequence: [optional] a list of tag probability
    @return: a list of dictionary of
        {'type': entity type, 'entity': entity, 'position': position in the input, 'probability': probability}
    """
    def update_collection(_tmp_entity, _tmp_entity_type, _tmp_prob, _tmp_pos, _out):
        if len(_tmp_entity) != 0 and _tmp_entity_type is not None:
            if _tmp_prob is None:
                _out.append({'type': _tmp_entity_type, 'entity': _tmp_entity, 'position': _tmp_pos})
            else:
                _out.append({'type': _tmp_entity_type, 'entity': _tmp_entity, 'position': _tmp_pos,
                             'probability': _tmp_prob})
            _tmp_entity = []
            _tmp_prob = []
            _tmp_entity_type = None
        return _tmp_entity, _tmp_entity_type, _tmp_prob, _tmp_pos, _out

    probability_sequence = [None] * len(tag_sequence) if probability_sequence is None else probability_sequence
    assert len(tag_sequence) == len(input_sequence) == len(probability_sequence), str(
        [len(tag_sequence), len(input_sequence), len(probability_sequence)])
    out = []
    tmp_entity = []
    tmp_prob = []
    tmp_pos = []
    tmp_entity_type = None
    for n, (_l, _i, _prob) in enumerate(zip(tag_sequence, input_sequence, probability_sequence)):
        if _l.startswith('B-'):
            _, _, _, _, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
            tmp_entity_type = '-'.join(_l.split('-')[1:])
            tmp_entity = [_i]
            tmp_prob = [_prob]
            tmp_pos = [n]
        elif _l.startswith('I-'):
            tmp_tmp_entity_type = '-'.join(_l.split('-')[1:])
            if len(tmp_entity) == 0:
                # if 'I' not start with 'B', skip it
                tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out = update_collection(
                    tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
            elif tmp_tmp_entity_type != tmp_entity_type:
                # if the type does not match with the B, skip
                tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out = update_collection(
                    tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
            else:
                tmp_entity.append(_i)
                tmp_pos.append(n)
                tmp_prob.append(_prob)
        elif _l == 'O':
            tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out = update_collection(
                tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
        else:
            raise ValueError('unknown tag: {}'.format(_l))
    _, _, _, _, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
    return out


class Dataset(torch.utils.data.Dataset):

    float_tensors = ['attention_mask', 'input_feature']

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


def load_hf(model: str,
            label2id: Dict = None,
            use_auth_token: bool = False,
            local_files_only: bool = False):
    """ load model instance from huggingface

    @param model: the huggingface model (`tner/roberta-large-tweetner-2021`) or path to local checkpoint
    @param label2id: [optional] label2id dictionary, which is not needed for already trained NER model,
         but need for fine-tuning model on NER
    @param use_auth_token: [optional] Huggingface transformers argument of `use_auth_token`
    @param local_files_only: [optional] Huggingface transformers argument of `local_files_only`
    @return: AutoModelForTokenClassification object
    """
    if label2id is not None:
        config = AutoConfig.from_pretrained(
            model,
            use_auth_token=use_auth_token,
            num_labels=len(label2id),
            id2label={v: k for k, v in label2id.items()},
            label2id=label2id,
            local_files_only=local_files_only)
    else:
        config = AutoConfig.from_pretrained(model, use_auth_token=use_auth_token, local_files_only=local_files_only)
    return AutoModelForTokenClassification.from_pretrained(
        model, config=config, use_auth_token=use_auth_token, local_files_only=local_files_only)

