import logging
import pickle
from typing import List, Dict
from tqdm import tqdm


# For evaluation (span-F1 score)
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from scipy.stats import bootstrap


def pickle_save(obj, path: str):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def f1_with_ci(label_list,
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
            res = bootstrap((data,), get_f1, confidence_level=c * 0.01, method='percentile', n_resamples=n_resamples,
                            random_state=np.random.default_rng(random_seed))
            ci[str(c)] = [res.confidence_interval.low, res.confidence_interval.high]
    return mean_score, ci


def span_f1(pred_list: List,
            label_list: List,
            label2id: Dict,
            span_detection_mode: bool = False,
            return_ci: bool = False):

    if span_detection_mode:
        return_ci = False

        def convert_to_binary_mask(entity_label):
            if entity_label == 'O':
                return entity_label
            prefix = entity_label.split('-')[0]  # B or I
            return '{}-entity'.format(prefix)

        label_list = [[convert_to_binary_mask(_i) for _i in i] for i in label_list]
        pred_list = [[convert_to_binary_mask(_i) for _i in i] for i in pred_list]

    # compute metrics
    logging.info('\n{}'.format(classification_report(label_list, pred_list)))
    m_micro, ci_micro = f1_with_ci(label_list, pred_list, average='micro', return_ci=return_ci)
    m_macro, ci_macro = f1_with_ci(label_list, pred_list, average='macro', return_ci=return_ci)
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
    target_names = sorted([k.replace('B-', '') for k in label2id.keys() if k.startswith('B-')])
    if not span_detection_mode:
        metric["per_entity_metric"] = {}
        for t in target_names:
            _label_list = [[_i if _i.endswith(t) else 'O' for _i in i] for i in label_list]
            _pred_list = [[_i if _i.endswith(t) else 'O' for _i in i] for i in pred_list]
            m, ci = f1_with_ci(_label_list, _pred_list, return_ci=return_ci)
            metric["per_entity_metric"][t] = {
                "f1": m,
                "f1_ci": ci,
                "precision": precision_score(_label_list, _pred_list),
                "recall": recall_score(_label_list, _pred_list)}
    return metric


def decode_ner_tags(tag_sequence, input_sequence, probability_sequence=None):
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

