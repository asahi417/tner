""" NER dataset """
import os
import logging
import requests
import json
import hashlib
from unicodedata import normalize
from typing import Dict, List
from itertools import chain
from os.path import join as pj

from datasets import load_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
# STOPWORDS = ['None', '#']
# STOPTAGS = ['junk']
CACHE_DIR = f"{os.path.expanduser('~')}/.cache/tner"
CHECKSUM_SHARED_LABEL = '460207e44f2b33737de03c29c7b02a3f'


__all__ = (
    "get_dataset",
    "concat_dataset",
    "get_shared_label"
)


def get_shared_label(cache_dir: str = None):
    """ universal label set to unify the NER datasets

    @param cache_dir: cache directly
    @return: a dictionary mapping from label to id
    """
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    url = f"https://raw.githubusercontent.com/asahi417/tner/master/unified_label2id.json"
    path = pj(cache_dir, "unified_label2id.json")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        if CHECKSUM_SHARED_LABEL == checksum:
            with open(path) as f:
                label2id = json.load(f)
            return label2id
        else:
            logging.warning('local `unified_label2id.json` has wrong checksum')
    with open(path, "w") as f:
        logging.info(f'downloading `unified_label2id.json` from {url}')
        r = requests.get(url)
        label2id = json.loads(r.content)
        json.dump(label2id, f)
    file_checksum = hashlib.md5(open(path, 'rb').read()).hexdigest()
    assert file_checksum == CHECKSUM_SHARED_LABEL,\
        f"checksum inconsistency {file_checksum} != {CHECKSUM_SHARED_LABEL}"
    return label2id


def get_hf_label2id(dataset, cache_dir: str = None):
    """ get `label2id` from TNER huggingface dataset https://huggingface.co/tner

    @param dataset: dataset name
    @param cache_dir: [optional] huggingface cache directly
    @return: a dictionary mapping from label to id
    """
    url = f"https://huggingface.co/datasets/tner/label2id/raw/main/files/{os.path.basename(dataset)}.json"
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    # url = f"https://huggingface.co/datasets/{dataset}/raw/main/dataset/label.multi.json"
    path = pj(cache_dir, f"{dataset}.label2id.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path) as f:
            label2id = json.load(f)
    except Exception:
        with open(path, "w") as f:
            r = requests.get(url)
            label2id = json.loads(r.content)
            json.dump(label2id, f)
    return label2id


def get_hf_dataset(dataset: str = 'tner/conll2003',
                   dataset_name: str = None,
                   cache_dir: str = None,
                   use_auth_token: bool = False):
    """ load dataset from TNER huggingface dataset https://huggingface.co/tner

    @param dataset: dataset alias on huggingface dataset hub
    @param dataset_name: [optional] dataset name to specify
    @param cache_dir: [optional] huggingface cache directly
    @return: (data, label2id)
        - data: a dictionary of {"tokens": [list of tokens], "tags": [list of tags]}
        - label2id: a dictionary mapping from label to id
    """
    if dataset_name is not None:
        data = load_dataset(dataset, dataset_name, use_auth_token=use_auth_token)
    else:
        data = load_dataset(dataset, use_auth_token=use_auth_token)
    label2id = get_hf_label2id(dataset, cache_dir)
    data = {k: {"tokens": data[k]["tokens"], "tags": data[k]["tags"]} for k in data.keys()}
    return data, label2id


def load_conll_format_file(data_path: str, label2id: Dict = None):
    """ load dataset from local IOB format file

    @param data_path: path to iob file
    @param label2id: [optional] dictionary of label2id (generate from dataset as default )
    @return: (data, label2id)
        - data: a dictionary of {"tokens": [list of tokens], "tags": [list of tags]}
        - label2id: a dictionary mapping from label to id
    """
    inputs, labels, seen_entity = [], [], []
    with open(data_path, 'r') as f:
        sentence, entity = [], []
        for n, line_raw in enumerate(f):
            line = normalize('NFKD', line_raw).strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(sentence) != 0:
                    assert len(sentence) == len(entity)
                    inputs.append(sentence)
                    labels.append(entity)
                    sentence, entity = [], []
            else:
                ls = line.split()
                if len(ls) < 2:
                    if line_raw.startswith('O'):
                        logging.warning(f'skip {ls} (line {n} of file {data_path}): '
                                        f'missing token (should be word and tag separated by '
                                        f'a half-space, eg. `London B-LOC`)')
                        continue
                    else:
                        ls = ['', ls[0]]
                # Examples could have no label for mode = "test"
                word, tag = ls[0], ls[-1]
                sentence.append(word)
                entity.append(tag)

        if len(sentence) != 0:
            assert len(sentence) == len(entity)
            inputs.append(sentence)
            labels.append(entity)

    all_labels = sorted(list(set(list(chain(*labels)))))
    if label2id is None:
        label2id = {t: n for n, t in enumerate(all_labels)}
    else:
        labels_not_found = [i for i in all_labels if i not in label2id]
        if len(labels_not_found) > 0:
            logging.warning(f'found entities not in the label2id (label2id was updated):\n\t - {labels_not_found}')
            label2id.update({i: len(label2id) + n for n, i in enumerate(labels_not_found)})
        assert all(i in label2id for i in all_labels), \
            f"label2id is not covering all the entity \n \t- {label2id} \n \t- {all_labels}"
    labels = [[label2id[__l] for __l in _l] for _l in labels]
    data = {"tokens": inputs, "tags": labels}
    return data, label2id


def get_conll_format_dataset(local_dataset: Dict):
    """ load dataset from local IOB files

    @param local_dataset: a dictionary of paths to local BIO files eg.
        {"train": "examples/local_dataset_sample/train.txt", "test": "examples/local_dataset_sample/test.txt"}
    @return: (data, label2id)
        - data: a dictionary of {"train": {"tokens": [list of tokens], "tags": [list of tags]}}
        - label2id: a dictionary mapping from label to id
    """
    data = {}
    label2id = None
    for file_name in sorted(local_dataset.keys()):
        file_path = local_dataset[file_name]
        assert os.path.exists(file_path), file_path
        _data, label2id = load_conll_format_file(file_path, label2id)
        data[file_name] = _data
    return data, label2id


def get_dataset_single(dataset: str = None,
                       local_dataset: Dict = None,
                       dataset_name: str = None,
                       cache_dir: str = None,
                       use_auth_token: bool = False):
    """ get NER dataset

    @param dataset: dataset name on huggingface tner organization (https://huggingface.co/datasets?search=tner)
    @param local_dataset: a dictionary of paths to local BIO files eg.
        {"train": "examples/local_dataset_sample/train.txt", "test": "examples/local_dataset_sample/test.txt"}
    @param dataset_name: [optional] data name of huggingface dataset
    @param cache_dir: [optional] cache directly
    @return: (data, label2id)
        - data: a dictionary of {"train": {"tokens": [list of tokens], "tags": [list of tags]}}
        - label2id: a dictionary mapping from label to id
    """
    if dataset is not None:
        if local_dataset is not None:
            logging.warning(f"local_dataset ({local_dataset}) is provided but ignored as dataset ({dataset}) is given")
        data, label2id = get_hf_dataset(
            dataset, dataset_name=dataset_name, cache_dir=cache_dir, use_auth_token=use_auth_token
        )

    else:
        assert local_dataset is not None, "need either of `dataset` or `local_dataset`"
        data, label2id = get_conll_format_dataset(local_dataset)
    return data, label2id


def concat_dataset(list_of_data, cache_dir: str = None, label2id: Dict = None):
    """ concat multiple NER dataset with a unified label set

    @param list_of_data: a list of output from `get_dataset` eg. [(data_A, label2id_A), (data_B, label2id_B), ... ]
    @param cache_dir: [optional] cache directly
    @param label2id: [optional] define label2id map
    @return: (data, label2id)
        - data: a dictionary of {"train": {"tokens": [list of tokens], "tags": [list of tags]}}
        - label2id: a dictionary mapping from label to id
    """
    # unify label2id
    unified_label_set = get_shared_label(cache_dir)
    all_labels = []
    normalized_entities = {}
    for _, _label2id in list_of_data:
        all_labels += list(_label2id.keys())
        entities = set('-'.join(i.split('-')[1:]) for i in _label2id.keys() if i != 'O')
        for entity in entities:
            normalized_entity = [k for k, v in unified_label_set.items() if entity in v]
            assert len(normalized_entity) <= 1, f'duplicated entity found in the shared label set\n {normalized_entity} \n {entity}'
            if len(normalized_entity) == 0:
                # logging.warning(f'Entity `{entity}` is not found in the shared label set {unified_label_set}. '
                #               f'Original entity (`{entity}`) will be used as label.')
                normalized_entities[entity] = entity
            else:
                normalized_entities[entity] = normalized_entity[0]
    all_labels = sorted([i for i in set(all_labels) if i != "O"])
    normalized_labels = [f"{i.split('-')[0]}-{normalized_entities['-'.join(i.split('-')[1:])]}" for i in all_labels]
    normalized_labels = list(set(normalized_labels))
    # input(normalized_labels)
    if label2id is not None:
        assert all(i in label2id.keys() for i in normalized_labels),\
            f"missing entity in label2id {label2id.keys()}: {normalized_labels}"
        normalized_label2id = label2id
    else:
        normalized_label2id = {k: n for n, k in enumerate(sorted(normalized_labels))}
        normalized_label2id.update({"O": len(normalized_label2id)})
        # input(normalized_label2id)

    # update labels & concat data
    concat_tokens = {}
    concat_tags = {}
    for data, _label2id in list_of_data:
        id2label = {v: k for k, v in _label2id.items()}
        for _split in data.keys():
            if _split not in concat_tokens:
                concat_tokens[_split] = []
                concat_tags[_split] = []
            concat_tokens[_split] += data[_split]['tokens']
            for tags in data[_split]['tags']:
                normalized_tag = []
                for t in tags:
                    if id2label[t] != 'O':
                        t = f"{id2label[t].split('-')[0]}-{normalized_entities['-'.join(id2label[t].split('-')[1:])]}"
                    else:
                        t = id2label[t]
                    normalized_tag.append(normalized_label2id[t])
                concat_tags[_split].append(normalized_tag)

    # sanity check
    assert concat_tags.keys() == concat_tokens.keys(), f"{concat_tags.keys()} != {concat_tokens.keys()}"
    for s in concat_tags.keys():
        assert len(concat_tags[s]) == len(concat_tokens[s]), f"{len(concat_tags[s])} != {len(concat_tokens[s])}"
        assert all(len(a) == len(b) for a, b in zip(concat_tags[s], concat_tokens[s]))
    data = {s: {"tokens": concat_tokens[s], "tags": concat_tags[s]} for s in concat_tags.keys()}
    return data, normalized_label2id


def get_dataset(dataset: List or str = None,
                local_dataset: List or Dict = None,
                dataset_name: List or str = None,
                concat_label2id: Dict = None,
                cache_dir: str = None,
                use_auth_token: bool = False):
    """ get NER datasets (concat mutiple datasets)

    @param dataset: dataset name (or a list of it) on huggingface tner organization (https://huggingface.co/datasets?search=tner)
            (eg. "tner/conll2003", ["tner/conll2003", "tner/ontonotes5"]]
    @param local_dataset: a dictionary (or a list) of paths to local BIO files eg.
            {"train": "examples/local_dataset_sample/train.txt", "test": "examples/local_dataset_sample/test.txt"}
    @param dataset_name: [optional] data name of huggingface dataset (should be same length as the `dataset`)
    @param concat_label2id: [optional] define label2id map for multiple dataset concatenation (nothing to do with single data)
    @param cache_dir: [optional] cache directly
    @return: (data, label2id)
        - data: a dictionary of {"train": {"tokens": [list of tokens], "tags": [list of tags]}}
        - label2id: a dictionary mapping from label to id
    """
    assert dataset is not None or local_dataset is not None, "`datasets` or `local_datasets` should be provided"
    dataset_list = []
    # load huggingface dataset
    if dataset is not None:
        if type(dataset) is str:
            assert dataset_name is None or type(dataset_name) is str, \
                f"`dataset_name` should be string but given {dataset_name}"
            data, label2id = get_dataset_single(
                dataset=dataset, dataset_name=dataset_name, cache_dir=cache_dir, use_auth_token=use_auth_token)
            dataset_list.append((data, label2id))
        else:
            assert dataset_name is None or (
                    type(dataset_name) is list and
                    len(dataset) == len(dataset_name)), \
                f"dataset_name not matched: {dataset} vs {dataset_name}"
            for n, d in enumerate(dataset):
                data, label2id = get_dataset_single(
                    dataset=d,
                    dataset_name=dataset_name[n] if dataset_name is not None else None,
                    cache_dir=cache_dir
                )
                dataset_list.append((data, label2id))
    # load custom dataset
    if local_dataset is not None:
        if type(local_dataset) is dict:
            data, label2id = get_dataset_single(local_dataset=local_dataset, cache_dir=cache_dir)
            dataset_list.append((data, label2id))
        else:
            for d in local_dataset:
                data, label2id = get_dataset_single(local_dataset=d, cache_dir=cache_dir)
                dataset_list.append((data, label2id))
    # concat datasets
    if len(dataset_list) > 1:
        logging.info(f'concat {len(dataset_list)} datasets')
        data, label2id = concat_dataset(dataset_list, label2id=concat_label2id, cache_dir=cache_dir)
    else:
        if concat_label2id is not None:
            logging.warning(f'concat_label2id is specified {concat_label2id} but not changed as is only one dataset')
        data, label2id = dataset_list[0]
    return data, label2id
