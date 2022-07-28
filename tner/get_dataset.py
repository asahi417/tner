""" Fetch/format NER dataset """
import os
import logging
import requests
import json
import hashlib
from typing import Dict
from itertools import chain
from os.path import join as pj

from datasets import load_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
STOPWORDS = ['None', '#']
STOPTAGS = ['junk']
CACHE_DIR = f"{os.path.expanduser('~')}/.cache/tner"
CHECKSUM_SHARED_LABEL = '460207e44f2b33737de03c29c7b02a3f'


__all__ = (
    "get_dataset",
    "concat_dataset",
    "get_shared_label"
)


def get_shared_label(cache_dir: str = None):
    """ universal label set to unify the NER labels """
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    url = f"https://raw.githubusercontent.com/asahi417/tner/master/unified_label2id.json"
    path = pj(cache_dir, "unified_label2id.json")
    if os.path.exists(path):
        checksum = hashlib.md5(open(path, 'rb').read()).hexdigest()
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
    """ Get `label2id` from TNER huggingface dataset https://huggingface.co/tner """
    cache_dir = CACHE_DIR if cache_dir is None else cache_dir
    url = f"https://huggingface.co/datasets/{dataset}/raw/main/dataset/label.json"
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


def get_hf_dataset(dataset: str = 'tner/conll2003', data_type: str = None, cache_dir: str = None):
    """ load dataset from TNER huggingface dataset https://huggingface.co/tner """
    if data_type is not None:
        data = load_dataset(dataset, data_type)
    else:
        data = load_dataset(dataset)
    label2id = get_hf_label2id(dataset, cache_dir)
    data = {k: {"tokens": data[k]["tokens"], "tags": data[k]["tags"]} for k in data.keys()}
    return data, label2id


def load_coll_format_file(data_path: str, label2id: Dict = None):
    """ load dataset from local IOB format file """
    inputs, labels, seen_entity = [], [], []
    with open(data_path, 'r') as f:
        sentence, entity = [], []
        for n, line in enumerate(f):
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(sentence) != 0:
                    assert len(sentence) == len(entity)
                    inputs.append(sentence)
                    labels.append(entity)
                    sentence, entity = [], []
            else:
                ls = line.split()
                if len(ls) < 2:
                    logging.warning(f'skip {ls}: too short')
                    continue
                # Examples could have no label for mode = "test"
                word, tag = ls[0], ls[-1]
                if tag in STOPTAGS:
                    logging.warning(f'skip tag {tag} from {ls}: STOPTAGS')
                    continue
                if word in STOPWORDS:
                    logging.warning(f'skip word {word} from {ls}: STOPWORDS')
                    continue
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
    return {"tokens": inputs, "tags": labels}, label2id


def get_conll_format_dataset(path_to_dataset: Dict, label2id: Dict = None):
    """ load dataset from local IOB files"""
    full_data = {}
    for file_name, file_path in path_to_dataset.items():
        assert os.path.exists(file_path), file_path
        data, label2id = load_coll_format_file(file_path, label2id)
        full_data[file_name] = data
    return full_data, label2id


def get_dataset(dataset: str = None,
                data_path: Dict = None,
                label2id: Dict = None,
                data_type: str = None,
                cache_dir: str = None):
    """ get NER dataset
    What should we do when we receive custom label2id for huggingface dataset?
    - normalize it by the SHARED_LABELS (in the first place, how we maintain this?)
    - put it on the github and maintain via the hash score of it? (sort of versioning)
    """
    if dataset is not None:
        if data_path is not None:
            logging.warning(f"data_path ({data_path}) is provided but ignored as dataset ({dataset}) is prioritized")
        data, label2id = get_hf_dataset(dataset, data_type=data_type, cache_dir=cache_dir)
    else:
        assert data_path is not None, "need either of `dataset` or `data_path`"
        data, label2id = get_conll_format_dataset(data_path, label2id=label2id)
    return data, label2id


def concat_dataset(list_of_data, cache_dir: str = None):
    """
    list_of_data = [(data_A, label2id_A), (data_B, label2id_B), ... ]
    """

    # unify label2id
    unified_label_set = get_shared_label(cache_dir)
    all_labels = []
    normalized_entities = {}
    for _, label2id in list_of_data:
        all_labels += list(label2id.keys())
        entities = set(i.rsplit('-', 1)[-1] for i in label2id.keys() if i != 'O')

        for entity in entities:
            normalized_entity = [k for k, v in unified_label_set.items() if entity in v]
            assert len(normalized_entity) <= 1, f'duplicated entity found in the shared label set\n {normalized_entity} \n {entity}'
            if len(normalized_entity) == 0:
                logging.warning(f'Entity `{entity}` is not found in the shared label set {unified_label_set}. '
                                f'Original entity (`{entity}`) will be used as label.')
                normalized_entities[entity] = entity
            else:
                normalized_entities[entity] = normalized_entity[0]
    all_labels = sorted([i for i in set(all_labels) if i != "O"])
    normalized_labels = [f"{i.rsplit('-', 1)[0]}-{normalized_entities[i.rsplit('-', 1)[1]]}" for i in all_labels]
    normalized_label2id = {k: n for n, k in enumerate(sorted(normalized_labels))}
    normalized_label2id.update({"O": len(normalized_label2id)})

    # update labels & concat data
    concat_tokens = {}
    concat_tags = {}
    for data, label2id in list_of_data:
        id2label = {v: k for k, v in label2id.items()}
        for _split in data.keys():
            if _split not in concat_tokens:
                concat_tokens[_split] = []
                concat_tags[_split] = []
            concat_tokens[_split] += data[_split]['tokens']
            for tags in data[_split]['tags']:
                normalized_tag = []
                for t in tags:
                    if id2label[t] != 'O':
                        t = f"{id2label[t].rsplit('-', 1)[0]}-{normalized_entities[id2label[t].rsplit('-', 1)[1]]}"
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
