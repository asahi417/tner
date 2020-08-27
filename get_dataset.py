""" Fetch dataset for NLP task """
import os
import logging
from logging.config import dictConfig
from typing import Dict, List

import torch


dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
STOPWORDS = ['None', '#']
__all__ = ("Dataset", "get_dataset_ner", "get_dataset_sentiment")
SHARED_NER_LABEL = {
        "date": ["DATE", "DAT"],
        "location": ["LOCATION", "LOC", "location"],
        "organization": ["ORGANIZATION", "ORG", "corporation", "group", "organization"],
        "person": ["PERSON", "PSN", "person"],
        "time": ["TIME", "TIM"],
        "artifact": ["ARTIFACT", "ART", "creative-work", "product", "artifact"],
        "percent": ["PERCENT", "PNT"], "other": ["OTHER", "MISC"], "money": ["MONEY", "MNY"]
    }


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


def apply_global_label(label_dict):
    new_label_dict = {}
    for tag, _id in label_to_id.items():
        location = tag.split('-')[0]
        mention = '-'.join(tag.split('-')[1:])
        fixed_mention = [k for k, v in SHARED_NER_LABEL.items() if mention in v]
        if len(fixed_mention) == 0:
            raise ValueError('undefined mention found: {}'.format(mention))
        if len(fixed_mention) > 1:
            raise ValueError('duplicated mention found: {}'.format(mention))
        fixed_mention = fixed_mention[0]
        new_label_dict['-'.join([location, fixed_mention])] = _id
    return new_label_dict


def get_dataset_ner(data_name: str = 'wnut_17',
                    label_to_id: dict = None,
                    allow_update: bool = True,
                    cache_dir: str = './cache'):
    """ download dataset file and return dictionary including training/validation split """
    label_to_id = dict() if label_to_id is None else label_to_id
    data_path = os.path.join(cache_dir, data_name)

    def decode_file(file_name, _label_to_id: Dict, custom_mapper: Dict = None):
        inputs, labels = [], []
        with open(os.path.join(data_path, file_name), 'r') as f:
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
                        continue
                    # Examples could have no label for mode = "test"
                    word, tag = ls[0], ls[-1]
                    if tag == 'junk':
                        continue
                    if custom_mapper and tag != 'O':  # map tag by custom dictionary
                        location, mention = tag.split('-')[0], '-'.join(tag.split('-')[1:])
                        if mention not in custom_mapper.values():
                            if mention not in custom_mapper.keys():
                                tag = 'O'
                            else:
                                tag = location + '-' + custom_mapper[mention]
                    if word in STOPWORDS:
                        continue
                    sentence.append(word)
                    if tag not in _label_to_id.keys():
                        assert allow_update
                        _label_to_id[tag] = len(_label_to_id)
                    entity.append(_label_to_id[tag])

        return _label_to_id, {"data": inputs, "label": labels}

    def decode_all_files(files: Dict, _label_to_id, custom_mapper: Dict = None):
        data_split = dict()
        for name, filepath in files.items():
            _label_to_id, data_dict = decode_file(filepath, _label_to_id=_label_to_id, custom_mapper=custom_mapper)
            data_split[name] = data_dict
            LOGGER.info('dataset {}/{}: {} entries'.format(data_name, filepath, len(data_dict['data'])))
        return data_split, _label_to_id

    if data_name == 'conll_2003':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('git clone https://github.com/mohammadKhalifa/xlm-roberta-ner')
            os.system('mv ./xlm-roberta-ner/data/coNLL-2003/* {}/'.format(data_path))
            os.system('rm -rf ./xlm-roberta-ner')
        files_info = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}
        data_split_all, label_to_id = decode_all_files(files_info, label_to_id)
    elif data_name == 'wnut_17':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/wnut17train.conll'  | tr '\t' ' ' > {}/train.txt.tmp".format(data_path))
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/emerging.dev.conll' | tr '\t' ' ' > {}/dev.txt.tmp".format(data_path))
            os.system("curl -L 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated' | tr '\t' ' ' > {}/test.txt.tmp".format(data_path))
        files_info = {'train': 'train.txt.tmp', 'valid': 'dev.txt.tmp', 'test': 'test.txt.tmp'}
        data_split_all, label_to_id = decode_all_files(files_info, label_to_id)
    elif data_name == 'wiki-ja':
        label_mapper = {"DATE": "DAT", "LOCATION": "LOC", "ORGANIZATION": "ORG", "PERSON": "PSN", "TIME": "TIM",
                        "ARTIFACT": "ART", "PERCENT": "PNT", "OTHER": "MISC", "MONEY": "MNY"}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('git clone https://github.com/Hironsan/IOB2Corpus')
            os.system('mv ./IOB2Corpus/hironsan.txt {}/valid.txt'.format(data_path))
            os.system('mv ./IOB2Corpus/ja.wikipedia.conll {}/train.txt'.format(data_path))
            os.system('rm -rf ./IOB2Corpus')
        files_info = {'valid': 'valid.txt', 'train': 'train.txt'}
        data_split_all, label_to_id = decode_all_files(files_info, label_to_id, custom_mapper=label_mapper)
    else:
        # for custom data
        if not os.path.exists(data_path):
            raise ValueError('unknown dataset: %s' % data_name)
        else:
            files_info = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}
            data_split_all, label_to_id = decode_all_files(files_info, label_to_id)
    if allow_update:
        return data_split_all, label_to_id
    else:
        return data_split_all

