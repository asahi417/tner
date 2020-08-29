""" Fetch dataset for NLP task """
import os
import logging
import zipfile
from logging.config import dictConfig
from typing import Dict

from .mecab_wrapper import MeCabWrapper

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
STOPWORDS = ['None', '#']
# Unified label set across different dataset
SHARED_NER_LABEL = {
    "date": ["DATE", "DAT"],
    "location": ["LOCATION", "LOC", "location"],
    "organization": ["ORGANIZATION", "ORG", "corporation", "group", "organization"],
    "person": ["PERSON", "PSN", "person", "PER"],
    "time": ["TIME", "TIM"],
    "artifact": ["ARTIFACT", "ART", "creative-work", "product", "artifact"],
    "percent": ["PERCENT", "PNT"],
    "other": ["OTHER", "MISC"],
    "money": ["MONEY", "MNY"]
}
CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)
__all__ = "get_dataset_ner"


def decode_file(file_name: str, data_path: str, label_to_id: Dict, fix_label_dict: bool):
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
                if word in STOPWORDS:
                    continue
                sentence.append(word)

                # convert tag into unified label set
                if tag != 'O':  # map tag by custom dictionary
                    location = tag.split('-')[0]
                    mention = '-'.join(tag.split('-')[1:])
                    fixed_mention = [k for k, v in SHARED_NER_LABEL.items() if mention in v]
                    if len(fixed_mention) == 0:
                        tag = 'O'
                    else:
                        tag = '-'.join([location, fixed_mention[0]])

                # if label dict is fixed, unknown tag type will be ignored
                if tag not in label_to_id.keys() and fix_label_dict:
                    tag = 'O'
                elif tag not in label_to_id.keys() and not fix_label_dict:
                    label_to_id[tag] = len(label_to_id)

                entity.append(label_to_id[tag])

    return label_to_id, {"data": inputs, "label": labels}


def decode_all_files(files: Dict, data_path: str, label_to_id: Dict, fix_label_dict: bool):
    data_split = dict()
    for name, filepath in files.items():
        label_to_id, data_dict = decode_file(
            filepath, data_path=data_path, label_to_id=label_to_id, fix_label_dict=fix_label_dict)
        data_split[name] = data_dict
        LOGGER.info('dataset {0}/{1}: {2} entries'.format(data_path, filepath, len(data_dict['data'])))
    return data_split, label_to_id


def get_dataset_ner(data_name: str = 'wnut_17',
                    label_to_id: dict = None):
    """ download dataset file and return dictionary including training/validation split

    :param data_name: data set name or path to the data
    :param label_to_id: fixed dictionary of (label: id). If given, ignore other labels
    :return: formatted data, label_to_id
    """

    data_path = os.path.join(CACHE_DIR, data_name)
    language = 'en'
    LOGGER.info('data_name: {}'.format(data_name))
    if data_name == 'conll_2003':
        files_info = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('git clone https://github.com/mohammadKhalifa/xlm-roberta-ner')
            os.system('mv ./xlm-roberta-ner/data/coNLL-2003/* {}/'.format(data_path))
            os.system('rm -rf ./xlm-roberta-ner')
    elif data_name == 'wnut_17':
        files_info = {'train': 'train.txt.tmp', 'valid': 'dev.txt.tmp', 'test': 'test.txt.tmp'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/wnut17train.conll'  | tr '\t' ' ' > {}/train.txt.tmp".format(data_path))
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/emerging.dev.conll' | tr '\t' ' ' > {}/dev.txt.tmp".format(data_path))
            os.system("curl -L 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated' | tr '\t' ' ' > {}/test.txt.tmp".format(data_path))
    elif data_name == 'wiki-ja':
        files_info = {'valid': 'valid.txt', 'train': 'train.txt'}
        language = 'ja'
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('git clone https://github.com/Hironsan/IOB2Corpus')
            os.system('mv ./IOB2Corpus/hironsan.txt {}/valid.txt'.format(data_path))
            os.system('mv ./IOB2Corpus/ja.wikipedia.conll {}/train.txt'.format(data_path))
            os.system('rm -rf ./IOB2Corpus')
    elif 'panx_dataset' in data_name:
        files_info = {'valid': 'dev.txt', 'train': 'train.txt', 'test': 'test.txt'}
        panx_la = data_name.split('/')[1]
        if not os.path.exists(data_path):
            if not os.path.exists(os.path.join(CACHE_DIR, 'panx_dataset')):
                assert os.path.exists(os.path.join(CACHE_DIR, 'AmazonPhotos.zip')), 'download from {0} to {1}'.format(
                    'https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN', CACHE_DIR)
                with zipfile.ZipFile('{}/AmazonPhotos.zip'.format(CACHE_DIR), 'r') as zip_ref:
                    zip_ref.extractall('{}/'.format(CACHE_DIR))
            os.makedirs(data_path, exist_ok=True)
            os.system('tar xzf {0}/panx_dataset/{1}.tar.gz -C {2}'.format(CACHE_DIR, panx_la, data_path))
            for v in files_info.values():
                os.system("sed -e 's/{0}://g' {1}/{2} > {1}/{2}.txt".format(panx_la, data_path, v.replace('.txt', '')))
                os.system("rm -rf {0}/{1}".format(data_path, v.replace('.txt', '')))
        if panx_la == 'ja':
            language = 'ja'
    else:
        # for custom data
        if not os.path.exists(data_path):
            raise ValueError('unknown dataset: %s' % data_name)
        else:
            files_info = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}

    fix_label_dict = True
    if label_to_id is None:
        label_to_id = dict()
        fix_label_dict = False

    data_split_all, label_to_id = decode_all_files(
        files_info, data_path, label_to_id=label_to_id, fix_label_dict=fix_label_dict)

    if language == 'ja':
        id_to_label = {v: k for k, v in label_to_id.items()}
        label_fixer = MeCabWrapper()
        data, label = [], []
        for k, v in data_split_all.items():
            for x, y in zip(v['data'], v['label']):
                y = [id_to_label[_y] for _y in y]
                _data, _label = label_fixer.fix_ja_labels(inputs=x, labels=y)
                _label = [label_to_id[_y] for _y in _label]
                data.append(_data)
                label.append(_label)
            v['data'] = data
            v['label'] = label

    return data_split_all, label_to_id, language
