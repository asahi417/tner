""" Fetch dataset for NLP task """
import os
import logging
from logging.config import dictConfig
from typing import Dict


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

__all__ = "get_dataset_ner"


def get_dataset_ner(data_name: str = 'wnut_17',
                    label_to_id: dict = None,
                    cache_dir: str = './cache'):
    """ download dataset file and return dictionary including training/validation split

    :param data_name: data set name or path to the data
    :param label_to_id: fixed dictionary of (label: id). If given, ignore other labels
    :param cache_dir:
    :return: formatted data, label_to_id
    """
    fix_label_dict = True
    if label_to_id is None:
        label_to_id = dict()
        fix_label_dict = False

    data_path = os.path.join(cache_dir, data_name)
    language = 'en'

    def decode_file(file_name, _label_to_id: Dict):
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
                    if tag not in _label_to_id.keys() and fix_label_dict:
                        tag = 'O'
                    elif tag not in _label_to_id.keys() and not fix_label_dict:
                        _label_to_id[tag] = len(_label_to_id)

                    entity.append(_label_to_id[tag])

        return _label_to_id, {"data": inputs, "label": labels}

    def decode_all_files(files: Dict, _label_to_id):
        data_split = dict()
        for name, filepath in files.items():
            _label_to_id, data_dict = decode_file(filepath, _label_to_id=_label_to_id)
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
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('git clone https://github.com/Hironsan/IOB2Corpus')
            os.system('mv ./IOB2Corpus/hironsan.txt {}/valid.txt'.format(data_path))
            os.system('mv ./IOB2Corpus/ja.wikipedia.conll {}/train.txt'.format(data_path))
            os.system('rm -rf ./IOB2Corpus')
        files_info = {'valid': 'valid.txt', 'train': 'train.txt'}
        data_split_all, label_to_id = decode_all_files(files_info, label_to_id)
        language = 'ja'
    # elif 'panx' in data_name:


    else:
        # for custom data
        if not os.path.exists(data_path):
            raise ValueError('unknown dataset: %s' % data_name)
        else:
            files_info = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}
            data_split_all, label_to_id = decode_all_files(files_info, label_to_id)

    return data_split_all, label_to_id, language


if __name__ == '__main__':
    # a, b = get_dataset_ner('conll_2003')
    test = {"B-organization": 0, "O": 1, "B-other": 2, "B-person": 3, "I-person": 4, "B-location": 5, "I-organization": 6, "I-other": 7, "I-location": 8}
    a, b, c = get_dataset_ner('wiki-ja', label_to_id=test, allow_update=False)
    a, b, c = get_dataset_ner('wiki-ja')
    print(b)
