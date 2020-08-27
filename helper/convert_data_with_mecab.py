""" convert character-level dataset into mecab tokenized dataset """
import argparse
import os
import logging
from glob import glob
from logging.config import dictConfig
from itertools import accumulate


# dependency
import gsutil_util
from mecab_tokenizer import JaTokenizer

Tokenizer = JaTokenizer()
dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
NUM_WORKER = int(os.getenv("NUM_WORKER", '4'))
PROGRESS_INTERVAL = int(os.getenv("PROGRESS_INTERVAL", '100'))
CACHE_DIR = os.getenv("CACHE_DIR", './cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
STOPWORDS = ['None', '#']
PREFIX = 'ja:'


def get_dataset(data_name: str = 'ner-cogent-ja', label_to_id: dict = None, allow_update: bool=True):
    """ download dataset file and return dictionary including training/validation split """
    label_to_id = dict() if label_to_id is None else label_to_id
    data_path = os.path.join(CACHE_DIR, data_name)
    if not os.path.exists(data_path):
        gsutil_util.download(
            bucket_name='nlp-entity-recognition',
            destination_file_name=data_path + '.zip',
            source_blob_name=data_name + '.zip')

    def decode_file(file_name, _label_to_id: dict):
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
                    tag = 'O' if len(ls) < 2 else ls[-1]
                    if tag == 'junk':
                        continue
                    # sentence.append(ls[0].split(':')[-1])  # panex is in a form of `ja:ク O`
                    word = ls[0].replace(PREFIX, '')
                    if word in STOPWORDS:
                        continue
                    sentence.append(word)
                    if tag not in _label_to_id.keys():
                        assert allow_update
                        _label_to_id[tag] = len(_label_to_id)
                    entity.append(_label_to_id[tag])
        return _label_to_id, {"data": inputs, "label": labels}

    data_split = dict()
    for _file in glob(data_path + '/*.txt'):
        filepath = _file.split('/')[-1]
        label_to_id, data_dict = decode_file(filepath, _label_to_id=label_to_id)
        data_split[filepath.replace('.txt', '')] = data_dict
        LOGGER.info('dataset %s/%s: %i entries' % (data_name, filepath, len(data_dict['data'])))
    if allow_update:
        return data_split, label_to_id
    else:
        return data_split


def fix_labels(inputs, labels):
    tokens = Tokenizer(''.join(inputs))
    tokens_len = [len(i) for i in tokens]
    assert sum(tokens_len) == len(labels)
    cum_tokens_len = list(accumulate(tokens_len))
    new_labels = [labels[i] for i in [0] + cum_tokens_len[:-1]]
    new_labels_fixed = []
    for i in range(len(new_labels)):
        if i == 0 or new_labels[i] == 'O':
            new_labels_fixed.append(new_labels[i])
        else:
            loc, mention = new_labels[i].split('-')
            if loc == 'B':
                new_labels_fixed.append(new_labels[i])
            else:
                if new_labels[i - 1] == 'O':
                    new_labels_fixed.append('B-{}'.format(mention))
                else:
                    prev_loc, prev_mention = new_labels[i - 1].split('-')
                    if prev_mention == mention:
                        new_labels_fixed.append('I-{}'.format(mention))
                    else:
                        new_labels_fixed.append('B-{}'.format(mention))
    assert len(tokens) == len(new_labels_fixed)
    return tokens, new_labels_fixed


def get_options():
    parser = argparse.ArgumentParser(
        description='finetune transformers to sentiment analysis',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--data', help='dataset `ner-cogent-ja`', default='ner-cogent-ja', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    opt = get_options()
    data_path_mecab = os.path.join(CACHE_DIR, opt.data + '-mecab')
    os.makedirs(data_path_mecab, exist_ok=True)

    tokenizer = JaTokenizer()
    data, _label_to_id = get_dataset(opt.data)
    data_path = os.path.join(CACHE_DIR, opt.data + '-mecab')

    _id_to_label = {v: k for k, v in _label_to_id.items()}
    for k, v in data.items():
        print(k)
        with open(os.path.join(data_path_mecab, '{}.txt'.format(k)), 'w') as f_writer:
            for _in, _label in zip(v['data'], v['label']):
                _label = [_id_to_label[i] for i in _label]
                _in = ''.join(_in)
                fixed_in, fixed_label = fix_labels(_in, _label)
                if len(fixed_in) == 0:
                    continue
                for x, y in zip(fixed_in, fixed_label):
                    f_writer.write('{} {}\n'.format(x, y))
                f_writer.write('\n')
