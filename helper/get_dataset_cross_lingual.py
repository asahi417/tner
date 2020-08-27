""" WIP Fetch dataset for NLP task """
import os
import logging
from itertools import accumulate
from logging.config import dictConfig
from typing import Dict

import torchtext
import MeCab

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()
STOPWORDS = ['None', '#']
__all__ = ("get_dataset_ner", "get_dataset_sentiment")


class JaTokenizer:
    """ MeCab tokenizer with

     Usage
    --------------
    >>> tokenizer = JaTokenizer()
    >>> tokenizer("日本ではサラリーマンが金曜日を華金と呼ぶ")
    ['日本', 'で', 'は', 'サラリーマン', 'が', '金曜日', 'を', '華金', 'と', '呼ぶ']
    >>> tokenizer("日本ではサラリーマンが金曜日を華金と呼ぶ", return_pos=True)
    [['日本', 'NOUN'], ['で', 'RANDOM'], ['は', 'RANDOM'], ['サラリーマン', 'NOUN'], ['が', 'RANDOM'], ['金曜日', 'NOUN'],
    ['を', 'RANDOM'], ['華金', 'NOUN'], ['と', 'RANDOM'], ['呼ぶ', 'VERB']]
    >>> tokenizer = JaTokenizer(False)
    >>> tokenizer("日本ではサラリーマンが金曜日を華金と呼ぶ")
    ['日本', 'で', 'は', 'サラリーマン', 'が', '金曜日', 'を', '華', '金', 'と', '呼ぶ']
    """
    pos_mapper = {
        "名詞": "NOUN",
        "形容詞": "ADJ",
        "動詞": "VERB",
        "RANDOM": "RANDOM"
    }

    def __init__(self, neologd: bool = True):
        self.__tagger = MeCab.Tagger()
        if neologd:
            f = os.popen('echo `mecab-config --dicdir`"/mecab-ipadic-neologd"')
            path_to_neologd = f.read().replace('\n', '')
            if os.path.exists(path_to_neologd):
                self.__tagger = MeCab.Tagger("-d {}".format(path_to_neologd))
            else:
                self.__tagger = MeCab.Tagger("")
        self.__tagger.parse('テスト')

    def __call__(self, sentence: str, return_pos: bool = False):
        def formatting(_raw, _pos):
            if not return_pos:
                return _raw
            try:
                _pos = self.pos_mapper[_pos]
            except KeyError:
                _pos = self.pos_mapper['RANDOM']
            return [_raw, _pos]

        parsed = self.__tagger.parse(sentence)
        if parsed is None:
            return None

        parsed_sentence = parsed.split("\n")
        out = [formatting(s.split("\t")[0], s.split("\t")[1].split(",")[0]) for s in parsed_sentence if "\t" in s]
        return out


def train_valid_split(_data, label, _export_path, _label_to_id):
    """ split into train/valid/text by 7:2:1 """
    assert len(_data) == len(label)
    os.makedirs(_export_path, exist_ok=True)
    id_to_label = {v: k for k, v in _label_to_id.items()}
    train_n = int(len(_data) * 0.7)
    valid_n = int(len(_data) * 0.2)

    with open(os.path.join(_export_path, 'train.txt'), 'w') as f:
        for x, y in zip(_data[:train_n], label[:train_n]):
            for _x, _y in zip(x, y):
                f.write("{} {}\n".format(_x, id_to_label[_y]))
            f.write('\n')

    with open(os.path.join(_export_path, 'valid.txt'), 'w') as f:
        for x, y in zip(_data[train_n:train_n + valid_n], label[train_n:train_n + valid_n]):
            for _x, _y in zip(x, y):
                f.write("{} {}\n".format(_x, id_to_label[_y]))
            f.write('\n')

    with open(os.path.join(_export_path, 'test.txt'), 'w') as f:
        for x, y in zip(_data[train_n + valid_n:], label[train_n + valid_n:]):
            for _x, _y in zip(x, y):
                f.write("{} {}\n".format(_x, id_to_label[_y]))
            f.write('\n')


def fix_labels(data: Dict, label_to_id: Dict, data_path: str):
    id_to_label = {v: k for k, v in label_to_id.items()}
    data_path = data_path + '-mecab'
    tokenizer = JaTokenizer()
    for k, v in data.items():
        if os.path.exists(os.path.join(data_path, '{}.txt'.format(k))):
            continue
        with open(os.path.join(data_path, '{}.txt'.format(k)), 'w') as f_writer:
            for _in, _label in zip(v['data'], v['label']):
                _label = [id_to_label[i] for i in _label]
                _in = ''.join(_in)

                tokens = tokenizer(''.join(_in))
                tokens_len = [len(i) for i in tokens]
                assert sum(tokens_len) == len(_label)
                cum_tokens_len = list(accumulate(tokens_len))
                new_labels = [_label[i] for i in [0] + cum_tokens_len[:-1]]
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
                if len(tokens) == 0:
                    continue
                for x, y in zip(tokens, new_labels_fixed):
                    f_writer.write('{} {}\n'.format(x, y))
                f_writer.write('\n')


def get_dataset_ner(data_name: str = 'wnut_17',
                    label_to_id: dict = None,
                    allow_update: bool = True,
                    cache_dir: str = './cache',
                    mecab: bool = False,
                    lang: str = None):
    """ download dataset file and return dictionary including training/validation split """
    label_to_id = dict() if label_to_id is None else label_to_id
    data_path = os.path.join(cache_dir, data_name)

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
                    tag = 'O' if len(ls) < 2 else ls[-1]
                    if tag == 'junk':
                        continue
                    # sentence.append(ls[0].split(':')[-1])  # panex is in a form of `ja:ク O`
                    word = ''.join(ls[0].split(':')[1:])
                    if word in STOPWORDS:
                        continue
                    sentence.append(word)
                    if tag not in _label_to_id.keys():
                        assert allow_update
                        _label_to_id[tag] = len(_label_to_id)
                    entity.append(_label_to_id[tag])

        return _label_to_id, {"data": inputs, "label": labels}

    def decode_all_files(files, _label_to_id):
        data_split = dict()
        for name, filepath in zip(['train', 'valid', 'test'], files):
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
        data_split_all, label_to_id = decode_all_files(['train.txt', 'valid.txt', 'test.txt'], label_to_id)
    # elif data_name == 'panx':
    #     if not os.path.exists(data_path):
    #         os.makedirs(data_path, exist_ok=True)
    #         raise ValueError(
    #             "Please download the AmazonPhotos.zip file on Amazon Cloud Drive mannually and save it to"
    #             "{}/AmazonPhotos.zip\n"
    #             "https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN".format(data_path)
    #         )
    elif data_name == 'wnut_17':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/wnut17train.conll'  | tr '\t' ' ' > {}/train.txt.tmp".format(data_path))
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/emerging.dev.conll' | tr '\t' ' ' > {}/dev.txt.tmp".format(data_path))
            os.system("curl -L 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated' | tr '\t' ' ' > {}/test.txt.tmp".format(data_path))
        data_split_all, label_to_id = decode_all_files(['train.txt.tmp', 'dev.txt.tmp', 'test.txt.tmp'], label_to_id)
    elif data_name in ['wiki-ja-500', 'wiki-news-ja-1000', 'wiki-ja-500-mecab', 'wiki-news-ja-1000-mecab']:
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('git clone https://github.com/Hironsan/IOB2Corpus')
            if data_name == 'wiki-ja-500':
                os.system('mv ./IOB2Corpus/hironsan.txt {}/tmp.txt'.format(data_path))
            else:
                os.system('mv ./IOB2Corpus/ja.wikipedia.conll {}/tmp.txt'.format(data_path))
            os.system('rm -rf ./IOB2Corpus')
            label_to_id, data = decode_file('tmp.txt', dict())
            train_valid_split(data['data'], data['label'], data_path, label_to_id)
            os.system('rm -rf {}/tmp.txt'.format(data_path))
        data_split_all, label_to_id = decode_all_files(['train.txt', 'valid.txt', 'test.txt'], label_to_id)
    else:
        raise ValueError('unknown dataset: %s' % data_name)

    if mecab:
        fix_labels(data_split_all, label_to_id, data_path)

    if allow_update:
        return data_split, label_to_id
    else:
        return data_split


def get_dataset_sentiment(
        data_name: str = 'sst', label_to_id: dict = None, allow_update: bool=True, cache_dir: str='./cache'):
    """ download dataset file and return dictionary including training/validation split """
    label_to_id = dict() if label_to_id is None else label_to_id

    def decode_data(iterator, _label_to_id: dict):
        list_text = []
        list_label = []
        for i in iterator:
            if data_name == 'sst' and i.label == 'neutral':
                continue
            list_text.append(' '.join(i.text))
            list_label.append(i.label)

        for unique_label in list(set(list_label)):
            if unique_label not in _label_to_id.keys():
                assert allow_update
                _label_to_id[unique_label] = len(_label_to_id)
        list_label = [int(_label_to_id[l]) for l in list_label]
        assert len(list_label) == len(list_text)
        return _label_to_id, {"data": list_text, "label": list_label}

    data_field, label_field = torchtext.data.Field(sequential=True), torchtext.data.Field(sequential=False)
    if data_name == 'imdb':
        iterator_split = torchtext.datasets.IMDB.splits(data_field, root=cache_dir, label_field=label_field)
    elif data_name == 'sst':
        iterator_split = torchtext.datasets.SST.splits(data_field, root=cache_dir, label_field=label_field)
    else:
        raise ValueError('unknown dataset: %s' % data_name)

    data_split, data = dict(), None
    for name, it in zip(['train', 'valid', 'test'], iterator_split):
        label_to_id, data = decode_data(it, _label_to_id=label_to_id)
        data_split[name] = data
        LOGGER.info('dataset %s/%s: %i' % (data_name, name, len(data['data'])))
    if allow_update:
        return data_split, label_to_id
    else:
        return data_split
