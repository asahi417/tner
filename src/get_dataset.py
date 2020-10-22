""" Fetch/format NER dataset """
import os
import zipfile
from typing import Dict
from itertools import chain

from .mecab_wrapper import MeCabWrapper
from .util import get_logger

LOGGER = get_logger()
STOPWORDS = ['None', '#']
CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)
# Shared label set across different dataset
SHARED_NER_LABEL = {
    "location": ["LOCATION", "LOC", "location", "Location"],
    "organization": ["ORGANIZATION", "ORG", "organization"],
    "person": ["PERSON", "PSN", "person", "PER"],
    "date": ["DATE", "DAT", 'YEAR', 'Year'],
    "time": ["TIME", "TIM", "Hours"],
    "artifact": ["ARTIFACT", "ART", "artifact"],
    "percent": ["PERCENT", "PNT"],
    "other": ["OTHER", "MISC"],
    "money": ["MONEY", "MNY", "Price"],
    "corporation": ["corporation"],  # Wnut 17
    "group": ["group", "NORP"],
    "product": ["product", "PRODUCT"],
    "rating": ["Rating", 'RATING'],  # restaurant review
    "amenity": ["Amenity"],
    "restaurant": ["Restaurant_Name"],
    "dish": ["Dish"],
    "cuisine": ["Cuisine"],
    "actor": ['ACTOR', 'Actor'],  # movie review
    "title": ['TITLE'],
    "genre": ['GENRE', 'Genre'],
    "director": ['DIRECTOR', 'Director'],
    "song": ['SONG'],
    "plot": ['PLOT', 'Plot'],
    "review": ['REVIEW'],
    'character': ['CHARACTER'],
    "ratings average": ['RATINGS_AVERAGE'],
    'trailer': ['TRAILER'],
    'opinion': ['Opinion'],
    'award': ['Award'],
    'origin': ['Origin'],
    'soundtrack': ['Soundtrack'],
    'relationship': ['Relationship'],
    'character name': ['Character_Name'],
    'quote': ['Quote'],
    "cardinal number": ["CARDINAL"],  # OntoNote 5
    "ordinal number": ["ORDINAL"],
    "quantity": ['QUANTITY'],
    "law": ['LAW'],
    "geopolitical area": ['GPE'],
    "work of art": ["WORK_OF_ART", "creative-work"],
    "facility": ["FAC"],
    "language": ["LANGUAGE"],
    "event": ["EVENT"]
}

__all__ = "get_dataset_ner"


def decode_file(file_name: str, data_path: str, label_to_id: Dict, fix_label_dict: bool, entity_first: bool = False):
    inputs, labels, seen_entity = [], [], []
    # seen_entity = list(label_to_id.keys())
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
                if entity_first:
                    tag, word = ls[0], ls[-1]
                else:
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

    id_to_label = {v: k for k, v in label_to_id.items()}
    unseen_entity_id = set(label_to_id.values()) - set(list(chain(*labels)))
    unseen_entity_label = {id_to_label[i] for i in unseen_entity_id}
    return label_to_id, unseen_entity_label, {"data": inputs, "label": labels}


def decode_all_files(files: Dict, data_path: str, label_to_id: Dict, fix_label_dict: bool, entity_first: bool = False):
    data_split = dict()
    unseen_entity = None
    for name, filepath in files.items():
        label_to_id, unseen_entity_set, data_dict = decode_file(
            filepath, data_path=data_path, label_to_id=label_to_id, fix_label_dict=fix_label_dict,
            entity_first=entity_first)
        if unseen_entity is None:
            unseen_entity = unseen_entity_set
        else:
            unseen_entity = unseen_entity.intersection(unseen_entity_set)
        data_split[name] = data_dict
        LOGGER.info('dataset {0}/{1}: {2} entries'.format(data_path, filepath, len(data_dict['data'])))
    return data_split, unseen_entity, label_to_id


def conll_formatting(file_token: str, file_tag: str, output_file: str):
    """ convert a separate ner/token file into single ner Conll 2003 format """
    tokens = [i.split(' ') for i in open(file_token, 'r').read().split('\n')]
    tags = [i.split(' ') for i in open(file_tag, 'r').read().split('\n')]
    with open(output_file, 'w') as f:
        assert len(tokens) == len(tags)
        for _token, _tag in zip(tokens, tags):
            assert len(_token) == len(_tag)
            for __token, __tag in zip(_token, _tag):
                f.write('{0} {1}\n'.format(__token, __tag))
            f.write('\n')


def get_dataset_ner(data_name: str = 'wnut_17', label_to_id: dict = None, fix_label_dict: bool = False):
    """ download dataset file and return dictionary including training/validation split

    :param data_name: data set name or path to the data
    :param label_to_id: fixed dictionary of (label: id). If given, ignore other labels
    :param fix_label_dict: not augment label_to_id based on dataset if True
    :return: formatted data, label_to_id
    """
    data_path = os.path.join(CACHE_DIR, data_name)
    post_process_mecab = False
    entity_first = False
    language = 'en'
    LOGGER.info('data_name: {}'.format(data_name))
    if data_name == 'conll_2003':
        files_info = {'train': 'train.txt', 'valid': 'dev.txt', 'test': 'test.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('wget -O {0}/data.tar.gz https://github.com/swiseman/neighbor-tagging/raw/master/data.tar.gz'.
                      format(CACHE_DIR))
            os.system('tar -xzf {0}/data.tar.gz -C {0}'.format(CACHE_DIR))
            for i in ['train', 'dev', 'test']:
                conll_formatting(
                    file_token=os.path.join(CACHE_DIR, 'data/conll2003/conll2003-{}.words'.format(i)),
                    file_tag=os.path.join(CACHE_DIR, 'data/conll2003/conll2003-{}.nertags'.format(i)),
                    output_file=os.path.join(data_path, '{}.txt'.format(i)))
    elif data_name == 'mit_restaurant':
        files_info = {'train': 'train.txt', 'valid': 'valid.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('wget -O {0} https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttrain.bio'.format(
                os.path.join(data_path, 'train.txt')))
            os.system('wget -O {0} https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttest.bio'.format(
                os.path.join(data_path, 'valid.txt')))
        entity_first = True
    elif data_name == 'ontonote5':
        files_info = {'train': 'train.txt', 'valid': 'dev.txt', 'test': 'test.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('wget -O {0}/data.tar.gz https://github.com/swiseman/neighbor-tagging/raw/master/data.tar.gz'.
                format(CACHE_DIR))
            os.system('tar -xzf {0}/data.tar.gz -C {0}'.format(CACHE_DIR))
            for i in ['train', 'dev', 'test']:
                conll_formatting(
                    file_token=os.path.join(CACHE_DIR, 'data/onto/{}.words'.format(i)),
                    file_tag=os.path.join(CACHE_DIR, 'data/onto/{}.ner'.format(i)),
                    output_file=os.path.join(data_path, '{}.txt'.format(i)))
    elif data_name == 'mit_movie_trivia':
        files_info = {'train': 'train.txt', 'valid': 'valid.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('wget -O {0} https://groups.csail.mit.edu/sls/downloads/movie/trivia10k13train.bio'.format(
                os.path.join(data_path, 'train.txt')))
            os.system('wget -O {0} https://groups.csail.mit.edu/sls/downloads/movie/trivia10k13test.bio'.format(
                os.path.join(data_path, 'valid.txt')))
        entity_first = True
    elif data_name == 'mit_movie':
        files_info = {'train': 'train.txt', 'valid': 'valid.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system('wget -O {0} https://groups.csail.mit.edu/sls/downloads/movie/engtrain.bio'.format(
                os.path.join(data_path, 'train.txt')))
            os.system('wget -O {0} https://groups.csail.mit.edu/sls/downloads/movie/engtest.bio'.format(
                os.path.join(data_path, 'valid.txt')))
        entity_first = True
    elif data_name == 'wnut_17':
        files_info = {'train': 'train.txt.tmp', 'valid': 'dev.txt.tmp', 'test': 'test.txt.tmp'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/wnut17train.conll'  | tr '\t' ' ' > {}/train.txt.tmp".format(data_path))
            os.system("curl -L 'https://github.com/leondz/emerging_entities_17/raw/master/emerging.dev.conll' | tr '\t' ' ' > {}/dev.txt.tmp".format(data_path))
            os.system("curl -L 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated' | tr '\t' ' ' > {}/test.txt.tmp".format(data_path))
    elif data_name == 'wiki_ja':
        files_info = {'test': 'test.txt'}
        language = 'ja'
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system(
                'wget -O {0}/test.txt https://raw.githubusercontent.com/Hironsan/IOB2Corpus/master/hironsan.txt'.
                format(data_path))
    elif data_name == 'wiki_news_ja':
        files_info = {'test': 'test.txt'}
        language = 'ja'
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            os.system(
                'wget -O {0}/test.txt https://github.com/Hironsan/IOB2Corpus/raw/master/ja.wikipedia.conll'.
                format(data_path))
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
            post_process_mecab = True
    else:
        # for custom data
        if not os.path.exists(data_path):
            raise ValueError('unknown dataset: %s' % data_name)
        else:
            files_info = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}
        if 'ja' in data_name:
            language = 'ja'

    label_to_id = dict() if label_to_id is None else label_to_id
    data_split_all, unseen_entity_set, label_to_id = decode_all_files(
        files_info, data_path, label_to_id=label_to_id, fix_label_dict=fix_label_dict, entity_first=entity_first)

    if post_process_mecab:
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

    return data_split_all, label_to_id, language, unseen_entity_set
