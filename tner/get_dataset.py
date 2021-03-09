""" Fetch/format NER dataset """
import os
import zipfile
import logging
import re
import requests
import tarfile
import shutil
from typing import Dict, List
from itertools import chain
from tqdm import tqdm
from .japanese_tokenizer import SudachiWrapper

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
STOPWORDS = ['None', '#']
PANX = ["ace", "bg", "da", "fur", "ilo", "lij", "mzn", "qu", "su", "vi", "af", "bh", "de", "fy", "io", "lmo", "nap",
        "rm", "sv", "vls", "als", "bn", "diq", "ga", "is", "ln", "nds", "ro", "sw", "vo", "am", "bo", "dv", "gan", "it",
        "lt", "ne", "ru", "szl", "wa", "an", "br", "el", "gd", "ja", "lv", "nl", "rw", "ta", "war", "ang", "bs", "eml",
        "gl", "jbo", "map-bms", "nn", "sa", "te", "wuu", "ar", "ca", "en", "gn", "jv", "mg", "no", "sah", "tg", "xmf",
        "arc", "cbk-zam", "eo", "gu", "ka", "mhr", "nov", "scn", "th", "yi", "arz", "cdo", "es", "hak", "kk", "mi",
        "oc", "sco", "tk", "yo", "as", "ce", "et", "he", "km", "min", "or", "sd", "tl", "zea", "ast", "ceb", "eu", "hi",
        "kn", "mk", "os", "sh", "tr", "zh-classical", "ay", "ckb", "ext", "hr", "ko", "ml", "pa", "si", "tt",
        "zh-min-nan", "az", "co", "fa", "hsb", "ksh", "mn", "pdc", "simple", "ug", "zh-yue", "ba", "crh", "fi", "hu",
        "ku", "mr", "pl", "sk", "uk", "zh", "bar", "cs", "fiu-vro", "hy", "ky", "ms", "pms", "sl", "ur", "bat-smg",
        "csb", "fo", "ia", "la", "mt", "pnb", "so", "uz", "be-x-old", "cv", "fr", "id", "lb", "mwl", "ps", "sq", "vec",
        "be", "cy", "frr", "ig", "li", "my", "pt", "sr", "vep"]
VALID_DATASET = ['conll2003', 'wnut2017', 'ontonotes5', 'mit_movie_trivia', 'mit_restaurant', 'fin', 'bionlp2004',
                 'bc5cdr'] + ['panx_dataset_{}'.format(i) for i in PANX]  # 'wiki_ja', 'wiki_news_ja'
CACHE_DIR = '{}/.cache/tner'.format(os.path.expanduser('~'))

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
    "event": ["EVENT"],
    "dna": ["DNA"],  # bionlp2004
    "protein": ["protein"],
    "cell type": ["cell_type"],
    "cell line": ["cell_line"],
    "rna": ["RNA"],
    "chemical": ["Chemical"],  # bc5cdr
    "disease": ["Disease"]
}

__all__ = ("get_dataset_ner", "VALID_DATASET", "SHARED_NER_LABEL")


def open_compressed_file(url, cache_dir):
    path = wget(url, cache_dir)
    if path.endswith('.tar.gz') or path.endswith('.tgz'):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(cache_dir)
        tar.close()
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)


def wget(url, cache_dir):
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)
    return '{}/{}'.format(cache_dir, filename)


def get_dataset_ner(data_names: (List, str) = None,
                    custom_data_path: str = None,
                    custom_data_language: str = 'en',
                    label_to_id: dict = None,
                    fix_label_dict: bool = False,
                    lower_case: bool = False):
    """ Fetch NER dataset

     Parameter
    -----------------
    data_names: list
        A list of dataset name
        eg) 'panx_dataset/*', 'conll2003', 'wnut2017', 'ontonote5', 'mit_movie_trivia', 'mit_restaurant'
    custom_data_path: str
        Filepath to custom dataset
    custom_data_language: str
        Language for custom_data_path dataset
    label_to_id: dict
        A dictionary of label to id
    fix_label_dict: bool
        Fixing given label_to_id dictionary (ignore label not in the dictionary in dataset)
    lower_case: bool
        Converting data into lowercased

     Return
    ----------------
    unified_data: dict
        A dataset consisting of 'train'/'valid' (only 'train' if more than one data set is used)
    label_to_id: dict
        A dictionary of label to id
    language: str
        Most frequent language in the dataset
    """
    assert data_names or custom_data_path, 'either `data_names` or `custom_data_path` should be not None'
    data_names = data_names if data_names else []
    data_names = [data_names] if type(data_names) is str else data_names
    custom_data_path = [custom_data_path] if custom_data_path else []
    data_list = data_names + custom_data_path
    logging.info('target dataset: {}'.format(data_list))
    data = []
    languages = []
    unseen_entity_set = {}
    param = dict(fix_label_dict=fix_label_dict, lower_case=lower_case, custom_language=custom_data_language)
    for d in data_list:
        param['label_to_id'] = label_to_id
        data_split_all, label_to_id, language, ues = get_dataset_ner_single(d, **param)
        data.append(data_split_all)
        languages.append(language)
        unseen_entity_set = ues if len(unseen_entity_set) == 0 else unseen_entity_set.intersection(ues)
    if len(data) > 1:
        unified_data = {
            'train': {
                'data': list(chain(*[d['train']['data'] for d in data])),
                'label': list(chain(*[d['train']['label'] for d in data]))
            }
        }
    else:
        unified_data = data[0]
    # use the most frequent language in the data
    freq = list(map(lambda x: (x, len(list(filter(lambda y: y == x, languages)))), set(languages)))
    language = sorted(freq, key=lambda x: x[1], reverse=True)[0][0]
    return unified_data, label_to_id, language, unseen_entity_set


def get_dataset_ner_single(data_name: str = 'wnut2017',
                           label_to_id: dict = None,
                           fix_label_dict: bool = False,
                           lower_case: bool = False,
                           custom_language: str = 'en',
                           allow_new_entity: bool = True,
                           cache_dir: str = None):
    """ download dataset file and return dictionary including training/validation split

    :param data_name: data set name or path to the data
    :param label_to_id: fixed dictionary of (label: id). If given, ignore other labels
    :param fix_label_dict: not augment label_to_id based on dataset if True
    :param lower_case: convert to lower case
    :param custom_language
    :param allow_new_entity
    :return: formatted data, label_to_id
    """
    post_process_ja = False
    entity_first = False
    to_bio = False
    language = 'en'
    cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    data_path = os.path.join(cache_dir, data_name)
    logging.info('data_name: {}'.format(data_name))

    if data_name in ['ontonotes5', 'conll2003']:

        files_info = {'train': 'train.txt', 'valid': 'dev.txt', 'test': 'test.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            # raise ValueError('please download Ontonotes5 from https://catalog.ldc.upenn.edu/LDC2013T19')
            url = 'https://github.com/asahi417/neighbor-tagging/raw/master/data.tar.gz'
            open_compressed_file(url, data_path)
            for i in ['train', 'dev', 'test']:
                if data_name == 'ontonotes5':
                    conll_formatting(
                        file_token='{}/data/onto/{}.words'.format(data_path, i),
                        file_tag='{}/data/onto/{}.ner'.format(data_path, i),
                        output_file='{}/{}.txt'.format(data_path, i))
                else:
                    conll_formatting(
                        file_token='{}/data/conll2003/conll2003-{}.words'.format(data_path, i),
                        file_tag='{}/data/conll2003/conll2003-{}.nertags'.format(data_path, i),
                        output_file='{}/{}.txt'.format(data_path, i))
    elif data_name == 'bc5cdr':
        files_info = {'train': 'train.txt', 'valid': 'dev.txt', 'test': 'test.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            url = 'https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_Data.zip'
            open_compressed_file(url, data_path)
            shutil.move('{}/CDR_Data/CDR.Corpus.v010516'.format(data_path), data_path)

        def __process_single(_r):
            title, body = _r.split('\n')[:2]
            entities = _r.split('\n')[2:]
            text = title.split('|t|')[-1] + ' ' + body.split('|a|')[-1]
            _tokens = []
            _tags = []
            last_end = 0
            for e in entities:
                start, end = e.split('\t')[1:3]
                try:
                    start, end = int(start), int(end)
                except ValueError:
                    continue
                mention = e.split('\t')[3]
                entity_type = e.split('\t')[4]
                assert text[start:end] == mention
                _tokens_tmp = list(filter(
                    lambda _x: len(_x) > 0, map(lambda m: m.replace(' ', ''), re.split(r'\b', text[last_end:start]))
                ))
                last_end = end
                _tokens += _tokens_tmp
                _tags += ['O'] * len(_tokens_tmp)
                _mention_token = mention.split(' ')
                _tokens += _mention_token
                _tags += ['B-{}'.format(entity_type)] + ['I-{}'.format(entity_type)] * (len(_mention_token) - 1)
                assert len(_tokens) == len(_tags)
            return _tokens, _tags

        def convert_to_iob(path, export):
            path = '{0}/{1}'.format(data_path, path)
            with open(path, 'r') as f:
                raw = list(filter(lambda _x: len(_x) > 0, f.read().split('\n\n')))
                token_tag = list(map(lambda _x: __process_single(_x), raw))
                tokens, tags = list(zip(*token_tag))
                conll_formatting(tokens=tokens, tags=tags, output_file=os.path.join(data_path, export), sentence_division='.')

        convert_to_iob('CDR.Corpus.v010516/CDR_DevelopmentSet.PubTator.txt', 'dev.txt')
        convert_to_iob('CDR.Corpus.v010516/CDR_TestSet.PubTator.txt', 'test.txt')
        convert_to_iob('CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt', 'train.txt')
    elif data_name == 'bionlp2004':  # https://www.aclweb.org/anthology/W04-1213.pdf
        files_info = {'train': 'Genia4ERtask1.iob2', 'valid': 'Genia4EReval1.iob2'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            url = 'http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz'
            open_compressed_file(url, data_path)
            url = 'http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz'
            open_compressed_file(url, data_path)
    elif data_name == 'fin':  # https://www.aclweb.org/anthology/U15-1010.pdf
        files_info = {'train': 'FIN5.txt', 'valid': 'FIN3.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            url = 'https://people.eng.unimelb.edu.au/tbaldwin/resources/finance-sec/financial_risk_assessment.tgz'
            open_compressed_file(url, data_path)
            for v in files_info.values():
                shutil.move('{}/dataset/{}'.format(data_path, v), data_path)
        to_bio = True
    elif data_name == 'mit_restaurant':
        files_info = {'train': 'restauranttrain.bio', 'valid': 'restauranttest.bio'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            url = 'https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttrain.bio'
            open_compressed_file(url, data_path)
            url = 'https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttest.bio'
            open_compressed_file(url, data_path)
        entity_first = True
    elif data_name == 'mit_movie_trivia':
        files_info = {'train': 'trivia10k13train.bio', 'valid': 'trivia10k13test.bio'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            url = 'https://groups.csail.mit.edu/sls/downloads/movie/trivia10k13train.bio'
            open_compressed_file(url, data_path)
            url = 'https://groups.csail.mit.edu/sls/downloads/movie/trivia10k13test.bio'
            open_compressed_file(url, data_path)
        entity_first = True
    elif data_name == 'wnut2017':
        files_info = {'train': 'train.txt', 'valid': 'valid.txt', 'test': 'test.txt'}
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            url = 'https://github.com/leondz/emerging_entities_17/raw/master/wnut17train.conll'
            open_compressed_file(url, data_path)
            with open('{}/wnut17train.conll'.format(data_path), 'r') as f:
                with open('{}/train.txt'.format(data_path), 'w') as f_w:
                    f_w.write(f.read().replace('\t', ' '))

            url = 'https://github.com/leondz/emerging_entities_17/raw/master/emerging.dev.conll'
            open_compressed_file(url, data_path)
            with open('{}/emerging.dev.conll'.format(data_path), 'r') as f:
                with open('{}/valid.txt'.format(data_path), 'w') as f_w:
                    f_w.write(f.read().replace('\t', ' '))

            url = 'https://raw.githubusercontent.com/leondz/emerging_entities_17/master/emerging.test.annotated'
            open_compressed_file(url, data_path)
            with open('{}/emerging.test.annotated'.format(data_path), 'r') as f:
                with open('{}/test.txt'.format(data_path), 'w') as f_w:
                    f_w.write(f.read().replace('\t', ' '))
    elif 'panx_dataset' in data_name:
        files_info = {'valid': 'dev.txt', 'train': 'train.txt', 'test': 'test.txt'}
        panx_la = data_name.replace('/', '_').split('_')[-1]
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            url = 'https://github.com/asahi417/neighbor-tagging/releases/download/0.0.0/wikiann.zip'
            open_compressed_file(url, cache_dir)
            tar = tarfile.open('{0}/panx_dataset/{1}.tar.gz'.format(cache_dir, panx_la), "r:gz")
            tar.extractall(data_path)
            tar.close()
            for v in files_info.values():
                os.system("sed -e 's/{0}://g' {1}/{2} > {1}/{2}.txt".format(panx_la, data_path, v.replace('.txt', '')))
                os.system("rm -rf {0}/{1}".format(data_path, v.replace('.txt', '')))
        if panx_la == 'ja':
            language = 'ja'
            post_process_ja = True
    else:
        # for custom data
        data_path = data_name
        language = custom_language
        if not os.path.exists(data_path):
            raise ValueError('unknown dataset: %s' % data_path)
        else:
            files_info = {'train': 'train.txt', 'valid': 'valid.txt'}

    label_to_id = dict() if label_to_id is None else label_to_id
    data_split_all, unseen_entity_set, label_to_id = decode_all_files(
        files_info, data_path, label_to_id=label_to_id, fix_label_dict=fix_label_dict, entity_first=entity_first,
        to_bio=to_bio, allow_new_entity=allow_new_entity)

    if post_process_ja:
        logging.info('Japanese tokenization post processing')
        id_to_label = {v: k for k, v in label_to_id.items()}
        label_fixer = SudachiWrapper()
        data, label = [], []
        for k, v in data_split_all.items():
            for x, y in tqdm(zip(v['data'], v['label'])):
                y = [id_to_label[_y] for _y in y]
                _data, _label = label_fixer.fix_ja_labels(inputs=x, labels=y)
                _label = [label_to_id[_y] for _y in _label]
                data.append(_data)
                label.append(_label)
            v['data'] = data
            v['label'] = label

    if lower_case:
        logging.info('convert into lower cased')
        data_split_all = {
            k: {'data': [[ii.lower() for ii in i] for i in v['data']], 'label': v['label']}
            for k, v in data_split_all.items()}
    return data_split_all, label_to_id, language, unseen_entity_set


def decode_file(file_name: str,
                data_path: str,
                label_to_id: Dict,
                fix_label_dict: bool,
                entity_first: bool = False,
                to_bio: bool = False,
                allow_new_entity: bool = False):
    inputs, labels, seen_entity = [], [], []
    past_mention = 'O'
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
                    if to_bio and mention == past_mention:
                        location = 'I'
                    elif to_bio:
                        location = 'B'

                    fixed_mention = [k for k, v in SHARED_NER_LABEL.items() if mention in v]
                    if len(fixed_mention) == 0 and allow_new_entity:
                        tag = '-'.join([location, mention])
                    elif len(fixed_mention) == 0:
                        tag = 'O'
                    else:
                        tag = '-'.join([location, fixed_mention[0]])
                    past_mention = mention
                else:
                    past_mention = 'O'

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


def decode_all_files(files: Dict, data_path: str, label_to_id: Dict, fix_label_dict: bool, entity_first: bool = False,
                     to_bio: bool = False, allow_new_entity: bool = False):
    data_split = dict()
    unseen_entity = None
    for name, filepath in files.items():
        label_to_id, unseen_entity_set, data_dict = decode_file(
            filepath, data_path=data_path, label_to_id=label_to_id, fix_label_dict=fix_label_dict,
            entity_first=entity_first, to_bio=to_bio, allow_new_entity=allow_new_entity)
        if unseen_entity is None:
            unseen_entity = unseen_entity_set
        else:
            unseen_entity = unseen_entity.intersection(unseen_entity_set)
        data_split[name] = data_dict
        logging.info('dataset {0}/{1}: {2} entries'.format(data_path, filepath, len(data_dict['data'])))
    return data_split, unseen_entity, label_to_id


def conll_formatting(output_file: str,
                     file_token: str=None,
                     file_tag: str=None,
                     tokens=None,
                     tags=None,
                     sentence_division=None):
    """ convert a separate ner/token file into single ner Conll 2003 format """
    if file_token:
        with open(file_token, 'r') as f:
            tokens = [i.split(' ') for i in f.read().split('\n')]
    if file_tag:
        with open(file_tag, 'r') as f:
            tags = [i.split(' ') for i in f.read().split('\n')]
    assert tokens and tags
    _end = False
    with open(output_file, 'w') as f:
        assert len(tokens) == len(tags)
        for _token, _tag in zip(tokens, tags):
            assert len(_token) == len(_tag)
            for __token, __tag in zip(_token, _tag):
                _end = False
                f.write('{0} {1}\n'.format(__token, __tag))
                if sentence_division and __token == sentence_division:
                    f.write('\n')
                    _end = True
            if _end:
                _end = False
            else:
                f.write('\n')
                _end = True

