""" MeCab tokenizer with Neologd to fix Japanese NER labels """
import os
from typing import List
from itertools import accumulate
try:
    import MeCab
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install mecab-python3 `pip install mecab-python3==0.996.5` ")

POS_MAPPER = {
    "名詞": "NOUN",
    "形容詞": "ADJ",
    "動詞": "VERB",
    "RANDOM": "RANDOM"
}

__all__ = "MeCabWrapper"


class MeCabWrapper:
    """ MeCab tokenizer with Neologd

     Usage
    --------------
    >>> tokenizer = MeCabWrapper()
    >>> tokenizer.tokenize("日本ではサラリーマンが金曜日を華金と呼ぶ")
    ['日本', 'で', 'は', 'サラリーマン', 'が', '金曜日', 'を', '華金', 'と', '呼ぶ']
    >>> tokenizer.tokenize("日本ではサラリーマンが金曜日を華金と呼ぶ", return_pos=True)
    [['日本', 'NOUN'], ['で', 'RANDOM'], ['は', 'RANDOM'], ['サラリーマン', 'NOUN'], ['が', 'RANDOM'], ['金曜日', 'NOUN'],
    ['を', 'RANDOM'], ['華金', 'NOUN'], ['と', 'RANDOM'], ['呼ぶ', 'VERB']]
    >>> tokenizer = MeCabWrapper(False)
    >>> tokenizer.tokenize("日本ではサラリーマンが金曜日を華金と呼ぶ")
    ['日本', 'で', 'は', 'サラリーマン', 'が', '金曜日', 'を', '華', '金', 'と', '呼ぶ']
    """

    def __init__(self, neologd: bool = False):
        self.__tagger = MeCab.Tagger()
        if neologd:
            f = os.popen('echo `mecab-config --dicdir`"/mecab-ipadic-neologd"')
            path_to_neologd = f.read().replace('\n', '')
            if os.path.exists(path_to_neologd):
                self.__tagger = MeCab.Tagger("-d {}".format(path_to_neologd))
            else:
                self.__tagger = MeCab.Tagger("")

        self.__tagger.parse('テスト')

    def tokenize(self, sentence: str, return_pos: bool = False):
        def formatting(_raw, _pos):
            if not return_pos:
                return _raw
            try:
                _pos = POS_MAPPER[_pos]
            except KeyError:
                _pos = POS_MAPPER['RANDOM']
            return [_raw, _pos]

        parsed = self.__tagger.parse(sentence)
        if parsed is None:
            return None

        parsed_sentence = parsed.split("\n")
        out = [formatting(s.split("\t")[0], s.split("\t")[1].split(",")[0]) for s in parsed_sentence if "\t" in s]
        return out

    def fix_ja_labels(self, inputs: List, labels: List):
        """ fix japanese NER tag with MeCab tokenizer """
        tokens = self.tokenize(''.join(inputs))
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
