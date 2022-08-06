from typing import List
from itertools import accumulate

from sudachipy import tokenizer
from sudachipy import dictionary

POS_MAPPER = {
    "名詞": "NOUN",
    "形容詞": "ADJ",
    "動詞": "VERB",
    "RANDOM": "RANDOM"
}


class TokenizerJA:
    """ Sudachi tokenizer

     Usage
    --------------
    >>> tokenizer = SudachiWrapper()
    >>> tokenizer.tokenize("日本ではサラリーマンが金曜日を華金と呼ぶ")
    ['日本', 'で', 'は', 'サラリーマン', 'が', '金曜日', 'を', '華金', 'と', '呼ぶ']
    >>> tokenizer.tokenize("日本ではサラリーマンが金曜日を華金と呼ぶ", return_pos=True)
    [['日本', 'NOUN'], ['で', 'RANDOM'], ['は', 'RANDOM'], ['サラリーマン', 'NOUN'], ['が', 'RANDOM'], ['金曜日', 'NOUN'],
    ['を', 'RANDOM'], ['華金', 'NOUN'], ['と', 'RANDOM'], ['呼ぶ', 'VERB']]
    """

    def __init__(self):
        self.tokenizer = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C

    def tokenize(self, sentence: str, return_pos: bool = False):

        def formatting(_raw, _pos):
            if not return_pos:
                return _raw
            try:
                _pos = POS_MAPPER[_pos]
            except KeyError:
                _pos = POS_MAPPER['RANDOM']
            return [_raw, _pos]

        parsed = self.tokenizer.tokenize(sentence, self.mode)
        # if parsed is None:
        #     return None
        tokens = list(map(lambda x: formatting(x.surface(), x.part_of_speech()[0]), parsed))
        return list(filter(lambda x: len(x) > 0, tokens))

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

