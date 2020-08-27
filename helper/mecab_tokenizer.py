""" MeCab tokenizer with Neologd """
import os
import MeCab

POS_MAPPER = {
    "名詞": "NOUN",
    "形容詞": "ADJ",
    "動詞": "VERB",
    "RANDOM": "RANDOM"
}

__all__ = ["JaTokenizer"]


class JaTokenizer:
    """ MeCab tokenizer with Neologd
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


if __name__ == '__main__':
    _tokenizer = JaTokenizer(False)
    print(_tokenizer("日本ではサラリーマンが金曜日を華金と呼ぶ"))
    print(_tokenizer("日本ではサラリーマンが金曜日を華金と呼ぶ", return_pos=True))
