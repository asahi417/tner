""" UnitTest """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner


class Test(unittest.TestCase):
    """ Test MeCabWrapper """

    def test_mecab(self):
        _tokenizer = tner.MeCabWrapper()
        logging.info(_tokenizer.tokenize("日本ではサラリーマンが金曜日を華金と呼ぶ"))
        logging.info(_tokenizer.tokenize("日本ではサラリーマンが金曜日を華金と呼ぶ", return_pos=True))


if __name__ == "__main__":
    unittest.main()
