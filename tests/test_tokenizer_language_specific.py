""" UnitTest """
import unittest
import logging
import tner

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class Test(unittest.TestCase):
    """ Test MeCabWrapper """

    def test_ja(self):
        _tokenizer = tner.TokenizerJA()
        logging.info(_tokenizer.tokenize("日本ではサラリーマンが金曜日を華金と呼ぶ"))
        logging.info(_tokenizer.tokenize("日本ではサラリーマンが金曜日を華金と呼ぶ", return_pos=True))


if __name__ == "__main__":
    unittest.main()
