""" UnitTest for dataset """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner
# from pprint import pprint

# transformers_model = 'asahi417/tner-xlm-roberta-large-ontonotes5'
transformers_model = './tner_output/ckpt_twitter_ner/baseline/roberta_large/model_tpkhxk/epoch_11'


class Test(unittest.TestCase):
    """ Test TrainTransformersNER """

    def test_1(self):
        model = tner.TransformersNER(transformers_model)
        test_sentences = [
            """I absolutely love the show "Sister Sister" it's a Netflix series ❤️'""",
            'Check out what I just added to my closet on Poshmark: Size 10 1/2 Leo girls tap shoes.. {{URL}} via {@Poshmark@} #shopmycloset'
            # 'I have an Apple computer.',
            # 'I like to eat an apple.'
        ]
        test_sentences = [i.split(' ') for i in test_sentences]
        test_result = model.predict(test_sentences, decode_bio=True)
        print(test_result)
        # pprint(list(zip(test_sentences, test_result)))


if __name__ == "__main__":
    unittest.main()
