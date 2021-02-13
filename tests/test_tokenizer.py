""" UnitTest """
import unittest
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import tner


class Test(unittest.TestCase):
    """ Test get_benchmark_dataset """

    def test(self):
        transforms = tner.Transforms('roberta-base')
        unified_data, label_to_id, language, unseen_entity_set = tner.get_dataset_ner('ontonotes5')
        tokens = unified_data['valid']['data'][:10]
        labels = unified_data['valid']['label'][:10]
        features = transforms.encode_plus_all(
            tokens=tokens,
            labels=labels,
            language=language,
            max_length=32)
        label_to_id = {v: k for k, v in label_to_id.items()}

        for n, i in enumerate(features):
            tokens_restore = transforms.tokenizer.convert_ids_to_tokens(i['input_ids'])
            labels_restore = [label_to_id[_l] if _l != -100 else 'PAD' for _l in i['labels']]
            print(list(zip(tokens_restore, labels_restore)))
            print(list(zip(tokens[n], [label_to_id[_l] for _l in labels[n] if _l != -100])))
            print()


if __name__ == "__main__":
    unittest.main()
