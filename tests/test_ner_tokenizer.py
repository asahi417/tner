""" UnitTest """
import unittest
import logging
from tner.ner_tokenizer import NERTokenizer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
id2label = {0: "O", 1: "B-LOC", 2: "I-LOC"}
tokens = [
    ["I", "live", "in", "London"],
    ["[AAAAAAA (B-LOCATION)]", "[GAGAGA (I-LOCATION)]", "[KLKLKL]", "[PPPPP (B-LOCATION)]"]
]
labels = [
    [0, 0, 0, 1],
    [1, 2, 0, 1]
]
models = [
    "roberta-base", "bert-base-cased", "bert-base-uncased", "xlm-roberta-base", "bert-base-multilingual-cased",
    "microsoft/deberta-base", "albert-base-v2", "gpt2", "xlnet-base-cased"
]


class Test(unittest.TestCase):
    """ Test get_benchmark_dataset """

    def test(self):
        for m in models:
            tokenizer = NERTokenizer(m, id2label=id2label, is_xlnet="xlnet" in m)
            for mask_by_padding_token in [True, False]:
                logging.info(f"testing {m}: (mask_by_padding_token: {mask_by_padding_token})")
                encode = tokenizer.encode_plus_all(
                    tokens=tokens,
                    labels=labels,
                    mask_by_padding_token=mask_by_padding_token
                )
                for n, e in enumerate(encode):
                    logging.info(f"sentence {n}: {tokens[n]}")
                    tokenized_input = tokenizer.tokenizer.convert_ids_to_tokens(e['input_ids'])
                    tokenized_label = e['labels']
                    for i, l in zip(tokenized_input, tokenized_label):
                        if l != -100:
                            logging.info(f"\t - {i}: {l}")


if __name__ == "__main__":
    unittest.main()
