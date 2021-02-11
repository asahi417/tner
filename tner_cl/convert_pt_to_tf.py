import argparse

from transformers import TFAutoModelForTokenClassification


def get_options():
    parser = argparse.ArgumentParser(description='command line tool to test finetuned NER model',)
    parser.add_argument('-c', '--checkpoint-dir', help='checkpoint to load', default=None, type=str)
    return parser.parse_args()


def main():
    opt = get_options()
    classifier = TFAutoModelForTokenClassification.from_pretrained(opt.checkpoint_dir, from_pt=True)
    classifier.save_pretrained(opt.checkpoint_dir)


if __name__ == '__main__':
    main()
