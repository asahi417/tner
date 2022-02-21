""" Cache embedding for contextualized prediction """
import argparse
import logging
import os
from tner.text_searcher import SentenceEmbedding
from tner.data import decode_file


def get_options():
    parser = argparse.ArgumentParser(description='Cache embedding for contextualized prediction')
    parser.add_argument('-f', '--file', help='csv file to index', required=True, type=str)
    parser.add_argument('-m', '--model', help='model', default='sentence-transformers/all-mpnet-base-v1', type=str)
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('-e', '--export-file', help='path to export the metric', required=True, type=str)
    return parser.parse_args()


def main():
    opt = get_options()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    embedding_model = SentenceEmbedding(opt.model)
    # run inference
    _, _, data = decode_file(opt.file)
    text = data["data"]
    embedding = embedding_model.embed(text, batch_size=opt.batch_size, show_progress_bar=True)
    os.makedirs(os.path.dirname(opt.export_file), exist_ok=True)
    with open(opt.export_file, 'w') as f:
        for e in embedding.tolist():
            f.write(','.join([str(i) for i in e]) + '\n')


if __name__ == '__main__':
    main()
