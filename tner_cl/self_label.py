""" command line tool to test finetuned NER model """
import logging
import argparse

from tner import TransformersNER, get_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_options():
    parser = argparse.ArgumentParser(description='Self labeling on dataset with finetuned NER model',)
    parser.add_argument('-m', '--model', help='model', required=True, type=str)
    parser.add_argument('--max-length', default=128, type=int, help='max sequence length for input sequence')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('-d', '--dataset', help='dataset to evaluate', default=None, type=str)
    parser.add_argument('--dataset-split', help='dataset to evaluate', default='train', type=str)
    parser.add_argument('--custom-dataset', help='custom data set', default=None, type=str)
    parser.add_argument('-e', '--export-file', help='path to export the metric', required=True, type=str)
    return parser.parse_args()


def format_data(opt):
    assert opt.dataset is not None or opt.custom_dataset is not None
    if opt.dataset is not None:
        split = opt.dataset_split
        data = opt.dataset.split(',')
        custom_data = None
    else:
        data = None
        split = 'tmp'
        custom_data = {'tmp': opt.custom_dataset}
    dataset_split, _, _, _ = get_dataset(data=data, custom_data=custom_data)
    dataset = dataset_split[split]
    data = dataset['data']
    return data


def main():
    opt = get_options()
    data = format_data(opt)
    classifier = TransformersNER(opt.model, max_length=opt.max_length)
    sp_tokens = classifier.tokenizer.tokenizer.special_tokens_map.values()
    pred_list, inputs_subtoken = classifier.predict(
        data,
        is_tokenized=True,
        decode_bio=False,
        batch_size=opt.batch_size)
    total_input, total_label = [], []
    for p, i in zip(pred_list, inputs_subtoken):
        p = [_p for _p, _i in zip(p, i) if _i not in sp_tokens]
        i = [_i for _i in i if _i not in sp_tokens]
        assert len(p) == len(i)
        sequence_label = []
        pointer = 0
        token_n = 1
        tmp_cache = ''
        for n in range(1, len(i)):
            tmp = classifier.tokenizer.tokenizer.convert_tokens_to_string(i[:n])
            if tmp == '':
                continue
            if tmp.replace(tmp_cache, '').replace(' ', '') == '':
                continue
            tmp_cache = tmp
            if len(tmp.split(' ')) != token_n:
                sequence_label.append(p[pointer])
                token_n += 1
                pointer = n
        sequence_label.append(p[pointer])
        sequence_input = tmp_cache.split(' ')
        assert len(sequence_input) == len(sequence_label), '{} != {}'.format(len(sequence_input), len(sequence_label))
        total_input.append(sequence_input)
        total_label.append(sequence_label)

    with open(opt.export_file, 'w') as f:
        for _i, _l in zip(total_input, total_label):
            for __i, __l in zip(_i, _l):
                f.write('{} {}'.format(__i, __l) + '\n')
            f.write('\n')


if __name__ == '__main__':
    main()
