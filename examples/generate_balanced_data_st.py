from random import shuffle, seed
from tner import get_dataset

custom_data = {
    'train_2020': 'cache/twitter_ner/2020.bio.train.txt',
    'roberta_large': 'tner_output/self_labeled_data/twitter_ner/2021.bio.train.roberta_large.txt',
    'roberta_base': 'tner_output/self_labeled_data/twitter_ner/2021.bio.train.roberta_base.txt'
}
data, label_to_id, _, _ = get_dataset(custom_data=custom_data)
id_to_label = {v: k for k, v in label_to_id.items()}

tmp_21_rl = list(zip(data['roberta_large']['data'], data['roberta_large']['label']))
tmp_21_rb = list(zip(data['roberta_base']['data'], data['roberta_base']['label']))

tmp = list(zip(data['train_2020']['data'], data['train_2020']['label']))
seed(0)
shuffle(tmp)

with open('tner_output/self_labeled_data/twitter_ner/all_balanced.bio.train.roberta_base.txt', 'w') as f:
    for _x, _y in tmp[:len(tmp_21_rb)] + tmp_21_rb:
        for __x, __y in zip(_x, _y):
            __y = id_to_label[__y].replace('work of art', 'creative-work')
            f.write('{} {}\n'.format(__x, __y))
        f.write('\n')


with open('tner_output/self_labeled_data/twitter_ner/all_balanced.bio.train.roberta_large.txt', 'w') as f:
    for _x, _y in tmp[:len(tmp_21_rl)] + tmp_21_rl:
        for __x, __y in zip(_x, _y):
            __y = id_to_label[__y].replace('work of art', 'creative-work')
            f.write('{} {}\n'.format(__x, __y))
        f.write('\n')


