from random import shuffle, seed
from tner import get_dataset

custom_data = {
    'train_2020': 'cache/twitter_ner/2020.bio.train.txt',
    'roberta_large': 'tner_output/self_labeled_data/twitter_ner/2021.bio.train.roberta_large.txt',
    'roberta_base': 'tner_output/self_labeled_data/twitter_ner/2021.bio.train.roberta_base.txt'
}
data, label_to_id, _, _ = get_dataset(custom_data=custom_data)
id_to_label = {v: k for k, v in label_to_id.items()}

x_20 = data['train_2020']['data']
y_20 = data['train_2020']['label']

x_21_rl = data['roberta_large']['data']
y_21_rl = data['roberta_large']['label']

x_21_rb = data['roberta_base']['data']
y_21_rb = data['roberta_base']['label']

tmp_21_rl = list(zip(x_21_rl, y_21_rl))
tmp_21_rb = list(zip(x_21_rb, y_21_rb))

tmp = list(zip(x_20, y_20))
seed(0)
shuffle(tmp)

with open('tner_output/self_labeled_data/twitter_ner/all_balanced.bio.train.roberta_base.txt', 'w') as f:
    for _x, _y in tmp + tmp_21_rb:
        _y = [id_to_label[__y] for __y in _y]
        for __x, __y in zip(_x, _y):
            __y = __y.replace('work of art', 'creative-work')
            f.write('{} {}\n'.format(__x, __y))
        f.write('\n')


with open('tner_output/self_labeled_data/twitter_ner/all_balanced.bio.train.roberta_large.txt', 'w') as f:
    for _x, _y in tmp + tmp_21_rl:
        _y = [id_to_label[__y] for __y in _y]
        for __x, __y in zip(_x, _y):
            __y = __y.replace('work of art', 'creative-work')
            f.write('{} {}\n'.format(__x, __y))
        f.write('\n')


