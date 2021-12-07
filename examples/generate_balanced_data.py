from random import shuffle
from tner import get_dataset

custom_data = {
    'train_2020': 'cache/twitter_ner/2020.bio.train.txt',
    'train_2021': 'cache/twitter_ner/2021.bio.train.txt'
}
data, label_to_id, _, _ = get_dataset(custom_data=custom_data)
id_to_label = {v: k for k, v in label_to_id.items()}
x_20 = data['train_2020']['data']
y_20 = data['train_2020']['label']
x_21 = data['train_2021']['data']
y_21 = data['train_2021']['label']

tmp = list(zip(x_20, y_20))
shuffle(tmp)
tmp = tmp[:len(x_21)]
tmp_21 = list(zip(x_21, y_21))

tmp = tmp + tmp_21

with open('cache/twitter_ner/all_balanced.bio.train.txt', 'w') as f:
    for _x, _y in tmp:
        _y = [id_to_label[__y] for __y in _y]
        for __x, __y in zip(_x, _y):
            __y = __y.replace('work of art', 'creative-work')
            f.write('{} {}\n'.format(__x, __y))
        f.write('\n')


