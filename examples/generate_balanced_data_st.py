from random import shuffle, seed
from tner import get_dataset

custom_data = {
    '2020.train': 'cache/twitter_ner/2020.train.txt',
    '2020.extra.roberta_large': 'cache/twitter_ner/2020.extra.roberta_large.txt',
    '2021.extra.roberta_large': 'cache/twitter_ner/2021.extra.roberta_large.txt',
}
data, label_to_id, _, _ = get_dataset(custom_data=custom_data)
id_to_label = {v: k for k, v in label_to_id.items()}


tmp_2020_train = list(zip(data['2020.train']['data'], data['2020.train']['label']))
data_trunc = {}
for _data in ['2020.extra.roberta_large', '2021.extra.roberta_large']:
    tmp_data = list(zip(data[_data]['data'], data[_data]['label']))
    seed(0)
    shuffle(tmp_data)
    data_trunc[_data] = tmp_data[:len(tmp_2020_train)]

    with open('./balance.{}.txt'.format(_data), 'w') as f:
        for _x, _y in tmp_2020_train + data_trunc[_data]:
            for __x, __y in zip(_x, _y):
                __y = id_to_label[__y].replace('work of art', 'creative-work')
                f.write('{} {}\n'.format(__x, __y))
            f.write('\n')

with open('./balance.concat.txt', 'w') as f:
    for _x, _y in tmp_2020_train + data_trunc['2020.extra.roberta_large'] + data_trunc['2021.extra.roberta_large']:
        for __x, __y in zip(_x, _y):
            __y = id_to_label[__y].replace('work of art', 'creative-work')
            f.write('{} {}\n'.format(__x, __y))
        f.write('\n')

