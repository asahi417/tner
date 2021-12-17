import pandas as pd
from tner import get_dataset

data_list = {
    '2020/dev': 'cache/twitter_ner/2020.bio.dev.txt',
    '2020/test': 'cache/twitter_ner/2020.bio.test.txt',
    '2020/train': 'cache/twitter_ner/2020.bio.train.txt',
    '2020/unlabelled': 'cache/twitter_ner/2020.bio.unlabelled.txt',
    '2021/test': 'cache/twitter_ner/2021.bio.test.txt',
    '2021/train': 'cache/twitter_ner/2021.bio.train.txt',
    '2021/unlabelled': 'cache/twitter_ner/2021.bio.unlabelled.txt',
}
data, label_to_id, _, _ = get_dataset(custom_data=data_list)
stats = {}
for k, data in data.items():
    token_size = [len(i) for i in data['data']]
    stats[k] = {
        'data_n': len(token_size),
        'max token size (train)': max(token_size),
        'min token size (train)': min(token_size),
        'avg token size (train)': sum(token_size)/len(token_size)
    }

df = pd.DataFrame(stats)
print(df)

