from datetime import datetime
import pandas as pd
import json
from glob import glob

path = 'cache/twitter_ner/prefiltered/*.jsonl'
datetime_format = '%Y-%m-%d'
data = []


def format_single_data(_data):
    _data = json.loads(_data)
    _data['created_at'] = _data['created_at'].split('T')[0]
    return _data


for i in glob(path):
    with open(i) as f:
        data += [format_single_data(i) for i in f.read().split('\n') if len(i) > 0]

df = pd.DataFrame(data)
df.pop('entities')
df.pop('text')
df['tweet'] = df.pop('text_clean')
df['created_at'] = [datetime.strptime(i, datetime_format) for i in df['created_at']]
df_2020 = df[df['created_at'] < '2020-09-01']
df_2021 = df[df['created_at'] >= '2020-09-01']
df_2020.to_csv('cache/twitter_ner/extra_csv/2020.extra_xl.csv', index=False)
df_2021.to_csv('cache/twitter_ner/extra_csv/2021.extra_xl.csv', index=False)


# data[self.column_datetime] = datetime.strptime(_dict[column_datetime], datetime_format),