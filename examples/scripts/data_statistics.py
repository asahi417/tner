import pandas as pd
from tner import get_dataset_ner

data_list = ['conll2003', 'wnut2017', 'ontonotes5', 'fin', 'bionlp2004',
             'bc5cdr', 'panx_dataset_en']
custom_data = ['cache/SemEval2022-Task11_Train-Dev/EN-English']
data_list += custom_data
stats = []
for i in data_list:
    data, label_to_id, _, _ = get_dataset_ner(i)
    _stats = {
        'data': i,
        'label': sorted(list(set([i.split('-')[-1] for i in label_to_id.keys() if i != 'O'])))
    }
    _stats['label size'] = len(_stats['label'])
    for k in data.keys():
        _stats[k] = len(data[k]['data'])

    token_size = [len(i) for i in data['train']['data']]
    _stats['max token size (train)'] = max(token_size)
    _stats['min token size (train)'] = min(token_size)
    _stats['avg token size (train)'] = sum(token_size)/len(token_size)
    stats.append(_stats)

df = pd.DataFrame(stats)
df.to_csv('examples/scripts/data_statistics.csv')

