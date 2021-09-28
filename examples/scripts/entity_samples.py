from itertools import permutations

import pandas as pd
from tner import get_dataset_ner


data_list = ['conll2003', 'wnut2017', 'ontonotes5', 'fin', 'bionlp2004',
             'bc5cdr', 'panx_dataset_en']
custom_data = ['cache/SemEval2022-Task11_Train-Dev/EN-English']
data_list += custom_data

output = {}
for i in data_list:
    data, label_to_id, _, _ = get_dataset_ner(i)
    id_to_label = {v: k for k, v in label_to_id.items()}
    for _type in data.keys():
        if _type not in output:
            output[_type] = {}
        if i not in output[_type]:
            output[_type][i] = {}
        entity_label = [[(_x, id_to_label[_j]) for _x, _j in zip(x, j) if label_to_id['O'] != _j] for x, j in
                        zip(data[_type]['data'], data[_type]['label'])]
        for sent in entity_label:
            place_holder = []
            current_label = None
            for token, label in sent:
                if label.startswith('B-') and len(place_holder) != 0:
                    mention = ' '.join(place_holder)
                    if current_label not in output[_type][i].keys():
                        output[_type][i][current_label] = []
                    if mention not in output[_type][i][current_label]:
                        output[_type][i][current_label] += [' '.join(place_holder)]
                    place_holder = []
                    current_label = None
                else:
                    place_holder.append(token)
                    if current_label is None:
                        current_label = label.split('-')[-1]
        if len(place_holder) != 0:
            mention = ' '.join(place_holder)
            if mention not in output[_type][i][current_label]:
                output[_type][i][current_label] += [' '.join(place_holder)]
        output[_type][i] = {k: sorted(v) for k, v in output[_type][i].items()}


for k, v in output.items():
    df = pd.DataFrame(v)
    df.to_csv('examples/scripts/entity_samples.{}.csv'.format(k))

overlap = {}
overlap_mean = {}
# types = permutations(['train', 'valid', 'test'], 2)
types = [('test', 'train'), ('valid', 'train')]

for i in data_list:

    overlap[i] = {}
    overlap_mean[i] = {}

    for a_name, b_name in types:
        if i not in output[a_name] or i not in output[b_name]:
            continue
        a = output[a_name][i]
        b = output[b_name][i]
        # _key = '({0} AND {1})/|{0}|'.format(a_name, b_name)
        _key = a_name
        overlap[i][_key] = {}

        entity_train = []
        entity_eval = []

        for entity_type in a.keys():

            entity_train += a[entity_type]
            entity_eval += b[entity_type]

            overlap[i][_key][entity_type] = \
                len(set(a[entity_type]).intersection(set(b[entity_type])))/len(a[entity_type]) * 100

        overlap_mean[i][_key] = len(set(entity_eval).intersection(set(entity_train)))/len(list(set(entity_eval))) * 100

# TODO ovelap is not saved
pd.DataFrame(overlap_mean).to_csv('examples/scripts/entity_samples.overlap_mean.csv')
