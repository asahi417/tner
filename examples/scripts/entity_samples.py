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
