import os
import json
import pandas as pd
from pprint import pprint
from glob import glob

data = ["ontonotes5", "conll2003",  "wnut2017", "panx_dataset/en", "bionlp2004", "bc5cdr", "fin",
        "mit_restaurant", "mit_movie_trivia"]
all_data_lower = data + ['all_lower']


def summary(base_model: bool = False):
    dict_out_domain = {
        'f1': {'es': {}, 'ner': {}},
        'recall': {'es': {}, 'ner': {}},
        'precision': {'es': {}, 'ner': {}}
    }
    checkpoint_dir = './ckpt'
    for i in glob('{}/*'.format(checkpoint_dir)):
        if not os.path.isdir(i):
            continue

        with open('{}/parameter.json'.format(i)) as f:
            param = json.load(f)
        if not base_model and param['transformers_model'] != "xlm-roberta-large":
            continue
        if base_model and param['transformers_model'] != "xlm-roberta-base":
            continue

        if not param['lower_case']:
            if len(param['dataset']) > 1:
                continue
            elif "mit_restaurant" != param['dataset'][0] and "mit_movie_trivia" != param['dataset'][0]:
                continue

        if len(param['dataset']) > 1:
            train_data = 'all_lower'
        else:
            train_data = param['dataset'][0]

        if train_data not in all_data_lower:
            continue

        for a in glob('{}/test*.json'.format(i)):
            test_data = a.split('test_')[-1].split('.json')[0]
            if 'lower' not in test_data:
                continue
            test_data = test_data.replace('-', '/')
            test_data_raw = test_data.replace('_ignore', '').replace('_lower', '')
            if test_data_raw not in all_data_lower:
                continue

            with open(a) as f:
                test = json.load(f)
            if 'test' in test.keys():
                metric = test['test']
            else:
                metric = test['valid']
            f1 = round(metric['f1'], 2)
            recall = round(metric['recall'], 2)
            precision = round(metric['precision'], 2)
            task = 'es' if 'ignore' in test_data else 'ner'

            if train_data not in dict_out_domain['f1'][task].keys():
                dict_out_domain['f1'][task][train_data] = {}
                dict_out_domain['recall'][task][train_data] = {}
                dict_out_domain['precision'][task][train_data] = {}
            dict_out_domain['f1'][task][train_data][test_data_raw] = f1
            dict_out_domain['recall'][task][train_data][test_data_raw] = recall
            dict_out_domain['precision'][task][train_data][test_data_raw] = precision

    for metric in ['f1', 'recall', 'precision']:
        for task in ['es', 'ner']:
            if task not in dict_out_domain[metric].keys():
                continue
            tmp_out = dict_out_domain[metric][task]
            tmp_df = pd.DataFrame(tmp_out).T
            pprint(tmp_df)
            tmp_df = tmp_df[data]
            tmp_df = tmp_df.T[all_data_lower].T
            if base_model:
                tmp_df.to_csv('./ckpt/summary_out_domain_{}_{}.lower.base.csv'.format(task, metric))
            else:
                tmp_df.to_csv('./ckpt/summary_out_domain_{}_{}.lower.csv'.format(task, metric))
            pprint(tmp_df)


if __name__ == '__main__':
    summary()
