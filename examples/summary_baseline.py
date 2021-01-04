import os
import json
import pandas as pd
from pprint import pprint
from glob import glob

data = ["ontonotes5", "conll2003",  "wnut2017", "panx_dataset/en", "bionlp2004", "bc5cdr", "fin",
        "mit_restaurant", "mit_movie_trivia"]
all_data = data + ["all_5000", "all_10000", "all_15000", "all_no_mit_5000", "all_no_mit_10000", "all_no_mit_15000"]


def summary():
    dict_in_domain = {'f1': {'es': {}, 'ner': {}}, 'recall': {'es': {}, 'ner': {}}, 'precision': {'es': {}, 'ner': {}}}
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
        if param['lower_case']:
            continue

        if len(param['dataset']) > 1:
            total_step = param['total_step']
            if 'mit_restaurant' in param['dataset']:
                assert 'mit_movie_trivia' in param['dataset']
                train_data = 'all_{}'.format(total_step)
            else:
                train_data = 'all_no_mit_{}'.format(total_step)
        else:
            train_data = param['dataset'][0]

        if train_data not in all_data:
            continue

        for a in glob('{}/test*.json'.format(i)):
            test_data = a.split('test_')[-1].split('.json')[0]
            if 'lower' in test_data:
                continue
            test_data = test_data.replace('-', '/')
            test_data_raw = test_data.replace('_ignore', '').replace('_lower', '')
            if test_data_raw not in all_data:
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
            if test_data == train_data:
                dict_in_domain['f1'][task][test_data_raw] = f1
                dict_in_domain['recall'][task][test_data_raw] = recall
                dict_in_domain['precision'][task][test_data_raw] = precision

    columns = ['recall', 'precision', 'f1']
    in_result = [list((dict_in_domain[c]['ner'].values())) for c in columns]
    in_result_key = list(dict_in_domain['f1']['ner'].keys())
    df = pd.DataFrame(in_result, columns=in_result_key, index=columns).T
    df.to_csv('./ckpt/summary_in_domain.csv')
    pprint(df)
    for metric in ['f1', 'recall', 'precision']:
        for task in ['es', 'ner']:
            if task not in dict_out_domain[metric].keys():
                continue
            tmp_out = dict_out_domain[metric][task]
            tmp_df = pd.DataFrame(tmp_out).T
            pprint(tmp_df)
            tmp_df = tmp_df[data]
            tmp_df = tmp_df.T[all_data].T
            tmp_df.to_csv('./ckpt/summary_out_domain_{}_{}.csv'.format(task, metric))
            pprint(tmp_df)


if __name__ == '__main__':
    summary()
