import re
import json
import os
import pandas as pd
from pprint import pprint
from glob import glob


def summary():
    in_domain_file = './ckpt/summary_in_domain.json'
    out_domain_file = './ckpt/summary_out_domain.json'
    if not os.path.exists(in_domain_file) or not os.path.exists(out_domain_file):
        dict_in_domain = {'f1': {'es': {}, 'ner': {}}, 'recall': {'es': {}, 'ner': {}}, 'precision': {'es': {}, 'ner': {}}}
        dict_out_domain = {
            'f1': {'es': {}, 'ner': {}},
            'recall': {'es': {}, 'ner': {}},
            'precision': {'es': {}, 'ner': {}}
        }
        checkpoint_dir = './ckpt'
        for i in glob('{}/*'.format(checkpoint_dir)):
            with open('{}/parameter.json'.format(i)) as f:
                param = json.load(f)
            train_data = '+'.join(param['dataset'])
            total_step = param['total_step']
            # print(dataset, total_step)
            for a in glob('{}/test*.json'.format(i)):
                # if 'ignore' in a:
                # test_data = a.split('test_')[-1].split('_ignore.json')[0]
                test_data = a.split('test_')[-1].split('.json')[0]
                test_data = test_data.replace('-', '/')
                with open(a) as f:
                    test = json.load(f)
                if 'test' in test.keys():
                    metric = test['test']
                else:
                    metric = test['valid']
                f1 = round(metric['f1'], 2)
                recall = round(metric['recall'], 2)
                precision = round(metric['precision'], 2)
                if 'ignore' in test_data:
                    test_data = test_data.replace('_ignore', '')
                    task = 'es'
                else:
                    task = 'ner'

                if train_data not in dict_out_domain['f1'][task].keys():
                    dict_out_domain['f1'][task][train_data] = {}
                    dict_out_domain['recall'][task][train_data] = {}
                    dict_out_domain['precision'][task][train_data] = {}
                dict_out_domain['f1'][task][train_data][test_data] = f1
                dict_out_domain['recall'][task][train_data][test_data] = f1
                dict_out_domain['precision'][task][train_data][test_data] = f1
                if test_data == train_data:
                    dict_in_domain['f1'][task][test_data] = f1
                    dict_in_domain['recall'][task][test_data] = recall
                    dict_in_domain['precision'][task][test_data] = precision
        with open(in_domain_file, 'w') as f:
            json.dump(dict_in_domain, f)
        with open(out_domain_file, 'w') as f:
            json.dump(dict_out_domain, f)

    else:
        with open(in_domain_file, 'r') as f:
            dict_in_domain = json.load(f)
        with open(out_domain_file, 'r') as f:
            dict_out_domain = json.load(f)

    columns = ['recall', 'precision', 'f1']
    in_result = [list((dict_in_domain[c]['ner'].values())) for c in columns]
    in_result_key = list(dict_in_domain['f1']['ner'].keys())
    df = pd.DataFrame(in_result, columns=in_result_key, index=columns).T
    df.to_csv('./ckpt/summary_in_domain.csv')
    for metric in ['f1', 'recall', 'precision']:
        for task in ['es', 'ner']:
            tmp_out = dict_out_domain[metric][task]
            pprint(tmp_out)
            tmp = pd.DataFrame(tmp_out).T
            pprint(tmp)
            input()
    df_in_domain = {k: {_k: pd.DataFrame(_v) for _k, _v in v.items()} for k, v in dict_in_domain.items()}
    pprint(df_in_domain)
    # df_in_domain = {k: v for k, v in dict_in_domain.items()}
    # df_out_domain = pd.DataFrame


if __name__ == '__main__':
    summary()
