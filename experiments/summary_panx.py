import os
import json
import pandas as pd
from pprint import pprint
from glob import glob

panx_data = ["panx_dataset/en", "panx_dataset/ja", "panx_dataset/ru", "panx_dataset/ko", "panx_dataset/es",
             "panx_dataset/ar"]


def summary():
    dict_in_domain = {'f1': {'es': {}, 'ner': {}}, 'recall': {'es': {}, 'ner': {}}, 'precision': {'es': {}, 'ner': {}}}
    dict_out_domain = {
        'f1': {'es': {}, 'ner': {}},
        'recall': {'es': {}, 'ner': {}},
        'precision': {'es': {}, 'ner': {}}
    }
    checkpoint_dir = './ckpt/model_large'
    for i in glob('{}/*'.format(checkpoint_dir)):
        if not os.path.isdir(i):
            continue

        with open('{}/parameter.json'.format(i)) as f:
            param = json.load(f)
        if param['transformers_model'] != "xlm-roberta-large":
            continue
        if param['lower_case']:
            continue

        if len(param['dataset']) > 1:
            continue
        train_data = param['dataset'][0]
        if train_data not in panx_data:
            continue

        for a in glob('{}/test*.json'.format(i)):
            test_data = a.split('test_')[-1].split('.json')[0]
            if 'lower' in test_data:
                continue
            test_data = test_data.replace('-', '/')
            test_data_raw = test_data.replace('_ignore', '').replace('_lower', '')
            if 'ignore' in test_data:
                continue
            if test_data_raw not in panx_data:
                continue

            with open(a) as f:
                test = json.load(f)
            metric = test['test']
            f1 = round(metric['f1'], 2)
            recall = round(metric['recall'], 2)
            precision = round(metric['precision'], 2)
            task = 'ner'

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
    df.to_csv('{}/summary_in_domain.panx.csv'.format(checkpoint_dir))
    pprint(df)
    for metric in ['f1', 'recall', 'precision']:
        task = 'ner'
        tmp_out = dict_out_domain[metric][task]
        tmp_df = pd.DataFrame(tmp_out).T
        pprint(tmp_df)
        tmp_df = tmp_df[panx_data]
        tmp_df = tmp_df.T[panx_data].T
        tmp_df.to_csv('{}/summary_out_domain.ner.{}.panx.csv'.format(checkpoint_dir, metric))
        pprint(tmp_df)


if __name__ == '__main__':
    summary()
