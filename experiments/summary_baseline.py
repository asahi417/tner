import os
import json
import pandas as pd
from pprint import pprint
from glob import glob

data_ = ["ontonotes5", "conll2003",  "wnut2017", "panx_dataset/en", "bionlp2004", "bc5cdr", "fin"]
data_lower = data_ + ["mit_restaurant", "mit_movie_trivia"]


def summary(base_model: bool = False, lower: bool = False):
    dict_in_domain = {'f1': {'es': {}, 'ner': {}}, 'recall': {'es': {}, 'ner': {}}, 'precision': {'es': {}, 'ner': {}}}
    dict_out_domain = {
        'f1': {'es': {}, 'ner': {}},
        'recall': {'es': {}, 'ner': {}},
        'precision': {'es': {}, 'ner': {}}
    }
    if lower:
        data = data_lower
    else:
        data = data_
    if base_model:
        if lower:
            checkpoint_dir = './ckpt/model_base_lower'
        else:
            checkpoint_dir = './ckpt/model_base'
    else:
        if lower:
            checkpoint_dir = './ckpt/model_large_lower'
        else:
            checkpoint_dir = './ckpt/model_large'

    print(glob('{}/*'.format(checkpoint_dir)))
    for i in glob('{}/*'.format(checkpoint_dir)):
        if not os.path.isdir(i):
            continue

        with open('{}/parameter.json'.format(i)) as f:
            param = json.load(f)
        if not base_model and param['transformers_model'] != "xlm-roberta-large":
            continue
        if base_model and param['transformers_model'] != "xlm-roberta-base":
            continue
        if lower:
            if not param['lower_case']:
                continue
        else:
            if param['lower_case']:
                continue
        print(i)
        if len(param['dataset']) > 1:
            train_data = 'all'
        else:
            train_data = param['dataset'][0]
            if train_data not in data:
                continue

        for a in glob('{}/test*.json'.format(i)):
            test_data = a.split('test_')[-1].split('.json')[0]
            if lower:
                if 'lower' not in test_data:
                    continue
            else:
                if 'lower' in test_data:
                    continue

            print(test_data)
            test_data = test_data.replace('-', '/')
            test_data_raw = test_data.replace('_ignore', '').replace('_lower', '').replace('_span', '')
            print(test_data_raw)
            if lower:
                if test_data_raw not in data_lower:
                    continue
            else:
                if test_data_raw not in data:
                    continue

            print(test_data)
            with open(a) as f:
                test = json.load(f)
            if 'test' in test.keys():
                metric = test['test']
            else:
                metric = test['valid']
            f1 = round(metric['f1'], 2)
            recall = round(metric['recall'], 2)
            precision = round(metric['precision'], 2)
            task = 'es' if 'ignore' in test_data or 'span' in test_data else 'ner'

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

    print(dict_in_domain)
    print(dict_out_domain)
    if not lower:
        columns = ['recall', 'precision', 'f1']
        in_result = [list((dict_in_domain[c]['ner'].values())) for c in columns]
        in_result_key = list(dict_in_domain['f1']['ner'].keys())
        df = pd.DataFrame(in_result, columns=in_result_key, index=columns).T
        df.to_csv('{}/summary_in_domain.csv'.format(checkpoint_dir))
        pprint(df)
    for metric in ['f1', 'recall', 'precision']:
        for task in ['es', 'ner']:
            if task not in dict_out_domain[metric].keys():
                continue
            tmp_out = dict_out_domain[metric][task]
            tmp_df = pd.DataFrame(tmp_out).T
            pprint(tmp_out)
            pprint(tmp_df.columns)
            if task == 'es':
                tmp_df = tmp_df[data]
                all_data = data + ["all"]
                tmp_df = tmp_df.T[all_data].T
            else:
                tmp_df = tmp_df.T['all']
            tmp_df.to_csv('{}/summary_out_domain.{}.{}.csv'.format(checkpoint_dir, task, metric))
            pprint(tmp_df)


if __name__ == '__main__':
    summary(True, True)
    summary(True, False)
    # summary(False, True)
    # summary(False, False)
