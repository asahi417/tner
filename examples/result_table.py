import re
import json
import pandas as pd
from pprint import pprint
from glob import glob


def summary(prefix: str = 'valid'):
    # df_in_domain = pd.DataFrame()
    # df_out_domain = pd.DataFrame
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
            metric = test[prefix]
            f1 = round(metric['f1'], 2)
            recall = round(metric['recall'], 2)
            precision = round(metric['precision'], 2)
            if 'ignore' in test_data:
                test_data = test_data.replace('_ignore', '')
                task = 'es'
            else:
                task = 'ner'

            if test_data != train_data:
                if train_data not in dict_out_domain['f1'][task].keys():
                    dict_out_domain['f1'][task][train_data] = {}
                    dict_out_domain['recall'][task][train_data] = {}
                    dict_out_domain['precision'][task][train_data] = {}
                dict_out_domain['f1'][task][train_data][test_data] = f1
                dict_out_domain['recall'][task][train_data][test_data] = f1
                dict_out_domain['precision'][task][train_data][test_data] = f1
            else:
                dict_in_domain['f1'][task][test_data] = f1
                dict_in_domain['recall'][task][test_data] = recall
                dict_in_domain['precision'][task][test_data] = precision
    with open('./ckpt/summary_in_domain.json', 'w') as f:
        json.dump(dict_in_domain, f)
    with open('./ckpt/summary_out_domain.json', 'w') as f:
        json.dump(dict_out_domain, f)


if __name__ == '__main__':
    summary()
