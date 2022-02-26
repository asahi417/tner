import os
import json
from glob import glob
import pandas as pd

path_baseline = './tner_output/model/baseline/roberta_large'
path_self_train = './tner_output/model/self_training/roberta_large_st'


def json_reader(path):
    with open(path) as f:
        return json.load(f)


def get_info(path):
    if path == 'test':
        return {}
    time_delta, ranking, th_prob, th_sim, imp = path.split('.')[-5:]
    return {
        'ranking': ranking,
        'time delta (day)': int(int(time_delta) / 24),
        'threshold (probability)': int(th_prob) * 0.01,
        'threshold (similarity)': int(th_sim) * 0.01,
        'contextual prediction discount factor': int(imp) * 0.01
    }


def main(path):
    # statistics
    files = glob('{}/eval/*.stats.json'.format(path))
    data = [json_reader(i) for i in files]
    for stats, _path in zip(data, files):
        stats['filename'] = os.path.basename(_path)
        stats.update(get_info(_path.replace('.stats.json', '')))

    df = pd.DataFrame(data)
    df['relative success'] = df['success'] - df['fail']
    df = df.sort_values(by=['relative success'])
    # metric
    metric = json_reader('{}/eval/metric.json'.format(path))
    output = []
    for k, m in metric.items():
        if 'per_entity_metric' in m:
            _tmp = m['per_entity_metric']
        _tmp_metric = {'f1': m['micro/f1'] * 100}
        _tmp_metric.update({'f1/{}'.format(k): v['f1'] * 100 for k, v in _tmp.items()})
        _tmp_metric.update(get_info(k))
        output.append(_tmp_metric)
    df_metric = pd.DataFrame(output)
    return df, df_metric


if __name__ == '__main__':
    for __path in [path_baseline, path_self_train]:
        a, b = main(__path)
        a.to_csv('{}/eval/summary.stats.csv'.format(__path))
        b.to_csv('{}/eval/summary.csv'.format(__path))
