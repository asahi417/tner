import json
import os
import requests
from os.path import join as pj

import pandas as pd

from matplotlib import pyplot as plt

MODELS_BASELINE = ['bert-base', 'bert-large', 'bertweet-base', 'bertweet-large', 'roberta-base', 'roberta-large',
                   'twitter-roberta-base-2019-90m', 'twitter-roberta-base-dec2020', 'twitter-roberta-base-dec2021']
MODELS_SELF_LABEL = ['roberta-large']
MODELS_RANDOM = ['bertweet-base', 'bert-base', 'bert-large', 'roberta-base', 'roberta-large',
                 'twitter-roberta-base-2019-90m', 'twitter-roberta-base-dec2021']
SPLIT = ['2020.test', '2021.test']
TMP_DIR = 'metric_files'
EXPORT_DIR = pj('output', 'model_finetuning')


def plot(path):
    pretty_name = {
        'bert-base': 'BERT (BASE)',
        'bert-large': 'BERT (LARGE)',
        'bertweet-base': 'BERTweet (BASE)',
        'bertweet-large': 'BERTweet (LARGE)',
        'roberta-base': 'RoBERTa (BASE)',
        'roberta-large': 'RoBERTa (LARGE)',
        'twitter-roberta-base-2019-90m': 'TimeLM (2019)',
        'twitter-roberta-base-dec2020': 'TimeLM (2020)',
        'twitter-roberta-base-dec2021': 'TimeLM (2021)'
    }
    labels = ['corporation', 'creative_work', 'event', 'group', 'location', 'person', 'product']
    _df = pd.read_csv(path)
    _df = _df[_df.experiment == '2020']
    _df = _df[['model'] + [f'{i}/f1' for i in labels]]
    _df.columns = ['model'] + [i.replace('_', ' ') for i in labels]
    _df.index = [pretty_name[i] for i in _df.pop('model')]
    _df = _df.T[list(pretty_name.values())]

    # plt.figure(figsize=(24, 24))
    ax = _df.plot.bar(width=0.9, figsize=(10, 14))

    # hatch
    bars = ax.patches
    patterns = ('', '-', '+', 'x', '/', '//', 'O', 'o', '\\', '\\\\')
    hatches = [p for p in patterns for i in range(len(_df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_xticklabels(_df.index, rotation=30)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, fontsize=24)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.7), ncol=2, fontsize=12)
    plt.ylabel('F1 score', fontsize=26)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(pj(EXPORT_DIR, 'entity_breakdown.png'))


def download(filename, url):
    print(f'download {url}')
    try:
        with open(filename) as f_reader:
            json.load(f_reader)
    except Exception:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f_reader:
            r = requests.get(url)
            f_reader.write(r.content)
    with open(filename) as f_reader:
        tmp = json.load(f_reader)
    return tmp


def format_result(metric, experiment_name):
    output = []
    for split in SPLIT:
        data = metric[split]
        data['split'] = split
        data['experiment'] = experiment_name
        data['model'] = m
        data_span = metric[f"{split} (span detection)"]
        data['type-ignored/f1'] = round(data_span['micro/f1'] * 100, 1)
        for avg in ["micro", "macro"]:
            data.pop(f'{avg}/recall')
            data.pop(f'{avg}/precision')
            data[f'{avg}/f1'] = round(data[f'{avg}/f1'] * 100, 1)
            for k, v in data.pop(f'{avg}/f1_ci').items():
                low = round(v[0] * 100, 1)
                high = round(v[1] * 100, 1)
                data[f'{avg}/f1_ci/{k}'] = f"{data[f'{avg}/f1']} [{low}, {high}]"
        if 'per_entity_metric' in data:
            for k, v in data.pop('per_entity_metric').items():
                data[f'{k}/f1'] = round(v['f1'] * 100, 1)
                for _k, _v in v['f1_ci'].items():
                    low = round(_v[0] * 100, 1)
                    high = round(_v[1] * 100, 1)
                    data[f'{k}/f1_ci/{_k}'] = f"{data[f'{k}/f1']} [{low}, {high}]"
        output.append(data)
    return output


full_output = []

# baseline
for m in MODELS_BASELINE:
    _metric = download(filename=f'{TMP_DIR}/baseline.{m}.json',
                       url=f"https://huggingface.co/tner/{m}-tweetner-2020/raw/main/eval/metric.json")
    full_output += format_result(_metric, '2020')

# 2021 model training
for m in MODELS_BASELINE:
    for _type in ['2021', '2020-2021-continuous', '2020-2021-concat']:
        _metric = download(filename=f'{TMP_DIR}/2021.{m}.{_type}.json',
                           url=f"https://huggingface.co/tner/{m}-tweetner-{_type}/raw/main/eval/metric.json")
        full_output += format_result(_metric, _type)

# self labeling
for m in MODELS_SELF_LABEL:
    for _type in ['selflabel2020', '2020-selflabel2020-continuous', '2020-selflabel2020-concat',
                  'selflabel2021', '2020-selflabel2021-continuous', '2020-selflabel2021-concat']:
        _metric = download(filename=f'{TMP_DIR}/selflabel.{m}.{_type}.json',
                           url=f"https://huggingface.co/tner/{m}-tweetner-{_type}/raw/main/eval/metric.json")
        full_output += format_result(_metric, _type)

# random
for m in MODELS_BASELINE:
    _metric = download(filename=f'{TMP_DIR}/random.{m}.json',
                       url=f"https://huggingface.co/tner/{m}-tweetner-random/raw/main/eval/metric.json")
    full_output += format_result(_metric, 'random')

df = pd.DataFrame(full_output)
os.makedirs(EXPORT_DIR, exist_ok=True)
for _split, g in df.groupby("split"):
    if _split not in SPLIT:
        continue
    g.to_csv(pj(EXPORT_DIR, f'result.{_split}.csv'), index=False)

plot(pj(EXPORT_DIR, f'result.2021.test.csv'))
