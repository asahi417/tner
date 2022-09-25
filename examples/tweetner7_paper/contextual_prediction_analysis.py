""" Preprocess retrievad tweets' feature and plot analysis. """
import argparse
import logging
import os
import statistics
import json
from datetime import datetime
from tqdm import tqdm
from os.path import join as pj

from scipy.stats import rankdata
import pandas as pd
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt

from cner import Retriever
from cner.ner import get_dataset
from cner.ner.model import decode_ner_tags, label_to_id

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# parameters
parser = argparse.ArgumentParser(description='Prepare Tweet Retriever')
parser.add_argument('-n', '--ner', help='ner model', default='roberta_large', type=str)
parser.add_argument('-b', '--batch-size', help='batch size of embedding and ner', default=128, type=int)
parser.add_argument('-c', '--chunk-size', help='batch size of embedding', default=100, type=int)
args = parser.parse_args()

# fixed parameters
datetime_format = '%Y-%m-%d'
export = pj('output', 'contextual_prediction_analysis')
os.makedirs(export, exist_ok=True)
index_extra = pj('cner_output', 'index_extra', args.ner)
index_test2021 = pj('cner_output', 'index_test2021', args.ner)
search_result_cache = pj('cner_output', 'index_extra', args.ner, 'search_cache.test2021.json')
search_feature_cache = pj('cner_output', 'index_extra', args.ner, 'search_feature.test2021.json')
analysis_feature_cache = pj('cner_output', 'index_extra', args.ner, 'analysis_feature_cache.test2021.json')
analysis_td_cache = pj('cner_output', 'index_extra', args.ner, 'analysis_td_cache.test2021.json')


def cosine_similarity(a, b):
    norm_a = sum(map(lambda _x: _x * _x, a)) ** 0.5
    norm_b = sum(map(lambda _x: _x * _x, b)) ** 0.5
    return sum(map(lambda _x: _x[0] * _x[1], zip(a, b)))/norm_a/norm_b


def safe_mean(_list): return None if len(_list) == 0 else statistics.mean(_list)


# cache search result of test set
if not os.path.exists(search_result_cache):
    timeout = 1
    max_retrieval = 50000
    # setup retriever
    retriever = Retriever(index_dir=index_extra)
    assert len(retriever.embedding_cache) != 0
    assert len(retriever.ner_prediction_cache) != 0
    # data setting
    id_to_label = {v: k for k, v in label_to_id.items()}
    data = get_dataset('2021.test', return_date=True)
    data['date'] = list(map(lambda _x: datetime.strptime(_x, datetime_format), data['date']))
    data = list(zip(*[data[k] for k in data.keys()]))
    search_id = 0
    search_result = {}
    for x, y, date, _id in tqdm(data):
        y = [id_to_label[_y] for _y in y]
        out = decode_ner_tags(y, x)
        for o in out:
            entity = ' '.join(o['entity'])
            search_output = retriever.search(query_string=entity, limit=max_retrieval, timeout=timeout)
            if len(search_output) == 0:
                continue
            for r in search_output:
                r.pop('embedding')
                r.pop('ner')
                r['date'] = r.pop('datetime').strftime(datetime_format)
            search_result[str(search_id)] = {
                'search_output': search_output,
                'query': {'date': date.strftime(datetime_format), 'id': _id, 'entity': entity, 'label': o['type']}
            }
            search_id += 1
    with open(search_result_cache, 'w') as f_write:
        json.dump(search_result, f_write)


if not os.path.exists(search_feature_cache):
    with open(search_result_cache) as f_reader:
        search_result = json.load(f_reader)

    retriever_test2021 = Retriever(index_dir=index_test2021)
    retriever_extra = Retriever(index_dir=index_extra)
    with open(search_feature_cache, 'w') as f_writer:
        for k, v in tqdm(list(search_result.items())):
            search_output = v['search_output']
            query = v['query']

            e_org = retriever_test2021.embedding_cache[query['id']]

            # model prediction
            model_pred = [{'type': p['type'], 'prob': statistics.mean(p['probability'])}
                          for p in retriever_test2021.ner_prediction_cache[query['id']]
                          if ' '.join(p['entity']) == query['entity']]
            if len(model_pred) != 1:  # no prediction on the target tweet OR unique entity appears in multiple times
                continue

            # aggregate context information
            output = {k: []}
            for _v in search_output:
                sim = cosine_similarity(retriever_extra.embedding_cache[_v['id']], e_org)
                td = (datetime.strptime(_v['date'], datetime_format)
                      - datetime.strptime(query['date'], datetime_format)).days
                context_pred = [{
                    'entity': query['entity'],
                    'pred_original': model_pred[0]['type'],
                    'prob_original': model_pred[0]['prob'],
                    'id_original': query['id'],
                    'pred_retrieval': p['type'],
                    'prob_retrieval': statistics.mean(p['probability']),
                    'id_retrieval': _v['id'],
                    'label': query['label'],
                    'similarity': sim,
                    'timedelta': td,
                    'es_score': _v['score']
                } for p in retriever_extra.ner_prediction_cache[_v['id']] if query['entity'] == ' '.join(p['entity'])]
                if len(context_pred) == 0:
                    continue
                output[k].append(context_pred[0])
            f_writer.write(json.dumps(output) + '\n')

list_td = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
list_rank = [25, 50, 75, 100, None]
list_td_str = [str(i) for i in list_td]
list_rank_str = [str(i) for i in list_rank]
list_bins = [20]
list_bins_calibration = [10, 20]
style = ['s-', 'o:', '^--', '<-.', '.-', "+:"]


def aggregate_single_query(_query_result):
    ids = []
    id_relevancy = {}
    id_td = {}
    entity_feature = {}
    max_td = max(list_td)
    for i in _query_result:

        # discard td more than max td
        if i['timedelta'] > max_td:
            continue

        # entity-wise feature
        if i['id_retrieval'] not in entity_feature:
            entity_feature[i['id_retrieval']] = []
        entity_feature[i['id_retrieval']].append({
            'prob_retrieval': float(i['prob_retrieval']),
            'flag_accuracy_retrieval': i['label'] == i['pred_retrieval'],
            'flag_accuracy_original': i['label'] == i['pred_original'],
            'flag_same_prediction': i['pred_retrieval'] == i['pred_original'],
        })

        # text-wise feature
        if i['id_retrieval'] in ids:
            continue
        ids.append(i['id_retrieval'])
        id_relevancy[i['id_retrieval']] = -1 * i['es_score']
        id_td[i['id_retrieval']] = i['timedelta']

    # aggregated output
    output = {}
    for _td in list_td:
        id_conditioned = [i for i in ids if id_td[i] <= _td]
        if len(id_conditioned) == 0:
            continue

        ranking = rankdata([id_relevancy[i] for i in id_conditioned], method='min')
        output[str(_td)] = {}
        for _rank in list_rank:
            prob_retrieval_positive = []
            accuracy_positive = []
            prob_retrieval_negative = []
            accuracy_negative = []
            same_prediction_negative = []
            for _id, r in zip(id_conditioned, ranking):
                if _rank is None or r <= _rank:
                    for i in entity_feature[_id]:
                        if i['flag_accuracy_original']:
                            prob_retrieval_positive.append(i['prob_retrieval'])
                            accuracy_positive.append(i['flag_accuracy_retrieval'])
                        else:
                            prob_retrieval_negative.append(i['prob_retrieval'])
                            accuracy_negative.append(i['flag_accuracy_retrieval'])
                            same_prediction_negative.append(i['flag_same_prediction'])

            output[str(_td)][str(_rank)] = {
                'prob_retrieval_positive': safe_mean(prob_retrieval_positive),
                'accuracy_positive': safe_mean(accuracy_positive),
                'prob_retrieval_negative': safe_mean(prob_retrieval_negative),
                'accuracy_negative': safe_mean(accuracy_negative),
                'same_prediction_negative': safe_mean(same_prediction_negative)
            }
    return output


# aggregate result per query
if not os.path.exists(analysis_feature_cache):
    tmp_output = {str(td): {str(r): [] for r in list_rank} for td in list_td}
    with open(search_feature_cache) as f_reader:
        for line in tqdm(f_reader.readlines()):
            tmp = json.loads(line)
            query_id = list(tmp.keys())[0]
            query_result = tmp[query_id]
            if len(query_result) == 0:
                continue
            # drop duplication
            query_result = list({i['id_retrieval']: i for i in query_result}.values())
            # data frame
            df = pd.DataFrame([{
                'flag_accuracy_retrieval': single_query['label'] == single_query['pred_retrieval'],
                'flag_accuracy_original': single_query['label'] == single_query['pred_original'],
                'flag_same_prediction': single_query['pred_retrieval'] == single_query['pred_original'],
                'similarity': float(single_query['similarity']),
                'prob_retrieval': float(single_query['prob_retrieval']),
                'prob_original': float(single_query['prob_original']),
                'timedelta': abs(int(single_query['timedelta'])),
                'id': query_id,
                'es_score': float(single_query['es_score'])
            } for single_query in query_result])
            for td in list_td:
                df_td = df[df['timedelta'] <= td]
                df_td.loc[:, 'ranking'] = rankdata(-1 * df_td['es_score'].values, method='min')
                df_td['ranking'] = rankdata(-1 * df_td['es_score'].values, method='min')
                for rank in list_rank:
                    if rank is None:
                        df_td_rank = df_td
                    else:
                        df_td_rank = df_td[df_td['ranking'] <= rank]
                    df_td_rank = df_td_rank[['flag_accuracy_retrieval', 'flag_accuracy_original',
                                             'flag_same_prediction', 'similarity', 'prob_retrieval',
                                             'prob_original', 'id']]
                    tmp_output[str(td)][str(rank)] += list(df_td_rank.T.to_dict().values())

    with open(analysis_feature_cache, 'w') as f_writer:
        json.dump(tmp_output, f_writer)

with open(analysis_feature_cache) as f:
    data = json.load(f)
pretty_name_feature = {'similarity': 'Similarity', 'prob_retrieval': 'Model Confidence'}
for feature_name in ['similarity', 'prob_retrieval']:
    for td, _data in data.items():
        if td not in list_td_str:
            continue
        for n_bins in list_bins:
            plt.figure()
            for n, rank in enumerate(list_rank_str):
                __data = _data[rank]
                if rank not in list_rank_str:
                    continue
                df = pd.DataFrame(__data)
                df = df[~df['flag_accuracy_original']]
                x, y = calibration_curve(df['flag_accuracy_retrieval'], df[feature_name], n_bins=n_bins, normalize=True)
                plt.plot(y, x * 100, style[n], label=f'Top-{rank}' if rank != 'None' else 'All', )
            plt.legend(loc='best')
            plt.ylabel('Ratio of Positives (%)', fontsize=14)
            plt.xlabel(pretty_name_feature[feature_name], fontsize=14)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(pj(export, f'ttr.negative.{feature_name}.td{td}.bin{n_bins}.png'))
            plt.close()

            plt.figure()
            for n, rank in enumerate(list_rank_str):
                __data = _data[rank]
                if rank not in list_rank_str:
                    continue
                df = pd.DataFrame(__data)
                df = df[df['flag_accuracy_original']]
                x, y = calibration_curve(df['flag_accuracy_retrieval'], df[feature_name], n_bins=n_bins, normalize=True)
                plt.plot(y, x * 100, style[n], label=f'Top-{rank}' if rank != 'None' else 'All')
            plt.legend(loc='best')
            plt.ylabel('Ratio of Positives (%)', fontsize=14)
            plt.xlabel(pretty_name_feature[feature_name], fontsize=14)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(pj(export, f'ttr.positive.{feature_name}.td{td}.bin{n_bins}.png'))
            plt.close()

df = pd.DataFrame(data[str(list_td[0])][str(list_rank[0])])
df = df.groupby('id').sample(1)  # sample one example from unique anchor tweet
plt.figure()
plt.plot([0, 1], [0, 100], linestyle='--', label='Ideally Calibrated')  # Plot perfectly calibrated
for n_bins in list_bins_calibration:
    x, y = calibration_curve(df['flag_accuracy_original'], df['prob_original'], n_bins=n_bins, normalize=True)
    plt.plot(y, x * 100, marker='.', linestyle=':', label=f'Bin: {n_bins}')
plt.legend(loc='best')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Ratio of Positives (%)', fontsize=12)
plt.savefig(pj(export, f'calibration_curve.png'))
plt.close()

# aggregate result per query
if not os.path.exists(analysis_td_cache):
    with open(analysis_td_cache, 'w') as f_writer:
        with open(search_feature_cache) as f_reader:
            for line in tqdm(f_reader.readlines()):
                tmp = json.loads(line)
                query_id = list(tmp.keys())[0]
                query_result = tmp[query_id]
                # drop duplicate
                query_result = list({i['id_retrieval']: i for i in query_result}.values())
                f_writer.write(json.dumps(aggregate_single_query(query_result)) + '\n')

metrics = ['prob_retrieval_positive', 'accuracy_positive', 'prob_retrieval_negative', 'accuracy_negative',
           'same_prediction_negative']
with open(analysis_td_cache) as f:
    output = {k: [] for k in metrics}
    for i in f.read().split('\n'):
        if len(i) == 0:
            continue
        tmp = json.loads(i)
        for td, v in tmp.items():
            if td not in list_td_str:
                continue
            for k in metrics:
                _dict = {'td': td}
                _dict.update({f'Top-{rank}' if rank != 'None' else 'All': _v[k] for rank, _v in v.items()
                              if rank in list_rank_str})
                output[k].append(_dict)

#############################################################
# PLOT (Accuracy for Positive/Negative Original Prediction) #
#############################################################
for accuracy in ['accuracy_positive', 'accuracy_negative']:
    plt.figure()
    df = pd.DataFrame(output[accuracy])
    df['td'] = df.pop('td').astype(int)
    df_g = df.groupby('td').mean() * 100
    df_g.plot(style=style)
    plt.legend(loc='best', fontsize=15)
    plt.ylabel('Ratio of Positives (%)', fontsize=18)
    plt.xlabel('Timedelta (day)', fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(pj(export, f'ttr.{accuracy.replace("accuracy_", "")}.td.png'))
    plt.close()

##########################################################
# PLOT (Break down of negative prediction for each rank) #
##########################################################
for rank in list_rank:
    rank_name = f'Top-{rank}' if rank is not None else 'All'

    df_n = pd.DataFrame(output['accuracy_negative'])
    df_n['td'] = df_n.pop('td').astype(int)
    df_n = df_n.groupby('td').mean()[rank_name] * 100

    df_s = pd.DataFrame(output['same_prediction_negative'])
    df_s['td'] = df_s.pop('td').astype(int)
    df_s = df_s.groupby('td').mean()[rank_name] * 100

    df_r = 100 - (df_s + df_n)

    plt.figure()
    df = pd.DataFrame([df_n, df_s, df_r], index=['Positive', 'Negative/Same Error', 'Negative/Other Errors']).T
    df.plot(style=style)
    plt.legend(loc='best', fontsize=15)
    plt.ylabel('Ratio (%)', fontsize=18)
    plt.xlabel('Timedelta (day)', fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(pj(export, f'negatives_breakdown.{rank_name}.png'))
    plt.close()
