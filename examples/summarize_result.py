import os
import json
from glob import glob

for i in glob('./tner_output/search/*'):
    basename = os.path.basename(i)
    print('* Model: {}'.format(basename))
    for h in glob('{}/*/metric.2nd.json'.format(i)):
        dataset = os.path.basename(os.path.dirname(h))
        print('\t - data: {}'.format(dataset))
        with open(h) as f:
            tmp = json.load(f)
        _, best_metric = tmp[0]
        best_models = [t[0] for t in tmp if t[1] == best_metric]
        if len(best_models) > 1:
            print('\t WARNING: {} best models: {}'.format(len(best_models), best_models))
        best_model = best_models[0]
        with open('{}/trainer_config.json'.format(os.path.dirname(best_model))) as f:
            config = json.load(f)
        print('\t - config: {}'.format(config))
        with open('{}/eval/metric.json'.format(best_model)) as f:
            tmp = json.load(f)
            print('\t - best micro f1 (test)  : {}'.format(tmp['test']['micro/f1']))
        full_metric = []
        for m in glob('{}/*/eval/metric.json'.format(os.path.dirname(best_model))):
            with open(m) as f:
                full_metric.append(json.load(f)['test']['micro/f1'])
        print('\t - oracle micro f1 (test): {}'.format(max(full_metric)))
        print()
