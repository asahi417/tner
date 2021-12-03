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
        best_model, _ = tmp[0]
        with open('{}/trainer_config.json'.format(os.path.dirname(best_model))) as f:
            config = json.load(f)
        print('\t - config: {}'.format(config))
        with open('{}/eval/metric.json'.format(best_model)) as f:
            tmp = json.load(f)
            print('\t - best micro f1 (test): {}'.format(tmp['test']['micro/f1']))
        print()
