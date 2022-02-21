import glob
import json
import os
import logging
import string
import random
from typing import List, Dict
from itertools import product

from .trainer import Trainer, get_model_instance
from .data import get_dataset, CACHE_DIR


def get_random_string(length: int = 6, exclude: List = None):
    tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    if exclude:
        while tmp in exclude:
            tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    return tmp


def evaluate(model,
             batch_size,
             max_length,
             base_model=None,
             data=None,
             custom_dataset=None,
             export_dir=None,
             data_cache_prefix: str = None,
             contextualisation_cache_prefix: str = None,
             force_update: bool = False,
             lower_case: bool = False,
             span_detection_mode: bool = False,
             adapter: bool = False,
             entity_list: bool = False,
             index_data_path: str = None,
             index_prediction_path: str = None,
             max_retrieval_size: int = 10,
             timeout: int = None,
             datetime_format: str = '%Y-%m-%d',
             timedelta_hour_before: float = None,
             timedelta_hour_after: float = None,
             embedding_model: str = 'sentence-transformers/all-mpnet-base-v1',
             threshold_prob: float = 0.75,  # discard prediction under the value
             threshold_similarity: float = 0.75,  # discard prediction under the value
             retrieval_importance: float = 0.75,  # weight to the retrieved predictions
             ranking_type: str = 'similarity'  # ranking type 'similarity'/'es_score'/'frequency'
             ):
    """ Evaluate question-generation model """
    metrics_dict = {}
    path_metric = None
    export_prediction = None
    if export_dir is not None:
        export_prediction = '{}/prediction'.format(export_dir)
        path_metric = '{}/metric.json'.format(export_dir)
        if os.path.exists(path_metric):
            try:
                with open(path_metric, 'r') as f:
                    metrics_dict = json.load(f)
                if not force_update:
                    return metrics_dict
            except Exception:
                logging.warning('error at reading {}'.format(path_metric))

        os.makedirs(export_dir, exist_ok=True)
    lm = get_model_instance(base_model, max_length, model_path=model, adapter=adapter,
                            index_data_path=index_data_path, index_prediction_path=index_prediction_path)
    lm.eval()
    dataset_split, _, _, _ = get_dataset(data=data,
                                         custom_data=custom_dataset,
                                         lower_case=lower_case,
                                         label_to_id=lm.label2id,
                                         fix_label_dict=True)

    if data_cache_prefix is not None and lm.crf_layer is not None:
        data_cache_prefix = '{}.crf'.format(data_cache_prefix)
    for split in dataset_split.keys():
        if split == 'train':
            continue
        if data_cache_prefix is not None:
            cache_data_path = '{}.{}.pkl'.format(data_cache_prefix, split)
        else:
            cache_data_path = None
        split_alias = split
        if span_detection_mode:
            split_alias = '{}/span_detection'.format(split_alias)
        if lower_case:
            split_alias = '{}/lower_case'.format(split_alias)
        if contextualisation_cache_prefix is not None:
            split_alias = '{}/{}'.format(split_alias, contextualisation_cache_prefix)
        cache_prediction_path = None
        if export_prediction is not None:
            if lower_case:
                cache_prediction_path = '{}.{}.lower.txt'.format(export_prediction, split)
                if contextualisation_cache_prefix is not None:
                    contextualisation_cache_prefix = '{}.{}.{}.lower.txt'.format(
                        export_prediction, split, contextualisation_cache_prefix)
            else:
                cache_prediction_path = '{}.{}.txt'.format(export_prediction, split)
                if contextualisation_cache_prefix is not None:
                    contextualisation_cache_prefix = '{}.{}.{}.txt'.format(
                        export_prediction, split, contextualisation_cache_prefix)
        dates = dataset_split[split]['date'] if 'date' in dataset_split[split] else None
        metrics_dict[split_alias] = lm.span_f1(
            inputs=dataset_split[split]['data'],
            labels=dataset_split[split]['label'],
            dates=dates,
            batch_size=batch_size,
            cache_data_path=cache_data_path,
            cache_prediction_path=cache_prediction_path,
            cache_prediction_path_contextualisation=contextualisation_cache_prefix,
            cache_embedding_path=cache_embedding_path,
            span_detection_mode=span_detection_mode,
            entity_list=entity_list,
            max_retrieval_size=max_retrieval_size,
            datetime_format=datetime_format,
            timedelta_hour_before=timedelta_hour_before,
            timedelta_hour_after=timedelta_hour_after,
            timeout=timeout,
            embedding_model=embedding_model,
            threshold_prob=threshold_prob,
            threshold_similarity=threshold_similarity,
            retrieval_importance=retrieval_importance,
            ranking_type=ranking_type
        )
    if path_metric is not None:
        with open(path_metric, 'w') as f:
            json.dump(metrics_dict, f)
    return metrics_dict


class GridSearcher:
    """ Grid search (epoch, batch, lr, random_seed, label_smoothing) """

    def __init__(self,
                 checkpoint_dir: str,
                 dataset: (str, List) = None,
                 custom_dataset: Dict = None,
                 model: str = 'xlm-roberta-large',
                 fp16: bool = False,
                 lower_case: bool = False,
                 epoch: int = 10,
                 epoch_partial: int = 5,
                 n_max_config: int = 5,
                 max_length: int = 128,
                 max_length_eval: int = 128,
                 batch_size: int = 32,
                 batch_size_eval: int = 16,
                 gradient_accumulation_steps: List or int = 1,
                 crf: List or bool = True,
                 lr: List or float = 1e-4,
                 weight_decay: List or float = None,
                 random_seed: List or int = 0,
                 lr_warmup_step_ratio: List or int = None,
                 max_grad_norm: List or float = None,
                 metric: str = 'micro/f1',
                 inherit_tner_checkpoint: bool = False,
                 adapter: bool = False,
                 adapter_config: Dict = None):
        self.inherit_tner_checkpoint = inherit_tner_checkpoint
        # evaluation configs
        assert metric in ['macro/f1', 'micro/f1']
        self.eval_config = {
            'max_length_eval': max_length_eval,
            'metric': metric
        }

        dataset_split, _, _, _ = get_dataset(dataset, lower_case=lower_case, custom_data=custom_dataset)
        assert 'valid' in dataset_split.keys(), 'Grid search need validation set.'
        # static configs
        self.static_config = {
            'dataset': dataset,
            'custom_dataset': custom_dataset,
            'model': model,
            'fp16': fp16,
            'batch_size': batch_size,
            'epoch': epoch,
            'max_length': max_length,
            'lower_case': lower_case,
            'adapter': adapter,
            'adapter_config': adapter_config
        }

        # dynamic config
        self.epoch = epoch
        self.epoch_partial = epoch_partial
        self.batch_size_eval = batch_size_eval
        self.checkpoint_dir = checkpoint_dir
        self.n_max_config = n_max_config

        def to_list(_val):
            if type(_val) != list:
                return [_val]
            assert len(_val) == len(set(_val)), _val
            if None in _val:
                _val.pop(_val.index(None))
                return [None] + sorted(_val, reverse=True)
            return sorted(_val, reverse=True)

        self.dynamic_config = {
            'lr': to_list(lr),
            'crf': to_list(crf),
            'random_seed': to_list(random_seed),
            'weight_decay': to_list(weight_decay),
            'lr_warmup_step_ratio': to_list(lr_warmup_step_ratio),
            'max_grad_norm': to_list(max_grad_norm),
            'gradient_accumulation_steps': to_list(gradient_accumulation_steps)
        }

        self.all_dynamic_configs = list(product(
            self.dynamic_config['lr'],
            self.dynamic_config['crf'],
            self.dynamic_config['random_seed'],
            self.dynamic_config['weight_decay'],
            self.dynamic_config['lr_warmup_step_ratio'],
            self.dynamic_config['max_grad_norm'],
            self.dynamic_config['gradient_accumulation_steps'],
        ))

    def initialize_searcher(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if os.path.exists('{}/config_static.json'.format(self.checkpoint_dir)):
            with open('{}/config_static.json'.format(self.checkpoint_dir)) as f:
                tmp = json.load(f)
            tmp_v = [tmp[k] for k in sorted(tmp.keys())]
            static_tmp_v = [self.static_config[k] for k in sorted(tmp.keys())]
            assert tmp_v == static_tmp_v, '{}\n not matched \n{}'.format(str(tmp_v), str(static_tmp_v))

        if os.path.exists('{}/config_dynamic.json'.format(self.checkpoint_dir)):
            with open('{}/config_dynamic.json'.format(self.checkpoint_dir)) as f:
                tmp = json.load(f)

            tmp_v = [tmp[k] for k in sorted(tmp.keys())]
            dynamic_tmp_v = [self.dynamic_config[k] for k in sorted(tmp.keys())]

            assert tmp_v == dynamic_tmp_v

        if os.path.exists('{}/config_eval.json'.format(self.checkpoint_dir)):
            with open('{}/config_eval.json'.format(self.checkpoint_dir)) as f:
                tmp = json.load(f)
            tmp_v = [tmp[k] for k in sorted(tmp.keys())]
            eval_tmp_v = [self.eval_config[k] for k in sorted(tmp.keys())]
            assert tmp_v == eval_tmp_v, '{}\n not matched \n{}'.format(str(tmp_v), str(eval_tmp_v))

        with open('{}/config_static.json'.format(self.checkpoint_dir), 'w') as f:
            json.dump(self.static_config, f)
        with open('{}/config_dynamic.json'.format(self.checkpoint_dir), 'w') as f:
            json.dump(self.dynamic_config, f)
        with open('{}/config_eval.json'.format(self.checkpoint_dir), 'w') as f:
            json.dump(self.eval_config, f)

        # add file handler
        logger = logging.getLogger()
        file_handler = logging.FileHandler('{}/grid_search.log'.format(self.checkpoint_dir))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
        logger.addHandler(file_handler)

        logging.info('INITIALIZE GRID SEARCHER: {} configs to try'.format(len(self.all_dynamic_configs)))

    def run(self, num_workers: int = 0, interval: int = 25):

        self.initialize_searcher()
        if self.static_config['dataset'] is not None:
            _data = '_'.join(sorted(self.static_config['dataset'])) if type(self.static_config['dataset']) is list else self.static_config['dataset']
            data_cache_prefix = '{}/data_encoded/{}.{}.{}{}'.format(
                CACHE_DIR,
                _data,
                self.static_config['model'],
                self.static_config['max_length'],
                '.lower' if self.static_config['lower_case'] else '',
            )
        else:
            data_cache_prefix = None

        ###########
        # 1st RUN #
        ###########
        checkpoints = []
        ckpt_exist = {}
        for trainer_config in glob.glob('{}/model_*/trainer_config.json'.format(self.checkpoint_dir)):
            with open(trainer_config, 'r') as f:
                ckpt_exist[os.path.dirname(trainer_config)] = json.load(f)

        for n, dynamic_config in enumerate(self.all_dynamic_configs):
            logging.info('## 1st RUN: Configuration {}/{} ##'.format(n, len(self.all_dynamic_configs)))
            config = self.static_config.copy()
            tmp_dynamic_config = {
                'lr': dynamic_config[0],
                'crf': dynamic_config[1],
                'random_seed': dynamic_config[2],
                'weight_decay': dynamic_config[3],
                'lr_warmup_step_ratio': dynamic_config[4],
                'max_grad_norm': dynamic_config[5],
                'gradient_accumulation_steps': dynamic_config[6],
            }
            config.update(tmp_dynamic_config)
            ex_dynamic_config = [(k_, [v[k] for k in sorted(tmp_dynamic_config.keys())]) for k_, v in ckpt_exist.items()]
            tmp_dynamic_config = [tmp_dynamic_config[k] for k in sorted(tmp_dynamic_config.keys())]
            duplicated_ckpt = [k for k, v in ex_dynamic_config if v == tmp_dynamic_config]

            if len(duplicated_ckpt) == 1:
                logging.info('skip as the config exists at {} \n{}'.format(duplicated_ckpt, config))
                checkpoint_dir = duplicated_ckpt[0]
            elif len(duplicated_ckpt) == 0:
                ckpt_name_exist = [os.path.basename(k).replace('model_', '') for k in ckpt_exist.keys()]
                ckpt_name_made = [os.path.basename(c).replace('model_', '') for c in checkpoints]
                model_ckpt = get_random_string(exclude=ckpt_name_exist + ckpt_name_made)
                checkpoint_dir = '{}/model_{}'.format(self.checkpoint_dir, model_ckpt)
            else:
                raise ValueError('duplicated checkpoints are found: \n {}'.format(duplicated_ckpt))

            if not os.path.exists('{}/epoch_{}'.format(checkpoint_dir, self.epoch_partial)):
                trainer = Trainer(checkpoint_dir=checkpoint_dir,
                                  disable_log=True,
                                  inherit_tner_checkpoint=self.inherit_tner_checkpoint,
                                  **config)
                trainer.train(
                    epoch_partial=self.epoch_partial, epoch_save=1, num_workers=num_workers, interval=interval)
            checkpoints.append(checkpoint_dir)

        path_to_metric_1st = '{}/metric.1st.json'.format(self.checkpoint_dir)
        metrics = {}
        for n, checkpoint_dir in enumerate(checkpoints):
            logging.info('## 1st RUN (EVAL): Configuration {}/{} ##'.format(n, len(checkpoints)))
            checkpoint_dir_model = '{}/epoch_{}'.format(checkpoint_dir, self.epoch_partial)
            try:

                metric = evaluate(
                    base_model=self.static_config['model'],
                    model=checkpoint_dir_model,
                    adapter=self.static_config['adapter'],
                    export_dir='{}/eval'.format(checkpoint_dir_model),
                    batch_size=self.batch_size_eval,
                    max_length=self.eval_config['max_length_eval'],
                    data=self.static_config['dataset'],
                    custom_dataset=self.static_config['custom_dataset'],
                    data_cache_prefix=data_cache_prefix,
                )
            except Exception:
                logging.exception('ERROR IN EVALUATION')
                continue
            metrics[checkpoint_dir_model] = metric['valid'][self.eval_config['metric']]

        metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        with open(path_to_metric_1st, 'w') as f:
            json.dump(metrics, f)

        logging.info('1st RUN RESULTS')
        for n, (k, v) in enumerate(metrics):
            logging.info('\t * rank: {} | metric: {} | model: {} |'.format(n, round(v, 3), k))

        if self.epoch_partial == self.epoch:
            logging.info('No 2nd phase as epoch_partial == epoch')
            return

        ###########
        # 2nd RUN #
        ###########

        metrics = metrics[:min(len(metrics), self.n_max_config)]
        checkpoints = []
        for n, (checkpoint_dir_model, _metric) in enumerate(metrics):
            logging.info('## 2nd RUN: Configuration {}/{}: {}'.format(n, len(metrics), _metric))
            model_ckpt = os.path.dirname(checkpoint_dir_model)
            if not os.path.exists('{}/epoch_{}'.format(model_ckpt, self.epoch)):
                trainer = Trainer(checkpoint_dir=model_ckpt,
                                  disable_log=True,
                                  inherit_tner_checkpoint=self.inherit_tner_checkpoint)
                trainer.train(epoch_save=1, num_workers=num_workers, interval=interval)

            checkpoints.append(model_ckpt)

        metrics = {}
        for n, checkpoint_dir in enumerate(checkpoints):
            logging.info('## 2nd RUN (EVAL): Configuration {}/{} ##'.format(n, len(checkpoints)))
            for checkpoint_dir_model in sorted(glob.glob('{}/epoch_*'.format(checkpoint_dir))):
                try:
                    metric = evaluate(
                        base_model=self.static_config['model'],
                        model=checkpoint_dir_model,
                        adapter=self.static_config['adapter'],
                        export_dir='{}/eval'.format(checkpoint_dir_model),
                        batch_size=self.batch_size_eval,
                        max_length=self.eval_config['max_length_eval'],
                        data=self.static_config['dataset'],
                        custom_dataset=self.static_config['custom_dataset'],
                        data_cache_prefix=data_cache_prefix
                    )
                except Exception:
                    logging.exception('ERROR IN EVALUATION')
                    continue

                metrics[checkpoint_dir_model] = metric['valid'][self.eval_config['metric']]

        metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        logging.info('2nd RUN RESULTS: \n{}'.format(str(metrics)))
        for n, (k, v) in enumerate(metrics):
            logging.info('\t * rank: {} | metric: {} | model: {} |'.format(n, round(v, 3), k))

        with open('{}/metric.2nd.json'.format(self.checkpoint_dir), 'w') as f:
            json.dump(metrics, f)

