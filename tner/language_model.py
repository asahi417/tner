"""
https://github.com/Adapter-Hub/adapter-transformers/blob/cea53a392068f56b260b8a51a1c5f05f130a9a2a/src/transformers/adapters/training.py#L6
"""
import os
import json
import logging
import pickle
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime, timedelta

import transformers
import torch

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from .tokenizer import TokenizerFixed, Dataset, PAD_TOKEN_LABEL_ID
from .text_searcher import WhooshSearcher, SentenceEmbedding

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message

__all__ = 'TransformersNER'


def pickle_save(obj, path: str):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    with open(path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


def euclidean_distance(a, b):
    return sum(map(lambda x: (x[0] - x[1])**2, zip(a, b))) ** 0.5


def cosine_similarity(a, b):
    norm_a = sum(map(lambda x: x * x, a)) ** 0.5
    norm_b = sum(map(lambda x: x * x, b)) ** 0.5
    return sum(map(lambda x: x[0] * x[1], zip(a, b)))/norm_a/norm_b


def load_hf(model_name, cache_dir, label2id = None, with_adapter_heads: bool = False):
    """ load huggingface checkpoints """
    logging.info('initialize language model with `{}`'.format(model_name))

    def _load_hf(local_files_only=False):
        if label2id is not None:
            config = transformers.AutoConfig.from_pretrained(
                model_name,
                num_labels=len(label2id),
                id2label={v: k for k, v in label2id.items()},
                label2id=label2id,
                cache_dir=cache_dir,
                local_files_only=local_files_only)
        else:
            config = transformers.AutoConfig.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only)

        if with_adapter_heads:
            model = transformers.AutoModelWithHeads.from_pretrained(
                model_name, config=config, cache_dir=cache_dir, local_files_only=local_files_only)
        else:
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                model_name, config=config, cache_dir=cache_dir, local_files_only=local_files_only)
        return model

    try:
        return _load_hf()
    except Exception:
        return _load_hf(True)


class TransformersNER:

    def __init__(self,
                 model: str,
                 max_length: int = 128,
                 crf: bool = False,
                 label2id: Dict = None,
                 cache_dir: str = None,
                 adapter: bool = False,
                 adapter_model: str = None,  # adapter model to load
                 adapter_task_name: str = 'ner',
                 adapter_non_linearity: str = None,
                 adapter_config: str = "pfeiffer",
                 adapter_reduction_factor: int = None,
                 adapter_language: str = 'en',
                 index_data_path: str = None,
                 index_prediction_path: str = None,
                 index_embedding_path: str = None):
        self.model_name = model
        self.max_length = max_length
        self.crf_layer = None

        # adapter configuration
        self.adapter = adapter
        self.adapter_task_name = adapter_task_name

        # load model
        model = load_hf(self.model_name, cache_dir, label2id)

        # load crf layer
        if 'crf_state_dict' in model.config.to_dict().keys() or crf:
            assert not self.adapter and adapter_model is None, 'CRF is not compatible with adapters'
            logging.info('use CRF')
            self.crf_layer = ConditionalRandomField(
                num_tags=len(model.config.id2label),
                constraints=allowed_transitions(constraint_type="BIO", labels=model.config.id2label)
            )
            if 'crf_state_dict' in model.config.to_dict().keys():
                logging.info('loading pre-trained CRF layer')
                self.crf_layer.load_state_dict(
                    {k: torch.FloatTensor(v) for k, v in model.config.crf_state_dict.items()}
                )

        if self.adapter or adapter_model is not None:
            logging.info('use Adapter')
            if adapter_model is not None:
                # load a pre-trained from Hub if specified
                logging.info('loading pre-trained Adapter from {}'.format(adapter_model))
                # initialize with AutoModelWithHeads to get label2id
                tmp_model = load_hf(self.model_name, cache_dir, with_adapter_heads=True)
                tmp_model.load_adapter(adapter_model, load_as=self.adapter_task_name, source='hf')
                label2id = tmp_model.config.prediction_heads['ner']['label2id']
                # with the label2id, initialize the AutoModelForTokenClassification
                model = load_hf(self.model_name, cache_dir, label2id)
                model.load_adapter(adapter_model, load_as=self.adapter_task_name, source='hf')
            else:
                # otherwise, add a fresh adapter
                adapter_config = transformers.AdapterConfig.load(
                    adapter_config,
                    adapter_reduction_factor=adapter_reduction_factor,
                    adapter_non_linearity=adapter_non_linearity,
                    language=adapter_language
                )
                model.add_adapter(self.adapter_task_name, config=adapter_config)

        self.model = model
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
            if self.crf_layer is not None:
                self.crf_layer = torch.nn.DataParallel(self.crf_layer)
        self.model.to(self.device)
        if self.crf_layer is not None:
            self.crf_layer.to(self.device)
        logging.info('{} GPUs are in use'.format(torch.cuda.device_count()))

        # load pre processor
        if self.crf_layer is not None:
            self.tokenizer = TokenizerFixed(
                self.model_name, cache_dir=cache_dir, id2label=self.id2label, padding_id=self.label2id['O']
            )
        else:
            self.tokenizer = TokenizerFixed(self.model_name, cache_dir=cache_dir, id2label=self.id2label)

        # setup text search engine
        self.sentence_embedding_model = None
        self.searcher = None
        self.searcher_prediction = None
        if index_data_path is not None:
            self.searcher = WhooshSearcher(index_path=index_data_path, embedding_path=index_embedding_path)
        if index_prediction_path is not None:
            assert self.searcher is not None
            with open(index_prediction_path) as f:
                self.searcher_prediction = json.load(f)

    def train(self):
        self.model.train()
        if self.adapter:
            if self.parallel:
                self.model.module.train_adapter(self.adapter_task_name)
            else:
                self.model.train_adapter(self.adapter_task_name)

    def eval(self):
        self.model.eval()
        if self.adapter:
            if self.parallel:
                self.model.module.set_active_adapters(self.adapter_task_name)
            else:
                self.model.set_active_adapters(self.adapter_task_name)

    def save(self, save_dir):

        def model_state(model):
            if self.parallel:
                return model.module
            return model

        if self.adapter:
            model_state(self.model).save_adapter(save_dir, self.adapter_task_name)
        else:
            if self.crf_layer is not None:
                model_state(self.model).config.update(
                    {'crf_state_dict': {k: v.tolist() for k, v in model_state(self.crf_layer).state_dict().items()}})
            model_state(self.model).save_pretrained(save_dir)
            self.tokenizer.tokenizer.save_pretrained(save_dir)

    def encode_to_loss(self, encode: Dict):
        assert 'labels' in encode
        encode = {k: v.to(self.device) for k, v in encode.items()}
        output = self.model(**encode)
        if self.crf_layer is not None:
            loss = - self.crf_layer(output['logits'], encode['labels'], encode['attention_mask'])
        else:
            loss = output['loss']
        return loss.mean() if self.parallel else loss

    def encode_to_prediction(self, encode: Dict):
        encode = {k: v.to(self.device) for k, v in encode.items()}
        output = self.model(**encode)
        prob = torch.softmax(output['logits'], dim=-1)
        prob, ind = torch.max(prob, dim=-1)
        prob = prob.cpu().detach().float().tolist()
        ind = ind.cpu().detach().int().tolist()

        if self.crf_layer is not None:
            if self.parallel:
                best_path = self.crf_layer.module.viterbi_tags(output['logits'])
            else:
                best_path = self.crf_layer.viterbi_tags(output['logits'])
            pred_results = []
            for tag_seq, _ in best_path:
                pred_results.append(tag_seq)
            ind = pred_results
        return ind, prob

    def get_data_loader(self,
                        inputs,
                        labels: List = None,
                        batch_size: int = None,
                        num_workers: int = 0,
                        shuffle: bool = False,
                        drop_last: bool = False,
                        mask_by_padding_token: bool = False,
                        cache_data_path: str = None):
        """ Transform features (produced by BERTClassifier.preprocess method) to data loader. """
        if cache_data_path is not None and os.path.exists(cache_data_path):
            logging.info('loading preprocessed feature from {}'.format(cache_data_path))
            out = pickle_load(cache_data_path)
        else:
            out = self.tokenizer.encode_plus_all(
                tokens=inputs,
                labels=labels,
                max_length=self.max_length,
                mask_by_padding_token=mask_by_padding_token)

            # remove overflow text
            logging.info('encode all the data: {}'.format(len(out)))

            # cache the encoded data
            if cache_data_path is not None:
                os.makedirs(os.path.dirname(cache_data_path), exist_ok=True)
                pickle_save(out, cache_data_path)
                logging.info('preprocessed feature is saved at {}'.format(cache_data_path))

        batch_size = len(out) if batch_size is None else batch_size
        return torch.utils.data.DataLoader(
            Dataset(out), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    def span_f1(self,
                inputs: List,
                labels: List,
                dates: List = None,
                datetime_format: str = '%Y-%m-%d',
                batch_size: int = None,
                num_workers: int = 0,
                cache_data_path: str = None,
                cache_prediction_path: str = None,
                cache_prediction_path_contextualisation: str = None,
                cache_embedding_path: str = None,
                span_detection_mode: bool = False,
                entity_list: bool = False,
                max_retrieval_size: int = 10,
                timeout: int = None,
                timedelta_hour_before: float = None,
                timedelta_hour_after: float = None,
                embedding_model: str = 'sentence-transformers/all-mpnet-base-v1',
                threshold_prob: float = 0.75,  # discard prediction under the value
                threshold_similarity: float = 0.75,  # discard prediction under the value
                retrieval_importance: float = 0.75,  # weight to the retrieved predictions
                ranking_type: str = 'similarity'  # ranking type 'similarity'/'es_score'/'frequency'
                ):
        pred_list, label_list = self.predict(
            inputs=inputs,
            labels=labels,
            dates=dates,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_prediction_path=cache_prediction_path,
            cache_prediction_path_contextualisation=cache_prediction_path_contextualisation,
            cache_embedding_path=cache_embedding_path,
            cache_data_path=cache_data_path,
            max_retrieval_size=max_retrieval_size,
            decode_bio=False,
            timeout=timeout,
            datetime_format=datetime_format,
            timedelta_hour_before=timedelta_hour_before,
            timedelta_hour_after=timedelta_hour_after,
            embedding_model=embedding_model,
            threshold_prob=threshold_prob,
            threshold_similarity=threshold_similarity,
            retrieval_importance=retrieval_importance,
            ranking_type=ranking_type
        )

        if span_detection_mode:

            def convert_to_binary_mask(entity_label):
                if entity_label == 'O':
                    return entity_label
                prefix = entity_label.split('-')[0]  # B or I
                return '{}-entity'.format(prefix)

            label_list = [[convert_to_binary_mask(_i) for _i in i] for i in label_list]
            pred_list = [[convert_to_binary_mask(_i) for _i in i] for i in pred_list]

        # compute metrics
        logging.info('\n{}'.format(classification_report(label_list, pred_list)))
        metric = {
            "micro/f1": f1_score(label_list, pred_list, average='micro'),
            "micro/recall": recall_score(label_list, pred_list, average='micro'),
            "micro/precision": precision_score(label_list, pred_list, average='micro'),
            "macro/f1": f1_score(label_list, pred_list, average='macro'),
            "macro/recall": recall_score(label_list, pred_list, average='macro'),
            "macro/precision": precision_score(label_list, pred_list, average='macro'),
        }
        if not span_detection_mode:
            metric["per_entity_metric"] = {}
            f1 = f1_score(label_list, pred_list, average=None)
            r = recall_score(label_list, pred_list, average=None)
            p = precision_score(label_list, pred_list, average=None)
            target_names = sorted([k.replace('B-', '') for k in self.label2id.keys() if k.startswith('B-')])
            for _name, _p, _r, _f1 in zip(target_names, p, r, f1):
                metric["per_entity_metric"][_name] = {
                    "precision": _p, "recall": _r, "f1": _f1
                }
            if entity_list:
                pred_decode = [self.decode_ner_tags(_p, _i) for _p, _i in zip(pred_list, inputs)]
                label_decode = [self.decode_ner_tags(_p, _i) for _p, _i in zip(label_list, inputs)]
                missing_entity = []
                non_entity = []
                type_error_entity = []

                for p, l in zip(pred_decode, label_decode):
                    l_entity_dict = {' '.join(_i['entity']): _i['type'] for _i in l}
                    p_entity_dict = {' '.join(_i['entity']): _i['type'] for _i in p}
                    missing_entity += [_i for _i in l_entity_dict if _i not in p_entity_dict]
                    non_entity += [_i for _i in p_entity_dict if _i not in l_entity_dict]
                    type_error_entity += [
                        _i for _i in l_entity_dict if _i in p_entity_dict and l_entity_dict[_i] != p_entity_dict[_i]]
                metric['entity_list'] = {
                    'missing_entity': sorted(list(set(missing_entity))),
                    'non_entity_prediction': sorted(list(set(non_entity))),
                    'type_error': sorted(list(set(type_error_entity)))
                }
        return metric

    def predict(self,
                inputs: List,
                labels: List = None,
                dates: List = None,
                batch_size: int = None,
                num_workers: int = 0,
                cache_prediction_path: str = None,
                cache_prediction_path_contextualisation: str = None,
                cache_embedding_path: str = None,
                cache_data_path: str = None,
                max_retrieval_size: int = 10,
                decode_bio: bool = False,
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

        if dates is not None:
            assert len(dates) == len(inputs)
            if datetime_format is None:
                assert all(type(i) == datetime for i in dates)
            else:
                dates = [datetime.strptime(i, datetime_format) for i in dates]
        else:
            dates = [None] * len(inputs)

        if self.searcher is None:
            return self.base_predict(
                inputs, labels, batch_size, num_workers, cache_data_path, cache_prediction_path, decode_bio=decode_bio)

        out = self.base_predict(
            inputs, labels, batch_size, num_workers, cache_data_path, cache_prediction_path, decode_bio=True)
        pred_list, pred_decode = out[0]
        dummy_label = False
        if len(out) > 1:
            label_list, label_decode = out[1]
        else:
            dummy_label = True
            label_list = label_decode = [None] * len(pred_list)

        # get sentence embeddings
        if cache_embedding_path is not None and os.path.exists(cache_embedding_path):
            with open(cache_embedding_path) as f:
                embeddings = [[float(ii) for ii in i.split(',')] for i in f.read().split('\n') if len(i) > 0]
        else:
            if self.sentence_embedding_model is None or self.sentence_embedding_model.model != embedding_model:
                self.sentence_embedding_model = SentenceEmbedding(embedding_model)
            embeddings = self.sentence_embedding_model.embed(inputs).tolist()
            if cache_embedding_path is not None:
                with open(cache_embedding_path, 'w') as f:
                    for e in embeddings:
                        f.write(','.join([str(i) for i in e]) + '\n')
        assert len(embeddings) == len(inputs), str([len(embeddings), len(inputs)])

        # contextualization
        if cache_prediction_path_contextualisation is not None \
                and os.path.exists(cache_prediction_path_contextualisation):
            with open(cache_prediction_path_contextualisation) as f:
                tmp = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
                new_pred_list = [i['prediction'] for i in tmp]
        else:
            new_pred_list = []
            for pred_list_sent, pred_decode_sent, input_sent, embedding_sent, label_sent, date_sent in \
                    tqdm(list(zip(pred_list, pred_decode, inputs, embeddings, label_list, dates))):

                date_range_start = None
                date_range_end = None
                if date_sent is not None:
                    if timedelta_hour_before is not None:
                        date_range_start = date_sent - timedelta(hours=timedelta_hour_before)
                    if timedelta_hour_after is not None:
                        date_range_end = date_sent + timedelta(hours=timedelta_hour_after)
                    logging.info('date range: {} -- {}'.format(date_range_start, date_range_end))

                tmp_new_pred_list = self.contextualisation(
                    pred_list_sent=pred_list_sent,
                    input_sent=input_sent,
                    embedding_sent=embedding_sent,
                    pred_decode_sent=pred_decode_sent,
                    max_retrieval_size=max_retrieval_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    timeout=timeout,
                    date_range_start=date_range_start,
                    date_range_end=date_range_end,
                    threshold_prob=threshold_prob,
                    threshold_similarity=threshold_similarity,
                    retrieval_importance=retrieval_importance,
                    ranking_type=ranking_type)
                if tmp_new_pred_list is None:
                    new_pred_list.append(pred_list_sent)
                else:
                    new_pred_list.append(tmp_new_pred_list)

                    # For DEBUGGING
                    if tmp_new_pred_list != pred_list_sent:
                        logging.info('Label:{}'.format(self.decode_ner_tags(label_sent, input_sent)))
                        logging.info('Old:'.format(self.decode_ner_tags(pred_list_sent, input_sent)))
                        logging.info('New:'.format(self.decode_ner_tags(tmp_new_pred_list, input_sent)))

                if cache_prediction_path_contextualisation is not None:
                    os.makedirs(os.path.dirname(cache_prediction_path_contextualisation), exist_ok=True)
                    with open(cache_prediction_path_contextualisation, 'w') as f:
                        f.write(json.dumps({'prediction': new_pred_list}) + '\n')

        if not decode_bio:
            if dummy_label:
                return pred_list,
            return pred_list, label_list
        new_pred_decode = [self.decode_ner_tags(_p, _i) for _p, _i in zip(new_pred_list, inputs)]
        if dummy_label:
            return (new_pred_list, new_pred_decode),
        return (new_pred_list, new_pred_decode), (label_list, label_decode)

    def contextualisation(self,
                          pred_list_sent,
                          input_sent,
                          embedding_sent,
                          pred_decode_sent,
                          max_retrieval_size,
                          batch_size,
                          num_workers,
                          timeout: int,
                          date_range_start,
                          date_range_end,
                          threshold_prob,
                          threshold_similarity,
                          retrieval_importance,
                          ranking_type: str):

        def _get_context(query, probability, entity_type):
            """ get context given a query """
            # query documents
            retrieved_text = self.searcher.search(query, limit=max_retrieval_size, return_field=['text', 'id'],
                                                  timeout=timeout, date_range_start=date_range_start,
                                                  date_range_end=date_range_end)
            logging.info('query: {}, retrieved text: {}'.format(query, len(retrieved_text)))
            if len(retrieved_text) == 0:
                return None
            # get prediction on the retrieved text
            tmp_pred = []
            tmp_score = []  # es score
            tmp_embedding = []
            if self.searcher_prediction is None:
                to_run_prediction = [i['text'].split(' ') for i in retrieved_text]
                tmp_score = [i['score'] for i in retrieved_text]
                tmp_embedding = [i['embedding'] for i in retrieved_text]
                to_run_prediction_score = []
                to_run_prediction_embedding = []
            else:
                to_run_prediction = []
                to_run_prediction_embedding = []
                to_run_prediction_score = []
                for i in retrieved_text:
                    try:
                        tmp_pred.append(self.searcher_prediction[str(i['id'])])
                        tmp_score.append(i['score'])
                        tmp_embedding.append(i['embedding'])
                    except KeyError:
                        to_run_prediction.append(i['text'].split(' '))
                        to_run_prediction_score.append(i['score'])
                        to_run_prediction_embedding.append(i['embedding'])

            if len(to_run_prediction) > 0:
                logging.info('run prediction over {} docs'.format(len(to_run_prediction)))
                out = self.base_predict(
                    to_run_prediction, None, batch_size, num_workers, decode_bio=True)[0]
                _, tmp_decode = out[0]
                tmp_pred += tmp_decode
                tmp_score += to_run_prediction_score
                tmp_embedding += to_run_prediction_embedding

            # formatting the result
            _out = {}
            assert len(tmp_pred) == len(tmp_score) == len(tmp_embedding), str([len(tmp_pred), len(tmp_score), len(tmp_embedding)])
            for _pred, _score, _e in zip(tmp_pred, tmp_score, tmp_embedding):
                for __p in _pred:
                    print(__p.keys())
                    _probability = sum(__p['probability']) / len(__p['probability'])

                    print('prob', _probability, threshold_prob)
                    if _probability < threshold_prob:
                        continue

                    sim = cosine_similarity(embedding_sent, __p['embedding'])
                    print('sim', sim, threshold_similarity)
                    if sim < threshold_similarity:
                        continue


                    _key = ' '.join(__p['entity'])
                    if _key not in _out:
                        _out[_key] = {}
                    if __p['type'] not in _out[_key]:
                        _out[_key][__p['type']] = {
                            'count': 1,
                            'frequency': 1 * retrieval_importance,
                            'score': _score * retrieval_importance,
                            'similarity': sim * retrieval_importance,
                            'probability': _probability * retrieval_importance
                        }
                    else:
                        _out[_key][__p['type']]['count'] += 1
                        _out[_key][__p['type']]['frequency'] += 1 * retrieval_importance
                        _out[_key][__p['type']]['score'] += _score * retrieval_importance
                        _out[_key][__p['type']]['similarity'] += sim * retrieval_importance
                        _out[_key][__p['type']]['probability'] += _prob * retrieval_importance
            if probability > threshold_prob:
                max_score = max(tmp_score)
                if query not in _out:
                    _out[query] = {}
                if entity_type not in _out[query]:
                    _out[query][entity_type] = {
                        'count': 1,
                        'frequency': 1,
                        'score': max_score,
                        'similarity': cosine_similarity(embedding_sent, embedding_sent),
                        'probability': probability
                    }
                else:
                    _out[query][entity_type]['count'] += 1
                    _out[query][entity_type]['frequency'] += 1
                    _out[query][entity_type]['score'] += max_score
                    _out[query][entity_type]['similarity'] += cosine_similarity(embedding_sent, embedding_sent)
                    _out[query][entity_type]['probability'] += probability
            # aggregation
            _result = {}
            for k, v in _out.items():
                if ranking_type == 'similarity':
                    ranking_score = {k: v['similarity'] / v['count'] for k, v in v.items()}
                elif ranking_type == 'es_score':
                    ranking_score = {k: v['score']/v['count'] for k, v in v.items()}
                elif ranking_type == 'frequency':
                    ranking_score = {k: v['frequency'] for k, v in v.items()}
                elif ranking_type == 'probability':
                    ranking_score = {k: v['probability']/v['count'] for k, v in v.items()}
                else:
                    raise ValueError('unknown type: {}'.format(ranking_type))

                max_score = max(ranking_score.values())
                ranking_score = [(k, v) for k, v in ranking_score.items() if v == max_score]
                if len(ranking_score) == 1:
                    logging.warning('multiple candidate at ranking: {}'.format(ranking_score))
                new_entity_type, score = ranking_score[0]
                _out[k] = new_entity_type
            return _out

        context = {}  # {'Gaza': [{'score': 12.369498962574873, 'type': 'location'}]}
        for _dict in pred_decode_sent:
            _query = ' '.join(_dict['entity'])
            _prob = sum(_dict['probability'])/len(_dict['probability'])
            _context = _get_context(_query, _prob, _dict['type'])
            if _context is not None and _query in _context:
                context[_query] = _context[_query]
        if len(context) == 0:
            return None

        new_pred_list_sent = self.convert_tags_to_bio(pred_list_sent, input_sent, custom_dict=context)
        return new_pred_list_sent

    def base_predict(self,
                     inputs: List,
                     labels: List = None,
                     batch_size: int = None,
                     num_workers: int = 0,
                     cache_data_path: str = None,
                     cache_prediction_path: str = None,
                     decode_bio: bool = False):

        dummy_label = False
        if labels is None:
            labels = [[0] * len(i) for i in inputs]
            dummy_label = True
        if cache_prediction_path is not None and os.path.exists(cache_prediction_path):
            with open(cache_prediction_path) as f:
                tmp = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
                pred_list = [i['prediction'] for i in tmp]
                prob_list = [i['probability'] for i in tmp]
            label_list = [[self.id2label[__l] for __l in _l] for _l in labels]
            inputs_list = inputs
        else:
            self.model.eval()
            loader = self.get_data_loader(
                inputs,
                labels=labels,
                batch_size=batch_size,
                num_workers=num_workers,
                mask_by_padding_token=True,
                cache_data_path=cache_data_path)
            label_list = []
            pred_list = []
            prob_list = []
            ind = 0

            inputs_list = []
            for i in loader:
                label = i.pop('labels').cpu().tolist()
                pred, prob = self.encode_to_prediction(i)
                assert len(label) == len(pred) == len(prob), str([len(label), len(pred), len(prob)])
                input_ids = i.pop('input_ids').cpu().tolist()
                for _i, _p, _prob, _l in zip(input_ids, pred, prob, label):
                    assert len(_i) == len(_p) == len(_l)
                    tmp = [(__p, __l, __prob) for __p, __l, __prob in zip(_p, _l, _prob) if __l != PAD_TOKEN_LABEL_ID]
                    tmp_pred = list(list(zip(*tmp))[0])
                    tmp_label = list(list(zip(*tmp))[1])
                    tmp_prob = list(list(zip(*tmp))[2])
                    if len(tmp_label) != len(labels[ind]):
                        if len(tmp_label) < len(labels[ind]):
                            logging.info('found sequence possibly more than max_length')
                            logging.info('{}: \n\t - model loader: {}\n\t - label: {}'.format(ind, tmp_label, labels[ind]))
                            tmp_pred = tmp_pred + [self.label2id['O']] * (len(labels[ind]) - len(tmp_label))
                            tmp_prob = tmp_prob + [0.0] * (len(labels[ind]) - len(tmp_label))
                        else:
                            raise ValueError(
                                '{}: \n\t - model loader: {}\n\t - label: {}'.format(ind, tmp_label, labels[ind]))
                    assert len(tmp_pred) == len(labels[ind])
                    pred_list.append(tmp_pred)
                    label_list.append(labels[ind])
                    inputs_list.append(inputs[ind])
                    prob_list.append(tmp_prob)
                    ind += 1
            label_list = [[self.id2label[__l] for __l in _l] for _l in label_list]
            pred_list = [[self.id2label[__p] for __p in _p] for _p in pred_list]
            if cache_prediction_path is not None:
                os.makedirs(os.path.dirname(cache_prediction_path), exist_ok=True)
                with open(cache_prediction_path, 'w') as f:
                    for _pred, _prob in zip(pred_list, prob_list):
                        f.write(json.dumps({'prediction': _pred, 'probability': _prob}) + '\n')

        if not decode_bio:
            if dummy_label:
                return pred_list,
            return pred_list, label_list
        pred_decode = [self.decode_ner_tags(_p, _i, _prob) for _p, _prob, _i in zip(pred_list, prob_list, inputs_list)]
        label_decode = [self.decode_ner_tags(_p, _i) for _p, _i in zip(label_list, inputs_list)]
        if dummy_label:
            return (pred_list, pred_decode),
        return (pred_list, pred_decode), (label_list, label_decode)

    @staticmethod
    def convert_tags_to_bio(tag_sequence, input_sequence, custom_dict: Dict = None):

        def update_collection(_tmp_entity, _tmp_type, _new_tag_sequence):
            if len(_tmp_entity) > 0:
                assert _tmp_type is not None
                if custom_dict is not None and ' '.join(_tmp_entity) in custom_dict:
                    _tmp_type = custom_dict[' '.join(_tmp_entity)]
                _new_tag_sequence += ['B-{}'.format(_tmp_type)] + ['I-{}'.format(_tmp_type)] * (len(_tmp_entity) - 1)
                _tmp_entity = []
                _tmp_type = None

            return _tmp_entity, _tmp_type, _new_tag_sequence

        tmp_entity = []
        tmp_entity_type = None
        new_tag_sequence = []
        for _l, _i in zip(tag_sequence, input_sequence):
            if _l.startswith('B-'):
                tmp_entity, tmp_entity_type, new_tag_sequence = update_collection(
                    tmp_entity, tmp_entity_type, new_tag_sequence)
                tmp_entity = [_i]
                tmp_entity_type = '-'.join(_l.split('-')[1:])
            elif _l.startswith('I-'):
                tmp_tmp_entity_type = '-'.join(_l.split('-')[1:])
                if len(tmp_entity) == 0:
                    # if 'I' not start with 'B', skip it
                    new_tag_sequence.append('O')
                elif tmp_tmp_entity_type != tmp_entity_type:
                    # if the type does not match with the B, skip
                    tmp_entity, tmp_entity_type, new_tag_sequence = update_collection(
                        tmp_entity, tmp_entity_type, new_tag_sequence)
                    new_tag_sequence.append('O')
                else:
                    tmp_entity.append(_i)
            elif _l == 'O':
                tmp_entity, tmp_entity_type, new_tag_sequence = update_collection(
                    tmp_entity, tmp_entity_type, new_tag_sequence)
                new_tag_sequence.append('O')
            else:
                raise ValueError('unknown tag: {}'.format(_l))
        _, _, new_tag_sequence = update_collection(tmp_entity, tmp_entity_type, new_tag_sequence)
        assert len(new_tag_sequence) == len(tag_sequence), '\nnew: {}\nold: {}'.format(new_tag_sequence, tag_sequence)
        return new_tag_sequence

    @staticmethod
    def decode_ner_tags(tag_sequence, input_sequence, probability_sequence=None, custom_dict: Dict = None):
        # print(tag_sequence)
        # print(input_sequence)
        # print(probability_sequence)

        def update_collection(_tmp_entity, _tmp_entity_type, _tmp_prob, _out):
            if len(_tmp_entity) != 0 and _tmp_entity_type is not None:
                if custom_dict is not None and ' '.join(_tmp_entity) in custom_dict:
                    _tmp_entity_type = custom_dict[' '.join(_tmp_entity)]
                if _tmp_prob is None:
                    _out.append({'type': _tmp_entity_type, 'entity': _tmp_entity})
                else:
                    _out.append({'type': _tmp_entity_type, 'entity': _tmp_entity, 'probability': _tmp_prob})
                _tmp_entity = []
                _tmp_prob = []
                _tmp_entity_type = None
            return _tmp_entity, _tmp_entity_type, _tmp_prob, _out

        probability_sequence = [None] * len(tag_sequence) if probability_sequence is None else probability_sequence
        assert len(tag_sequence) == len(input_sequence) == len(probability_sequence), str([len(tag_sequence), len(input_sequence), len(probability_sequence)])
        out = []
        tmp_entity = []
        tmp_prob = []
        tmp_entity_type = None
        for _l, _i, _prob in zip(tag_sequence, input_sequence, probability_sequence):
            if _l.startswith('B-'):
                _, _, _, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, out)
                tmp_entity_type = '-'.join(_l.split('-')[1:])
                tmp_entity = [_i]
                tmp_prob = [_prob]
            elif _l.startswith('I-'):
                tmp_tmp_entity_type = '-'.join(_l.split('-')[1:])
                if len(tmp_entity) == 0:
                    # if 'I' not start with 'B', skip it
                    tmp_entity, tmp_entity_type, tmp_prob, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, out)
                elif tmp_tmp_entity_type != tmp_entity_type:
                    # if the type does not match with the B, skip
                    tmp_entity, tmp_entity_type, tmp_prob, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, out)
                else:
                    tmp_entity.append(_i)
                    tmp_prob.append(_prob)
            elif _l == 'O':
                tmp_entity, tmp_entity_type, tmp_prob, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, out)

            else:
                raise ValueError('unknown tag: {}'.format(_l))
        _, _, _, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, out)
        return out

