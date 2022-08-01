""" Transformer NER model """
import json
import logging
import os
from typing import List, Dict
from tqdm import tqdm
from packaging.version import parse

import torch
from transformers import AutoConfig, AutoModelForTokenClassification

# For CRF Layer
from allennlp import __version__
from allennlp.modules import ConditionalRandomField
if parse("2.10.0") < parse(__version__):
    from allennlp.modules.conditional_random_field import allowed_transitions
else:
    from allennlp.modules.conditional_random_field.conditional_random_field import allowed_transitions

from .util import pickle_save, pickle_load, span_f1, decode_ner_tags
from .get_dataset import get_dataset
from .ner_tokenizer import NERTokenizer


PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset wrapper converting into tensor """
    float_tensors = ['attention_mask', 'input_feature']

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


class TransformersNER:

    def __init__(self, model: str, max_length: int = 128, crf: bool = False, use_auth_token: bool = False):
        self.model_name = model
        self.max_length = max_length
        self.crf_layer = None
        # load model
        logging.info('initialize language model with `{}`'.format(model))
        try:
            self.model = self.load_hf(use_auth_token)
        except Exception:
            self.model = self.load_hf(use_auth_token, True)

        # load crf layer
        if 'crf_state_dict' in self.model.config.to_dict().keys() or crf:
            logging.info('use CRF')
            self.crf_layer = ConditionalRandomField(
                num_tags=len(self.model.config.id2label),
                constraints=allowed_transitions(constraint_type="BIO", labels=self.model.config.id2label)
            )
            if 'crf_state_dict' in self.model.config.to_dict().keys():
                logging.info('loading pre-trained CRF layer')
                self.crf_layer.load_state_dict(
                    {k: torch.FloatTensor(v) for k, v in self.model.config.crf_state_dict.items()}
                )
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)
            if self.crf_layer is not None:
                self.crf_layer = torch.nn.DataParallel(self.crf_layer)
        self.model.to(self.device)
        if self.crf_layer is not None:
            self.crf_layer.to(self.device)
        logging.info(f'{torch.cuda.device_count()} GPUs are in use')

        # load pre processor
        if self.crf_layer is not None:
            self.tokenizer = NERTokenizer(self.model_name, id2label=self.id2label, padding_id=self.label2id['O'], use_auth_token=use_auth_token)
        else:
            self.tokenizer = NERTokenizer(self.model_name, id2label=self.id2label, use_auth_token=use_auth_token)

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
                        inputs,  # list of tokenized sentences
                        labels: List = None,
                        batch_size: int = None,
                        shuffle: bool = False,
                        drop_last: bool = False,
                        mask_by_padding_token: bool = False,
                        cache_file_feature: str = None,
                        return_loader: bool = True):
        """ Transform features (produced by BERTClassifier.preprocess method) to data loader. """
        if cache_file_feature is not None and os.path.exists(cache_file_feature):
            logging.info(f'loading preprocessed feature from {cache_file_feature}')
            out = pickle_load(cache_file_feature)
        else:
            out = self.tokenizer.encode_plus_all(
                tokens=inputs, labels=labels, max_length=self.max_length, mask_by_padding_token=mask_by_padding_token)

            # remove overflow text
            logging.info(f'encode all the data: {len(out)}')

            # cache the encoded data
            if cache_file_feature is not None:
                os.makedirs(os.path.dirname(cache_file_feature), exist_ok=True)
                pickle_save(out, cache_file_feature)
                logging.info(f'preprocessed feature is saved at {cache_file_feature}')
        if return_loader:
            return torch.utils.data.DataLoader(
                Dataset(out), batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_last)

        return list(Dataset(out))

    def span_f1(self,
                inputs: List,
                labels: List,
                batch_size: int = None,
                cache_file_feature: str = None,
                cache_file_prediction: str = None,
                span_detection_mode: bool = False,
                return_ci: bool = False):
        output = self.predict(
            inputs=inputs,
            labels=labels,
            batch_size=batch_size,
            cache_file_prediction=cache_file_prediction,
            cache_file_feature=cache_file_feature,
            return_loader=True
        )
        return span_f1(output['prediction'], output['label'], self.label2id, span_detection_mode, return_ci=return_ci)

    def predict(self,
                inputs: List,
                labels: List = None,
                batch_size: int = None,
                cache_file_feature: str = None,
                cache_file_prediction: str = None,
                return_loader: bool = False):
        # split by halfspace if its string
        inputs = [i.split(' ') if type(i) is not list else i for i in inputs]
        dummy_label = False
        if labels is None:
            labels = [[0] * len(i) for i in inputs]
            dummy_label = True
        if cache_file_prediction is not None and os.path.exists(cache_file_prediction):
            with open(cache_file_prediction) as f:
                tmp = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
                pred_list = [i['prediction'] for i in tmp]
                prob_list = [i['probability'] for i in tmp]
            label_list = [[self.id2label[__l] for __l in _l] for _l in labels]
            inputs_list = inputs
        else:
            self.model.eval()
            loader = self.get_data_loader(inputs,
                                          labels=labels,
                                          batch_size=batch_size,
                                          mask_by_padding_token=True,
                                          cache_file_feature=cache_file_feature,
                                          return_loader=return_loader)
            label_list = []
            pred_list = []
            prob_list = []
            ind = 0

            inputs_list = []
            for i in tqdm(loader):
                if not return_loader:
                    i = {k: torch.unsqueeze(v, 0) for k, v in i.items()}
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
                            logging.debug('found sequence possibly more than max_length')
                            logging.debug(f'{ind}: \n\t - model loader: {tmp_label}\n\t - label: {labels[ind]}')
                            tmp_pred = tmp_pred + [self.label2id['O']] * (len(labels[ind]) - len(tmp_label))
                            tmp_prob = tmp_prob + [0.0] * (len(labels[ind]) - len(tmp_label))
                        else:
                            raise ValueError(f'{ind}: \n\t - model loader: {tmp_label}\n\t - label: {labels[ind]}')
                    assert len(tmp_pred) == len(labels[ind])
                    assert len(inputs[ind]) == len(tmp_pred)
                    pred_list.append(tmp_pred)
                    label_list.append(labels[ind])
                    inputs_list.append(inputs[ind])
                    prob_list.append(tmp_prob)
                    ind += 1
            label_list = [[self.id2label[__l] for __l in _l] for _l in label_list]
            pred_list = [[self.id2label[__p] for __p in _p] for _p in pred_list]
            if cache_file_prediction is not None:
                os.makedirs(os.path.dirname(cache_file_prediction), exist_ok=True)
                with open(cache_file_prediction, 'w') as f:
                    for _pred, _prob in zip(pred_list, prob_list):
                        f.write(json.dumps({'prediction': _pred, 'probability': _prob}) + '\n')

        output = {'prediction': pred_list,
                  'probability': prob_list,
                  'input': inputs_list,
                  'entity_prediction': [decode_ner_tags(_p, _i, _prob) for _p, _prob, _i in
                                        zip(pred_list, prob_list, inputs_list)]}
        if not dummy_label:
            output['label'] = label_list
            output['entity_label'] = [decode_ner_tags(_p, _i) for _p, _i in zip(label_list, inputs_list)]
        return output

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, save_dir):

        def model_state(model):
            if self.parallel:
                return model.module
            return model

        if self.crf_layer is not None:
            model_state(self.model).config.update(
                {'crf_state_dict': {k: v.tolist() for k, v in model_state(self.crf_layer).state_dict().items()}})
        model_state(self.model).save_pretrained(save_dir)
        self.tokenizer.tokenizer.save_pretrained(save_dir)

    def load_hf(self, use_auth_token, local_files_only=False):
        config = AutoConfig.from_pretrained(
            self.model_name,
            use_auth_token=use_auth_token,
            num_labels=len(label_to_id),
            id2label={v: k for k, v in label_to_id.items()},
            label2id=label_to_id,
            local_files_only=local_files_only)
        return AutoModelForTokenClassification.from_pretrained(
            self.model_name, config=config, local_files_only=local_files_only)

    def evaluate(self,
                 batch_size,
                 data_split,
                 cache_file_feature: str = None,
                 cache_file_prediction: str = None,
                 span_detection_mode: bool = False,
                 return_ci: bool = False):
        self.eval()
        data = get_dataset(data_split)
        return self.span_f1(
            inputs=data['data'],
            labels=data['label'],
            batch_size=batch_size,
            cache_file_feature=cache_file_feature,
            cache_file_prediction=cache_file_prediction,
            span_detection_mode=span_detection_mode,
            return_ci=return_ci)
