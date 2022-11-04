""" Transformer NER model """
import json
import logging
import os
from typing import List, Dict
from tqdm import tqdm
from packaging.version import parse

import torch

# For CRF Layer
from allennlp import __version__
from allennlp.modules import ConditionalRandomField
if parse("2.10.0") > parse(__version__):
    from allennlp.modules.conditional_random_field import allowed_transitions
else:
    from allennlp.modules.conditional_random_field.conditional_random_field import allowed_transitions

from .get_dataset import get_dataset
from .util import pickle_save, pickle_load, span_f1, decode_ner_tags, Dataset, load_hf
from .ner_tokenizer import NERTokenizer


PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index


class TransformersNER:
    """ TransformersNER """

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __init__(self,
                 model: str,
                 max_length: int = 128,
                 crf: bool = False,
                 use_auth_token: bool = False,
                 label2id: Dict = None,
                 non_entity_symbol: str = 'O'):
        """ TransformersNER

        @param model: the huggingface model (`tner/roberta-large-tweetner-2021`) or path to local checkpoint
        @param max_length: [optional] max length of language model input
        @param crf: [optional] to use CRF or not (trained model should follow the model config)
        @param use_auth_token: [optional] Huggingface transformers argument of `use_auth_token`
        @param label2id: [optional] label2id dictionary, which is not needed for already trained NER model,
         but need for fine-tuning model on NER
        @param non_entity_symbol: [optional] label for non-entity ('O' as default)
        """
        self.model_name = model
        self.max_length = max_length
        self.crf_layer = None
        self.non_entity_symbol = non_entity_symbol
        # load model
        logging.info(f'initialize language model with `{model}`')
        try:
            self.model = load_hf(self.model_name, label2id, use_auth_token)
        except Exception:
            self.model = load_hf(self.model_name, label2id, use_auth_token, True)
        self.is_xlnet = self.model.config.model_type == 'xlnet'
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
        logging.info(f'label2id: {self.label2id}')
        assert 'O' in self.label2id, f'invalid label2id {self.label2id}'

        # GPU setup https://github.com/asahi417/tner/issues/33
        try:
            # Mac M1 Support https://github.com/asahi417/tner/issues/30
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        except Exception:
            self.device = 'cpu'
        if self.device == 'cpu':
            self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1

        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)
            if self.crf_layer is not None:
                self.crf_layer = torch.nn.DataParallel(self.crf_layer)
        self.model.to(self.device)
        if self.crf_layer is not None:
            self.crf_layer.to(self.device)
        logging.info(f'device   : {self.device}')
        logging.info(f'gpus     : {torch.cuda.device_count()}')

        # load pre processor
        if self.crf_layer is not None:
            self.tokenizer = NERTokenizer(
                self.model_name,
                id2label=self.id2label,
                padding_id=self.label2id[self.non_entity_symbol],
                use_auth_token=use_auth_token)
        else:
            self.tokenizer = NERTokenizer(self.model_name, id2label=self.id2label, use_auth_token=use_auth_token)

    def encode_to_loss(self, encode: Dict):
        """ map encoded feature to loss value for model fine-tuning

        @param encode: dictionary of output from `encode_plus` module of tokenizer
        @return: tensor of loss value
        """
        assert 'labels' in encode
        encode = {k: v.to(self.device) for k, v in encode.items()}
        output = self.model(**encode)
        if self.crf_layer is not None:
            loss = - self.crf_layer(output['logits'], encode['labels'], encode['attention_mask'])
        else:
            loss = output['loss']
        return loss.mean() if self.parallel else loss

    def encode_to_prediction(self, encode: Dict):
        """ map encoded feature to model prediction

        @param encode: dictionary of output from `encode_plus` module of tokenizer
        @return: (ind, prob) where `ind` is a predicted tag_id sequence and `prob` is the associated probability
        """
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
                        shuffle: bool = False,
                        drop_last: bool = False,
                        mask_by_padding_token: bool = False,
                        cache_file_feature: str = None,
                        separator: str = ' ',
                        max_length: int = None):
        """ get data loader (`torch.utils.data.DataLoader`) for model output

        @param inputs: a list of tokenized sentences ([["I", "live",...], ["You", "live", ...]])
        @param labels: [optional] a list of label sequences
        @param batch_size: [optional] batch size
        @param shuffle: [optional] shuffle instances in the loader
        @param drop_last: [optional] drop remanining batch
        @param [optional] mask_by_padding_token: Padding sequence has two cases:
            (i) Padding upto max_length: if True, padding such tokens by {PADDING_TOKEN}, else by "O"
            (ii) Intermediate sub-token: For example, we have tokens in a sentence ["New", "York"] with labels
                ["B-LOC", "I-LOC"], which language model tokenizes into ["New", "Yor", "k"]. If mask_by_padding_token
                is True, the new label is ["B-LOC", "I-LOC", {PADDING_TOKEN}], otherwise ["B-LOC", "I-LOC", "I-LOC"].
        @param cache_file_feature: [optional] save & load precompute data loader
        @param separator: [optional] token separator (eg. '' for Japanese and Chinese)
        @param max_length: [optional] max length of language model input
        @return: `torch.utils.data.DataLoader` object or list if `return_list = True`
        """
        if cache_file_feature is not None and os.path.exists(cache_file_feature):
            logging.info(f'loading preprocessed feature from {cache_file_feature}')
            out = pickle_load(cache_file_feature)
        else:
            out = self.tokenizer.encode_plus_all(
                tokens=inputs,
                labels=labels,
                max_length=self.max_length if max_length is None else max_length,
                mask_by_padding_token=mask_by_padding_token,
                separator=separator
            )
            # remove overflow text
            logging.info(f'encode all the data: {len(out)}')

            # cache the encoded data
            if cache_file_feature is not None:
                os.makedirs(os.path.dirname(cache_file_feature), exist_ok=True)
                pickle_save(out, cache_file_feature)
                logging.info(f'preprocessed feature is saved at {cache_file_feature}')
        batch_size = batch_size if batch_size is not None else len(out)
        return torch.utils.data.DataLoader(
            Dataset(out), batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_last)

    def predict(self,
                inputs: List,
                labels: List = None,
                batch_size: int = None,
                cache_file_feature: str = None,
                cache_file_prediction: str = None,
                max_length: int = None,
                separator: str = ' '):
        """ get model prediction

        @param inputs: a list of tokenized sentences ([["I", "live",...], ["You", "live", ...]])
        @param labels: [optional] a list of label sequences
        @param batch_size: [optional] batch size
        @param cache_file_feature: [optional] save & load precompute data loader
        @param cache_file_prediction: [optional] save & load precompute model predicti
        @param max_length: [optional] max length of language model inputon
        @param separator: [optional] token separator (eg. '' for Japanese and Chinese)
        @return: a dictionary containing
            {'prediction': a sequence of predictions for each input,
             'probability': a sequence of probability for each input,
             'input': a list of input (tokenized if it's not),
             'entity_prediction': a list of entities for each input}
        """
        # split by a half-space if its string
        assert all(type(i) in [list, str] for i in inputs), f'invalid input type {inputs}'
        inputs = [i.split(' ') if type(i) is str else i for i in inputs]
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
            loader = self.get_data_loader(
                inputs,
                labels=labels,
                batch_size=batch_size,
                mask_by_padding_token=True,
                cache_file_feature=cache_file_feature,
                max_length=max_length,
                separator=separator
            )
            label_list = []
            pred_list = []
            prob_list = []
            ind = 0

            inputs_list = []
            for i in tqdm(loader):
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
                            tmp_pred = tmp_pred + [self.label2id[self.non_entity_symbol]] * (len(labels[ind]) - len(tmp_label))
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

    def evaluate(self,
                 dataset: List or str = None,
                 dataset_name: List or str = None,
                 local_dataset: List or Dict = None,
                 batch_size: int = None,
                 dataset_split: str = 'test',
                 cache_dir: str = None,
                 cache_file_feature: str = None,
                 cache_file_prediction: str = None,
                 span_detection_mode: bool = False,
                 return_ci: bool = False,
                 unification_by_shared_label: bool = True,
                 separator: str = ' '):
        """ evaluate model on the dataset

        @param dataset: dataset name (or a list of it) on huggingface tner organization (https://huggingface.co/datasets?search=tner)
            (eg. "tner/conll2003", ["tner/conll2003", "tner/ontonotes5"]]
        @param local_dataset: a dictionary (or a list) of paths to local BIO files eg.
            {"train": "examples/local_dataset_sample/train.txt", "test": "examples/local_dataset_sample/test.txt"}
        @param dataset_name: [optional] data name of huggingface dataset (should be sa
        @param cache_dir: [optional] cache directly
        @param batch_size: [optional] batch size
        @param dataset_split: [optional] data split of the dataset ('test' as default)
        @param cache_file_feature: [optional] save & load precompute data loader
        @param cache_file_prediction: [optional] save & load precompute model prediction
        @param span_detection_mode: [optional] return F1 of entity span detection (ignoring entity type error and cast
            as binary sequence classification as below)
            - NER                  : ["O", "B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG"]
            - Entity-span detection: ["O", "B-ENT", "I-ENT", "O", "B-ENT", "O", "B-ENT"]
        @param return_ci: [optional] return confidence interval by bootstrap
        @param unification_by_shared_label: [optional] map entities into a shared form
        @param separator: [optional] token separator (eg. '' for Japanese and Chinese)
        @return: a dictionary containing span f1 scores
        """
        self.eval()
        data, _ = get_dataset(
            dataset=dataset,
            dataset_name=dataset_name,
            local_dataset=local_dataset,
            concat_label2id=self.label2id,
            cache_dir=cache_dir)
        assert dataset_split in data, f'{dataset_split} is not in {data.keys()}'
        output = self.predict(
            inputs=data[dataset_split]['tokens'],
            labels=data[dataset_split]['tags'],
            batch_size=batch_size,
            cache_file_prediction=cache_file_prediction,
            cache_file_feature=cache_file_feature,
            separator=separator
        )
        return span_f1(output['prediction'], output['label'], span_detection_mode, return_ci=return_ci,
                       unification_by_shared_label=unification_by_shared_label)

    def save(self, save_dir: str):
        """ save checkpoint

        @param save_dir: directly to save the checkpoint files
        """

        def model_state(model):
            if self.parallel:
                return model.module
            return model

        if self.crf_layer is not None:
            model_state(self.model).config.update(
                {'crf_state_dict': {k: v.tolist() for k, v in model_state(self.crf_layer).state_dict().items()}})
        logging.info(f'saving model weight at {save_dir}')
        model_state(self.model).save_pretrained(save_dir)
        logging.info(f'saving tokenizer at {save_dir}')
        self.tokenizer.tokenizer.save_pretrained(save_dir)
