import os
import logging
import pickle
from typing import List, Dict
from itertools import groupby

import transformers
import torch

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

from .tokenizer import TokenizerFixed, Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message

__all__ = 'TransformersNER'


def decode_ner_tags(tag_sequence, non_entity: str = 'O'):
    """ take tag sequence, return list of entity
    input:  ["B-LOC", "O", "O", "B-ORG", "I-ORG", "O"]
    return: [['LOC', [0, 1]], ['ORG', [3, 5]]]
    """
    unique_type = list(set(i.split('-')[-1] for i in tag_sequence if i != non_entity))
    result = []
    for i in unique_type:
        mask = [t.split('-')[-1] == i for t in tag_sequence]

        # find blocks of True in a boolean list
        group = list(map(lambda x: list(x[1]), groupby(mask)))
        length = list(map(lambda x: len(x), group))
        group_length = [[sum(length[:n]), sum(length[:n]) + len(g)] for n, g in enumerate(group) if all(g)]

        # get entity
        for g in group_length:
            result.append([i, g])
    result = sorted(result, key=lambda x: x[1][0])
    return result


def pickle_save(obj, path: str):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    with open(path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


def load_hf(model_name, cache_dir, label2id, local_files_only=False):
    """ load huggingface checkpoints """
    logging.info('initialize language model with `{}`'.format(model_name))
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
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_name, config=config, cache_dir=cache_dir, local_files_only=local_files_only)
    return model


class TransformersNER:

    def __init__(self,
                 model: str,
                 max_length: int = 512,
                 crf: bool = True,
                 label2id: Dict = None,
                 cache_dir: str = None):
        self.model_name = model
        self.max_length = max_length
        self.crf = crf

        # load model
        try:
            self.model = load_hf(self.model_name, cache_dir, label2id)
        except ValueError:
            self.model = load_hf(self.model_name, cache_dir, label2id, local_files_only=True)

        # load crf layer
        self.crf_layer = ConditionalRandomField(
            num_tags=len(self.model.config.id2label),
            constraints=allowed_transitions(constraint_type="BIO", labels=self.model.config.id2label)
        )
        if 'crf_state_dict' in self.model.config.to_dict().keys():
            state = {k: torch.FloatTensor(v) for k, v in self.model.config.crf_state_dict.items()}
            self.crf_layer.load_state_dict(state)
            self.crf = True

        # load pre processor
        self.tokenizer = TokenizerFixed(self.model_name, cache_dir=cache_dir)

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = False
        if torch.cuda.device_count() > 1:
            self.parallel = True
            self.model = torch.nn.DataParallel(self.model)
            self.crf_layer = torch.nn.DataParallel(self.crf_layer)
        self.model.to(self.device)
        self.crf_layer.to(self.device)
        logging.info('{} GPUs are in use'.format(torch.cuda.device_count()))

        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, save_dir):
        self.model.config.update({'crf_state_dict': {k: v.tolist() for k, v in self.crf_layer.state_dict().items()}})
        if self.parallel:
            self.model.module.save_pretrained(save_dir)
        else:
            self.model.save_pretrained(save_dir)
        self.tokenizer.tokenizer.save_pretrained(save_dir)

    def encode_to_loss(self, encode: Dict):
        assert 'labels' in encode
        encode = {k: v.to(self.device) for k, v in encode.items()}
        output = self.model(**encode)
        print(output.keys())
        if self.crf:
            loss = -self.crf_layer(output['logits'], output['labels'], encode['attention_mask'])
        else:
            loss = output['loss']
        return loss.mean() if self.parallel else loss

    def encode_to_prediction(self, encode: Dict):
        encode = {k: v.to(self.device) for k, v in encode.items()}
        output = self.model(**encode)
        if self.crf:
            best_path = self.crf_layer.viterbi_tags(output['logit'], encode['attention_mask'])
            pred_results = []
            for tag_seq, _ in range(best_path):
                pred_results.append(tag_seq)
            return pred_results
        else:
            return torch.max(output['logit'], dim=-1)[1].cpu().tolist()

    def get_data_loader(self,
                        inputs,
                        labels: List = None,
                        batch_size: int = None,
                        num_workers: int = 0,
                        shuffle: bool = False,
                        drop_last: bool = False,
                        cache_path: str = None):
        """ Transform features (produced by BERTClassifier.preprocess method) to data loader. """
        if cache_path is not None and os.path.exists(cache_path):
            logging.info('loading preprocessed feature from {}'.format(cache_path))
            out = pickle_load(cache_path)
        else:
            out = self.tokenizer.encode_plus_all(tokens=inputs, labels=labels, max_length=self.max_length)
            # remove overflow text
            logging.info('encode all the data: {}'.format(len(out)))

            # cache the encoded data
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                pickle_save(out, cache_path)
                logging.info('preprocessed feature is saved at {}'.format(cache_path))

        batch_size = len(out) if batch_size is None else batch_size
        return torch.utils.data.DataLoader(
            Dataset(out), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    def predict(self,
                inputs: List,
                batch_size: int = None,
                num_workers: int = 0,
                decode_bio: bool = False):
        self.eval()
        loader = self.get_data_loader(inputs, batch_size=batch_size, num_workers=num_workers)
        pred_list = []
        for i in loader:
            pred = self.encode_to_prediction(i)
            if decode_bio:
                pred = [self.model.config.id2label[_p] for _p in pred]
                pred = decode_ner_tags(pred)
            pred_list.append(pred)
        return pred_list


