import logging
from typing import List
from itertools import groupby
import transformers
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .tokenizer import Transforms, Dataset

__all__ = 'TransformersNER'


class TransformersNER:
    """ Named-Entity-Recognition (NER) API for an inference """

    def __init__(self, transformers_model: str, cache_dir: str = None):
        """ Named-Entity-Recognition (NER) API for an inference

         Parameter
        ------------
        transformers_model: str
            model name on transformers model hub or path to model directory
        """
        logging.info('*** initialize network ***')
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(transformers_model)
        self.id_to_label = {v: str(k) for k, v in self.model.config.label2id.items()}
        self.transforms = Transforms(transformers_model, cache_dir=cache_dir)

        # GPU allocation
        self.n_gpu = torch.cuda.device_count()
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model.to(self.device)

    @staticmethod
    def decode_ner_tags(tag_sequence, tag_probability, non_entity: str = 'O'):
        """ take tag sequence, return list of entity
        input:  ["B-LOC", "O", "O", "B-ORG", "I-ORG", "O"]
        return: [['LOC', [0, 1]], ['ORG', [3, 5]]]
        """
        assert len(tag_sequence) == len(tag_probability)
        unique_type = list(set(i.split('-')[-1] for i in tag_sequence if i != non_entity))
        result = []
        for i in unique_type:
            mask = [t.split('-')[-1] == i for t, p in zip(tag_sequence, tag_probability)]

            # find blocks of True in a boolean list
            group = list(map(lambda x: list(x[1]), groupby(mask)))
            length = list(map(lambda x: len(x), group))
            group_length = [[sum(length[:n]), sum(length[:n]) + len(g)] for n, g in enumerate(group) if all(g)]

            # get entity
            for g in group_length:
                result.append([i, g])
        result = sorted(result, key=lambda x: x[1][0])
        return result

    def predict(self, x: List, max_seq_length: int = 128):
        """ Get prediction

         Parameter
        ----------------
        x: list
            batch of input texts
        max_seq_length: int
            maximum sequence length for running an inference

         Return
        ----------------
        entities: list
            list of dictionary where each consists of
                'type': (str) entity type
                'position': (list) start position and end position
                'mention': (str) mention
        """
        self.model.eval()
        encode_list = self.transforms.encode_plus_all(x, max_length=max_seq_length)
        data_loader = torch.utils.data.DataLoader(Dataset(encode_list), batch_size=len(encode_list))
        encode = list(data_loader)[0]
        logit = self.model(**{k: v.to(self.device) for k, v in encode.items()}, return_dict=True)['logits']
        entities = []
        for n, e in enumerate(encode['input_ids'].cpu().tolist()):
            sentence = self.transforms.tokenizer.decode(e, skip_special_tokens=True)

            pred = torch.max(logit[n], dim=-1)[1].cpu().tolist()
            activated = nn.Softmax(dim=-1)(logit[n])
            prob = torch.max(activated, dim=-1)[0].cpu().tolist()
            pred = [self.id_to_label[_p] for _p in pred]
            tag_lists = self.decode_ner_tags(pred, prob)

            _entities = []
            for tag, (start, end) in tag_lists:
                mention = self.transforms.tokenizer.decode(e[start:end], skip_special_tokens=True)
                if not len(mention.strip()): continue
                start_char = len(self.transforms.tokenizer.decode(e[:start], skip_special_tokens=True))
                if sentence[start_char] == ' ':
                    start_char += 1
                end_char = start_char + len(mention)
                if mention != sentence[start_char:end_char]:
                    logging.warning('entity mismatch: {} vs {}'.format(mention, sentence[start_char:end_char]))
                result = {'type': tag, 'position': [start_char, end_char], 'mention': mention,
                          'probability': sum(prob[start: end])/(end - start)}
                _entities.append(result)

            entities.append({'entity': _entities, 'sentence': sentence})
        return entities
