""" NER specific tokenization pipeline """
import os
import re
from itertools import groupby
from typing import List, Dict

import transformers
import torch
from torch import nn


PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning

__all__ = ('TokenizerFixed', 'Dataset', 'PAD_TOKEN_LABEL_ID')


def additional_special_tokens(tokenizer):
    """ get additional special token for beginning/separate/ending, {'input_ids': [0], 'attention_mask': [1]} """
    encode_first = tokenizer.encode_plus('sent1', 'sent2')
    # group by block boolean
    sp_token_mask = [i in tokenizer.all_special_ids for i in encode_first['input_ids']]
    group = [list(g) for _, g in groupby(sp_token_mask)]
    length = [len(g) for g in group]
    group_length = [[sum(length[:n]), sum(length[:n]) + len(g)] for n, g in enumerate(group) if all(g)]
    assert len(group_length) == 3, 'more than 3 special tokens group: {}'.format(group)
    sp_token_start = {k: v[group_length[0][0]:group_length[0][1]] for k, v in encode_first.items()}
    sp_token_sep = {k: v[group_length[1][0]:group_length[1][1]] for k, v in encode_first.items()}
    sp_token_end = {k: v[group_length[2][0]:group_length[2][1]] for k, v in encode_first.items()}
    return sp_token_start, sp_token_sep, sp_token_end


class TokenizerFixed:
    """ NER specific transform pipeline"""

    def __init__(self,
                 transformer_tokenizer: str,
                 cache_dir: str = None,
                 id2label: Dict = None,
                 padding_id: int = None):
        """ NER specific transform pipeline """
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_tokenizer, cache_dir=cache_dir)
        except Exception:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                transformer_tokenizer, cache_dir=cache_dir, local_files_only=True)
        self.id2label = id2label
        self.padding_id = padding_id if padding_id is not None else PAD_TOKEN_LABEL_ID
        self.label2id = {v: k for k, v in id2label.items()}
        self.pad_ids = {"labels": self.padding_id, "input_ids": self.tokenizer.pad_token_id, "__default__": 0}
        # Prefix is the special token that tokenizer will add to the beginning of each token/sentence to encode.
        # - tokenizer with prefix (XLM): 'I live in London' --> ['▁I', '▁live', '▁in', '▁London']
        # - tokenizer without prefix (RoBERTa): 'I live in London' --> ['I', 'Ġlive', 'Ġin', 'ĠLondon']
        # This is important when you fix the sequence labeling mismatch, because we count the number of subtokens for
        # each token in the sentence. We have to add prefix (half-space) at the begging of each token unless it's the
        # first token of the sentence when the tokenizer has no prefix, otherwise we shouldn't since tokenizer with
        # prefix will add prefix to each token automatically.
        self.prefix = self.__sp_token_prefix()
        # find special tokens to be added
        self.sp_token_start, _, self.sp_token_end = additional_special_tokens(self.tokenizer)

    def __sp_token_prefix(self):
        sentence_go_around = ''.join(self.tokenizer.tokenize('get tokenizer specific prefix'))
        prefix = sentence_go_around[:list(re.finditer('get', sentence_go_around))[0].span()[0]]
        return prefix if prefix != '' else None

    def encode_plus_en(self,
                       tokens,
                       labels: List = None,
                       is_tokenized: bool = True,
                       max_seq_length: int = 128,
                       mask_by_padding_token: bool = False):
        """ encoder for languages which split words by half-space """
        if not is_tokenized:
            tokens = tokens.split(' ')
        encode = self.tokenizer.encode_plus(
            ' '.join(tokens), max_length=max_seq_length, padding='max_length', truncation=True)
        if labels:
            assert is_tokenized
            assert len(tokens) == len(labels)
            fixed_labels = []
            for n, (label, word) in enumerate(zip(labels, tokens)):
                fixed_labels.append(label)
                if n != 0 and self.prefix is None:
                    sub_length = len(self.tokenizer.tokenize(' ' + word))
                else:
                    sub_length = len(self.tokenizer.tokenize(word))
                if sub_length > 1:
                    if mask_by_padding_token:
                        fixed_labels += [PAD_TOKEN_LABEL_ID] * (sub_length - 1)
                    else:
                        if self.id2label[label] == 'O':
                            fixed_labels += [self.label2id['O']] * (sub_length - 1)
                        else:
                            entity = '-'.join(self.id2label[label].split('-')[1:])
                            fixed_labels += [self.label2id['I-{}'.format(entity)]] * (sub_length - 1)
            tmp_padding = PAD_TOKEN_LABEL_ID if mask_by_padding_token else self.pad_ids['labels']
            fixed_labels = [tmp_padding] * len(self.sp_token_start['input_ids']) + fixed_labels
            fixed_labels = fixed_labels[:min(len(fixed_labels), max_seq_length - len(self.sp_token_end['input_ids']))]
            fixed_labels = fixed_labels + [tmp_padding] * (max_seq_length - len(fixed_labels))
            encode['labels'] = fixed_labels
        return encode

    # def fixed_encode_ja(self, tokens, labels: List = None, max_seq_length: int = 128):
    #     """ fixed encoding for language without halfspace in between words """
    #     dummy = '@'
    #     # get special tokens at start/end of sentence based on first token
    #     encode_all = self.tokenizer.batch_encode_plus(tokens)
    #     # token_ids without prefix/special tokens
    #     # `wifi` will be treated as `_wifi` and change the tokenize result, so add dummy on top of the sentence to fix
    #     token_ids_all = [[self.tokenizer.convert_tokens_to_ids(_t.replace(self.prefix, '').replace(dummy, ''))
    #                       for _t in self.tokenizer.tokenize(dummy+t)
    #                       if len(_t.replace(self.prefix, '').replace(dummy, '')) > 0]
    #                      for t in tokens]
    #
    #     for n in range(len(tokens)):
    #         if n == 0:
    #             encode = {k: v[n][:-len(self.sp_token_end[k])] for k, v in encode_all.items()}
    #             if labels:
    #                 encode['labels'] = [self.pad_ids['labels']] * len(self.sp_token_start['labels']) + [labels[n]]
    #                 encode['labels'] += [self.pad_ids['labels']] * (len(encode['input_ids']) - len(encode['labels']))
    #         else:
    #             encode['input_ids'] += token_ids_all[n]
    #             # other attribution without prefix/special tokens
    #             tmp_encode = {k: v[n] for k, v in encode_all.items()}
    #             s, e = len(self.sp_token_start['input_ids']), -len(self.sp_token_end['input_ids'])
    #             input_ids_with_prefix = tmp_encode.pop('input_ids')[s:e]
    #             prefix_length = len(input_ids_with_prefix) - len(token_ids_all[n])
    #             for k, v in tmp_encode.items():
    #                 s, e = len(self.sp_token_start['input_ids']) + prefix_length, -len(self.sp_token_end['input_ids'])
    #                 encode[k] += v[s:e]
    #             if labels:
    #                 encode['labels'] += [labels[n]] + [self.pad_ids['labels']] * (len((token_ids_all[n])) - 1)
    #
    #     # add special token at the end and padding/truncate accordingly
    #     for k in encode.keys():
    #         encode[k] = encode[k][:min(len(encode[k]), max_seq_length - len(self.sp_token_end[k]))]
    #         encode[k] += self.sp_token_end[k]
    #         pad_id = self.pad_ids[k] if k in self.pad_ids.keys() else self.pad_ids['__default__']
    #         encode[k] += [pad_id] * (max_seq_length - len(encode[k]))
    #     return encode

    def encode_plus_all(self,
                        tokens: List,
                        labels: List = None,
                        is_tokenized: bool = True,
                        language: str = 'en',
                        max_length: int = None,
                        mask_by_padding_token: bool = False):
        max_length = self.tokenizer.max_len_single_sentence if max_length is None else max_length
        shared_param = {'language': language, 'max_length': max_length, 'mask_by_padding_token': mask_by_padding_token,
                        'is_tokenized': is_tokenized}
        if labels:
            return [self.encode_plus(*i, **shared_param) for i in zip(tokens, labels)]
        else:
            return [self.encode_plus(i, **shared_param) for i in tokens]

    def encode_plus(self,
                    tokens,
                    labels: List = None,
                    is_tokenized: bool = True,
                    language: str = 'en',
                    max_length: int = 128,
                    mask_by_padding_token: bool = False):
        if language == 'ja':
            raise ValueError('Need to refactor')
        else:
            return self.encode_plus_en(tokens, labels, is_tokenized, max_length, mask_by_padding_token)

    @property
    def all_special_ids(self):
        return self.tokenizer.all_special_ids

    @property
    def all_special_tokens(self):
        return self.tokenizer.all_special_tokens

    def tokenize(self, *args, **kwargs):
        return self.tokenizer.tokenize(*args, **kwargs)


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset wrapper converting into tensor """
    float_tensors = ['attention_mask']

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
