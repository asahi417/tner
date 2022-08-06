""" NER tokenizer, which handle the sequence mismatch of annotation/tokenization of the dataset vs language model.
E.g.) Given a sentence "I live in Tokyo.", you tokenize it by half-space i.e. ["I", "live", "in", "Tokyo"], but
the language model tokenizes it in rather ["I", "liv", "e", "in", "To", "kyo"]. This module is designed to handle
such sequence mismatch by projecting the user-defined-tokenization to language-model-tokenization, and retain the
original tokenization after the prediction.
"""
import os
import re
from itertools import groupby
from typing import List, Dict

import torch
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index


class NERTokenizer:
    """ NER Tokenizer """

    def __init__(self,
                 tokenizer_name: str,
                 id2label: Dict,
                 padding_id: int = None,
                 use_auth_token: bool = False,
                 is_xlnet: bool = False):
        """ NER Tokenizer

        @param tokenizer_name: the alias of huggingface tokenizer (`tner/roberta-large-tweetner-2021`)
        @param id2label: dictionary of id to label (`{"0": "O", "1": "B-ORG", ...}`)
        @param padding_id: [optional] id of padding tokne
        @param use_auth_token: [optional] Huggingface transformers argument of use_auth_token
        @param is_xlnet: [optional] XLNet's tokenizer
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, use_auth_token=use_auth_token)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, use_auth_token=use_auth_token, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = PAD_TOKEN_LABEL_ID
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.pad_ids = {"labels": PAD_TOKEN_LABEL_ID if padding_id is None else padding_id,
                        "input_ids": self.tokenizer.pad_token_id, "__default__": 0}
        self.prefix = self.__sp_token_prefix()
        self.sp_token_start, _, self.sp_token_end = self.__additional_special_tokens()
        self.is_xlnet = is_xlnet

    def __sp_token_prefix(self):
        """ return language model-specific prefix token """
        sentence_go_around = ''.join(self.tokenizer.tokenize('get tokenizer specific prefix'))
        prefix = sentence_go_around[:list(re.finditer('get', sentence_go_around))[0].span()[0]]
        return prefix if prefix != '' else None

    def __additional_special_tokens(self):
        """ return language model-specific special token for beginning/separate/ending.
        {'input_ids': [0], 'attention_mask': [1]} """
        encode_first = self.tokenizer.encode_plus('sent1', 'sent2')
        # group by block boolean
        sp_token_mask = [i in self.tokenizer.all_special_ids for i in encode_first['input_ids']]
        group = [list(g) for _, g in groupby(sp_token_mask)]
        length = [len(g) for g in group]
        group_length = [[sum(length[:n]), sum(length[:n]) + len(g)] for n, g in enumerate(group) if all(g)]
        sp_token_empty = {k: [] for k in encode_first.keys()}
        if len(group_length) == 0:
            sp_token_start = sp_token_sep = sp_token_end = sp_token_empty
        elif len(group_length) < 3:
            if not group[0][0]:
                sp_token_start = sp_token_empty
            else:
                _group_length = group_length.pop(0)
                sp_token_start = {k: v[_group_length[0]:_group_length[1]] for k, v in encode_first.items()}
            if not any(g[0] for g in group[1:-1]):
                sp_token_sep = sp_token_empty
            else:
                _group_length = group_length.pop(0)
                sp_token_sep = {k: v[_group_length[0]:_group_length[1]] for k, v in encode_first.items()}
            if not group[-1][0]:
                sp_token_end = sp_token_empty
            else:
                _group_length = group_length.pop(0)
                sp_token_end = {k: v[_group_length[0]:_group_length[1]] for k, v in encode_first.items()}
        else:
            assert len(group_length) == 3, f'more than 3 special tokens group: {group}'
            sp_token_start = {k: v[group_length[0][0]:group_length[0][1]] for k, v in encode_first.items()}
            sp_token_sep = {k: v[group_length[1][0]:group_length[1][1]] for k, v in encode_first.items()}
            sp_token_end = {k: v[group_length[2][0]:group_length[2][1]] for k, v in encode_first.items()}
        return sp_token_start, sp_token_sep, sp_token_end

    def encode_plus(self,
                    tokens: List,
                    labels: List = None,
                    max_length: int = None,
                    mask_by_padding_token: bool = False,
                    separator: str = ' '):
        """ return encoded feature given half-space-split tokens

        @param tokens: an input sentence tokenized by half-space, ["I", "live", "in", "London"]
        @param labels: [optional] a sequence of label corresponding to the input token
        @param max_length: [optional] max length of language model input
        @param [optional] mask_by_padding_token: Padding sequence has two cases:
            (i) Padding upto max_length: if True, padding such tokens by {PADDING_TOKEN}, else by "O"
            (ii) Intermediate sub-token: For example, we have tokens in a sentence ["New", "York"] with labels
                ["B-LOC", "I-LOC"], which language model tokenizes into ["New", "Yor", "k"]. If mask_by_padding_token
                is True, the new label is ["B-LOC", "I-LOC", {PADDING_TOKEN}], otherwise ["B-LOC", "I-LOC", "I-LOC"].
        @param separator: [optional] token separator (eg. '' for Japanese and Chinese)
        @return: a dictionary of encoded feature
        """
        if max_length is None:
            encode = self.tokenizer.encode_plus(separator.join(tokens))
        else:
            encode = self.tokenizer.encode_plus(
                separator.join(tokens), max_length=max_length, padding='max_length', truncation=True
            )
        if labels:
            assert len(tokens) == len(labels)
            fixed_labels = []
            for n, (label, word) in enumerate(zip(labels, tokens)):
                fixed_labels.append(label)
                if n != 0 and self.prefix is None:
                    sub_length = len(self.tokenizer.tokenize(separator + word))
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
                            fixed_labels += [self.label2id[f'I-{entity}']] * (sub_length - 1)
            tmp_padding = PAD_TOKEN_LABEL_ID if mask_by_padding_token else self.pad_ids['labels']
            fixed_labels = [tmp_padding] * len(self.sp_token_start['input_ids']) + fixed_labels
            fixed_labels = fixed_labels[:min(len(fixed_labels), max_length - len(self.sp_token_end['input_ids']))]

            if self.is_xlnet:  # XLNet pad before the sentence
                fixed_labels = [tmp_padding] * (max_length - len(fixed_labels) - len(self.sp_token_end['input_ids'])) + \
                               fixed_labels + [tmp_padding] * len(self.sp_token_end['input_ids'])
            else:
                fixed_labels = fixed_labels + [tmp_padding] * (max_length - len(fixed_labels))
            assert len(fixed_labels) == len(encode['input_ids'])
            encode['labels'] = fixed_labels
        return encode

    def encode_plus_all(self,
                        tokens: List,
                        labels: List = None,
                        max_length: int = None,
                        mask_by_padding_token: bool = False,
                        separator: str = ' '):
        """ batch processing of `self.encode_plus`

        @param tokens: a list of input sentences tokenized by half-space, [["I", "live", ...], ["You", "live", ...]]
        @param labels: [optional]
        @param max_length: [optional] max length of language model input
        @param [optional] mask_by_padding_token: Padding sequence has two cases:
            (i) Padding upto max_length: if True, padding such tokens by {PADDING_TOKEN}, else by "O"
            (ii) Intermediate sub-token: For example, we have tokens in a sentence ["New", "York"] with labels
                ["B-LOC", "I-LOC"], which language model tokenizes into ["New", "Yor", "k"]. If mask_by_padding_token
                is True, the new label is ["B-LOC", "I-LOC", {PADDING_TOKEN}], otherwise ["B-LOC", "I-LOC", "I-LOC"].
        @param separator: [optional] token separator (eg. '' for Japanese and Chinese)
        @return: a list of dictionary of encoded feature
        """
        if self.is_xlnet and max_length is None:
            max_length = 512
        else:
            max_length = self.tokenizer.max_len_single_sentence if max_length is None else max_length
        shared_param = {'max_length': max_length, 'mask_by_padding_token': mask_by_padding_token,
                        'separator': separator}
        if labels:
            assert len(labels) == len(tokens), f"{len(labels)} != {len(tokens)}"
            return [self.encode_plus(*i, **shared_param) for i in zip(tokens, labels)]
        return [self.encode_plus(i, **shared_param) for i in tokens]
