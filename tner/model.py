""" Named-Entity-Recognition (NER) modeling """
import os
import random
import json
import logging
from time import time
from typing import List
from itertools import groupby
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

import transformers
import torch
from torch import nn
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch.utils.tensorboard import SummaryWriter

from .get_dataset import get_dataset_ner
from .checkpoint_versioning import Argument
from .tokenizer import Transforms


PROGRESS_INTERVAL = 100
CACHE_DIR = os.getenv("CACHE_DIR", './cache')
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning
os.makedirs(CACHE_DIR, exist_ok=True)

__all__ = ('TrainTransformersNER', 'TransformersNER')


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


class TrainTransformersNER:
    """ Named-Entity-Recognition (NER) trainer """

    def __init__(self,
                 dataset: (str, List) = None,
                 checkpoint: str = None,
                 checkpoint_dir: str = None,
                 transformers_model: str = 'xlm-roberta-large',
                 random_seed: int = 1234,
                 lr: float = 1e-5,
                 total_step: int = 5000,
                 warmup_step: int = 700,
                 weight_decay: float = 1e-7,
                 batch_size: int = 32,
                 max_seq_length: int = 128,
                 fp16: bool = False,
                 max_grad_norm: float = 1.0,
                 lower_case: bool = False,
                 num_worker: int = 1):
        """ Named-Entity-Recognition (NER) trainer

         Parameter
        -----------------
        dataset: list
            list of dataset for training, alias of preset dataset (see tner.VALID_DATASET) or path to custom dataset
            folder such as https://github.com/asahi417/tner/tree/master/tests/sample_data
            eg) ['panx_dataset/en', 'conll2003', 'tests/custom_dataset_sample']
        checkpoint: str
            checkpoint folder to load weight
        checkpoint_dir: str
            checkpoint directory ('checkpoint_dir/checkpoint' is regarded as a checkpoint path)
        transformers_model: str
            language model alias from huggingface (https://huggingface.co/transformers/v2.2.0/pretrained_models.html)
        random_seed: int
            random seed through the experiment
        lr: float
            learning rate
        total_step: int
            total training step
        warmup_step: int
            step for linear warmup
        weight_decay: float
            parameter for weight decay
        batch_size: int
            batch size for training
        max_seq_length: int
            language model's maximum sequence length
        fp16: bool
            training with mixture precision mode
        max_grad_norm: float
            gradient clipping
        lower_case: bool
            convert the dataset into lowercase
        num_worker: int
            number of worker for torch.Dataloader class
        """
        logging.info('*** initialize network ***')
        if num_worker == 1:
            os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
        self.num_worker = num_worker

        # checkpoint version
        self.args = Argument(
            dataset=dataset,
            checkpoint_dir=checkpoint_dir,
            checkpoint=checkpoint,
            transformers_model=transformers_model,
            random_seed=random_seed,
            lr=lr,
            total_step=total_step,
            warmup_step=warmup_step,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            fp16=fp16,
            max_grad_norm=max_grad_norm,
            lower_case=lower_case)

        self.is_trained = self.args.model_statistics is not None
        # fix random seed
        random.seed(self.args.random_seed)
        transformers.set_seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

        self.label_to_id = self.args.label_to_id
        self.id_to_label = None if self.label_to_id is None else {v: str(k) for k, v in self.label_to_id.items()}
        self.dataset_split = None
        self.language = None
        self.unseen_entity_set = None
        self.optimizer = None
        self.scheduler = None
        self.scale_loss = None
        self.n_gpu = torch.cuda.device_count()
        self.device = 'cuda' if self.n_gpu > 0 else 'cpu'
        self.model = None
        self.transforms = None
        self.__epoch = 0
        self.__step = 0

    def __setup_data(self, dataset_name, lower_case):
        """ set up dataset """
        # assert self.dataset_split is None, "dataset has already been loaded"
        if self.is_trained:
            self.dataset_split, self.label_to_id, self.language, self.unseen_entity_set = get_dataset_ner(
                dataset_name,
                label_to_id=self.label_to_id,
                fix_label_dict=True,
                lower_case=lower_case)
        else:
            self.dataset_split, self.label_to_id, self.language, self.unseen_entity_set = get_dataset_ner(
                dataset_name,
                lower_case=lower_case)
        self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}

    def __setup_model(self):
        """ set up language model """
        if self.model is not None:
            return
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(
            self.args.transformers_model,
            config=transformers.AutoConfig.from_pretrained(
                self.args.transformers_model,
                num_labels=len(self.id_to_label),
                id2label=self.id_to_label,
                label2id=self.label_to_id,
                cache_dir=CACHE_DIR)
        )
        self.transforms = Transforms(self.args.transformers_model)
        if not self.is_trained:
            # optimizer
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0}]
            self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=1e-8)

            # scheduler
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_step, num_training_steps=self.args.total_step)
        else:
            # load check point
            self.model.load_state_dict(self.args.model_statistics['model_state_dict'])

        # GPU allocation
        self.model.to(self.device)

        # GPU mixture precision
        if self.args.fp16:
            try:
                from apex import amp  # noqa: F401
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level='O1', max_loss_scale=2 ** 13, min_loss_scale=1e-5)
                self.master_params = amp.master_params
                self.scale_loss = amp.scale_loss
                logging.info('using `apex.amp`')
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        # multi-gpus
        if self.n_gpu > 1:
            # multi-gpu training (should be after apex fp16 initialization)
            self.model = torch.nn.DataParallel(self.model.cuda())
            logging.info('using `torch.nn.DataParallel`')
        logging.info('running on %i GPUs' % self.n_gpu)

    def __setup_loader(self, data_type: str, batch_size: int, max_seq_length: int):
        """ setup data loader """
        assert self.dataset_split, 'run __setup_data firstly'
        if data_type not in self.dataset_split.keys():
            return None
        is_train = data_type == 'train'
        features = self.transforms.encode_plus_all(
            tokens=self.dataset_split[data_type]['data'],
            labels=self.dataset_split[data_type]['label'],
            language=self.language,
            max_length=max_seq_length)
        data_obj = Dataset(features)
        return torch.utils.data.DataLoader(
            data_obj, num_workers=self.num_worker, batch_size=batch_size, shuffle=is_train, drop_last=is_train)

    @property
    def checkpoint(self):
        return self.args.checkpoint_dir

    def test(self,
             test_dataset: str = None,
             ignore_entity_type: bool = False,
             lower_case: bool = False,
             batch_size_validation: int = None,
             max_seq_length_validation: int = None):
        """ Test NER model on specific dataset

         Parameter
        -------------
        test_dataset: str
            target dataset to test, alias of preset dataset (see tner.VALID_DATASET) or path to custom dataset folder
            eg) https://github.com/asahi417/tner/tree/master/tests/sample_data
        ignore_entity_type: bool
            test without entity type (entity span detection)
        lower_case: bool
            converting test data into lower-cased
        """
        # setup model/dataset/data loader
        assert self.is_trained, 'finetune model before'
        if test_dataset is None:
            assert len(self.args.dataset) == 1, "test dataset can not be determined"
            dataset = self.args.dataset[0]
        else:
            dataset = test_dataset
        filename = 'test_{}{}{}.json'.format(
            dataset.replace('/', '-'), '_ignore' if ignore_entity_type else '', '_lower' if lower_case else '')
        filename = os.path.join(self.args.checkpoint_dir, filename)
        if os.path.exists(filename):
            return
        assert type(dataset) is str
        batch_size = batch_size_validation if batch_size_validation else self.args.batch_size
        max_seq_length = max_seq_length_validation if max_seq_length_validation else self.args.max_seq_length
        self.__setup_data(dataset, lower_case)
        self.__setup_model()

        if 'train' in self.dataset_split.keys():
            self.dataset_split.pop('train')

        data_loader = {k: self.__setup_loader(k, batch_size, max_seq_length) for k in self.dataset_split.keys()}

        logging.info('testing model on {}'.format(dataset))
        logging.info('data_loader: {}'.format(str(list(data_loader.keys()))))

        # run inference
        start_time = time()
        metrics = {}
        params = dict(ignore_entity_type=ignore_entity_type, unseen_entity_set=self.unseen_entity_set)
        for k, v in data_loader.items():
            assert v is not None, '{} data split is not found'.format(k)
            metrics[k] = self.__epoch_valid(v, prefix=k, **params)
            self.release_cache()

        # export result
        with open(filename, 'w') as f:
            json.dump(metrics, f)
        logging.info('[test completed, %0.2f sec in total]' % (time() - start_time))
        logging.info('export metrics at: {}'.format(filename))

    def train(self,
              monitor_validation: bool = False,
              batch_size_validation: int = 1,
              max_seq_length_validation: int = 128):
        """ Train NER model

         Parameter
        -------------
        monitor_validation: bool
            display validation result at the end of each epoch
        batch_size_validation: int
            batch size for validation monitoring
        max_seq_length_validation: int
            max seq length for validation monitoring
        """
        # setup model/dataset/data loader
        assert not self.is_trained, 'model has been already finetuned'
        self.__setup_data(self.args.dataset, self.args.lower_case)
        self.__setup_model()
        writer = SummaryWriter(log_dir=self.args.checkpoint_dir)

        data_loader = {'train': self.__setup_loader('train', self.args.batch_size, self.args.max_seq_length)}
        if monitor_validation and 'valid' in self.dataset_split.keys():
            data_loader['valid'] = self.__setup_loader('valid', batch_size_validation, max_seq_length_validation)
        else:
            data_loader['valid'] = None

        # start experiment
        start_time = time()
        logging.info('data_loader: %s' % str(list(data_loader.keys())))
        logging.info('*** start training from step %i, epoch %i ***' % (self.__step, self.__epoch))
        try:
            while True:
                if_training_finish = self.__epoch_train(data_loader['train'], writer=writer)
                self.release_cache()
                if data_loader['valid']:
                    try:
                        self.__epoch_valid(data_loader['valid'], writer=writer, prefix='valid')
                    except RuntimeError:
                        logging.exception('*** RuntimeError: skip validation ***')

                    self.release_cache()
                if if_training_finish:
                    break
                self.__epoch += 1
        except RuntimeError:
            logging.exception('*** RuntimeError ***')

        except KeyboardInterrupt:
            logging.info('*** KeyboardInterrupt ***')

        logging.info('[training completed, %0.2f sec in total]' % (time() - start_time))
        model_wts = self.model.module.state_dict() if self.n_gpu > 1 else self.model.state_dict()
        torch.save({'model_state_dict': model_wts}, os.path.join(self.args.checkpoint_dir, 'model.pt'))
        with open(os.path.join(self.args.checkpoint_dir, 'label_to_id.json'), 'w') as f:
            json.dump(self.label_to_id, f)
        writer.close()
        logging.info('ckpt saved at %s' % self.args.checkpoint_dir)
        self.is_trained = True

    def __epoch_train(self, data_loader, writer):
        """ single epoch training: returning flag which is True if training has been completed """
        self.model.train()
        for i, encode in enumerate(data_loader, 1):

            # update model
            encode = {k: v.to(self.device) for k, v in encode.items()}
            self.optimizer.zero_grad()
            loss = self.model(**encode, return_dict=True)['loss']
            if self.n_gpu > 1:
                loss = loss.mean()
            if self.args.fp16:
                with self.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.master_params(self.optimizer), self.args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            # optimizer and scheduler step
            self.optimizer.step()
            self.scheduler.step()

            # log instantaneous accuracy, loss, and learning rate
            inst_loss = loss.cpu().detach().item()
            inst_lr = self.optimizer.param_groups[0]['lr']
            if writer:
                writer.add_scalar('train/loss', inst_loss, self.__step)
                writer.add_scalar('train/learning_rate', inst_lr, self.__step)
            if self.__step % PROGRESS_INTERVAL == 0:
                logging.info('[epoch %i] * (training step %i) loss: %.3f, lr: %0.8f'
                             % (self.__epoch, self.__step, inst_loss, inst_lr))
            self.__step += 1

            # break
            if self.__step >= self.args.total_step:
                logging.info('reached maximum step')
                return True

        return False

    def __epoch_valid(self, data_loader, prefix, writer=None,
                      unseen_entity_set: set = None, ignore_entity_type: bool = False):
        """ single epoch validation/test """
        # aggregate prediction and true label
        self.model.eval()
        seq_pred, seq_true = [], []
        for encode in data_loader:
            encode = {k: v.to(self.device) for k, v in encode.items()}
            labels_tensor = encode.pop('labels')
            logit = self.model(**encode, return_dict=True)['logits']
            _true = labels_tensor.cpu().detach().int().tolist()
            _pred = torch.max(logit, 2)[1].cpu().detach().int().tolist()
            for b in range(len(_true)):
                _pred_list, _true_list = [], []
                for s in range(len(_true[b])):
                    if _true[b][s] != PAD_TOKEN_LABEL_ID:
                        _true_list.append(self.id_to_label[_true[b][s]])
                        if unseen_entity_set is None:
                            _pred_list.append(self.id_to_label[_pred[b][s]])
                        else:
                            __pred = self.id_to_label[_pred[b][s]]
                            if __pred in unseen_entity_set:
                                _pred_list.append('O')
                            else:
                                _pred_list.append(__pred)
                assert len(_pred_list) == len(_true_list)
                if len(_true_list) > 0:
                    if ignore_entity_type:
                        # ignore entity type and focus on entity position
                        _true_list = [i if i == 'O' else '-'.join([i.split('-')[0], 'entity']) for i in _true_list]
                        _pred_list = [i if i == 'O' else '-'.join([i.split('-')[0], 'entity']) for i in _pred_list]
                    seq_true.append(_true_list)
                    seq_pred.append(_pred_list)

        # compute metrics
        metric = {
            "f1": f1_score(seq_true, seq_pred) * 100,
            "recall": recall_score(seq_true, seq_pred) * 100,
            "precision": precision_score(seq_true, seq_pred) * 100,
        }

        try:
            summary = classification_report(seq_true, seq_pred)
            logging.info('[epoch {}] ({}) \n {}'.format(self.__epoch, prefix, summary))
        except Exception:
            logging.exception('classification_report raises error')
            summary = ''
        metric['summary'] = summary
        if writer:
            writer.add_scalar('{}/f1'.format(prefix), metric['f1'], self.__epoch)
            writer.add_scalar('{}/recall'.format(prefix), metric['recall'], self.__epoch)
            writer.add_scalar('{}/precision'.format(prefix), metric['precision'], self.__epoch)
        return metric

    def release_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()


class TransformersNER:
    """ Named-Entity-Recognition (NER) API for an inference """

    def __init__(self, checkpoint: str):
        """ Named-Entity-Recognition (NER) API for an inference

         Parameter
        ------------
        checkpoint: str
            path to model weight file
        """
        logging.info('*** initialize network ***')
        checkpoint = checkpoint.replace('model.pt', '')

        # checkpoint version
        self.args = Argument(checkpoint=checkpoint)
        if self.args.model_statistics is None:
            raise ValueError('model is not trained')
        self.label_to_id = self.args.label_to_id
        self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}
        self.model = transformers.AutoModelForTokenClassification.from_pretrained(
            self.args.transformers_model,
            config=transformers.AutoConfig.from_pretrained(
                self.args.transformers_model,
                num_labels=len(self.id_to_label),
                id2label=self.id_to_label,
                label2id=self.label_to_id,
                cache_dir=CACHE_DIR)
        )
        self.transforms = Transforms(self.args.transformers_model)
        self.model.load_state_dict(self.args.model_statistics['model_state_dict'])

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
