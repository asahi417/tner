""" Named-Entity-Recognition (NER) modeling """
import os
import random
import json
import logging
from time import time
from typing import List
import transformers
import torch
from torch import nn
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch.utils.tensorboard import SummaryWriter

from .get_dataset import get_dataset_ner
from .checkpoint_versioning import Argument
from .tokenizer import Transforms, Dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
PROGRESS_INTERVAL = 100
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning

__all__ = 'TrainTransformersNER'


class TrainTransformersNER:
    """ Named-Entity-Recognition (NER) trainer """

    def __init__(self,
                 checkpoint_dir: str,
                 dataset: (str, List) = None,
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
                 num_worker: int = 0,
                 cache_dir: str = None):
        """ Named-Entity-Recognition (NER) trainer

         Parameter
        -----------------
        checkpoint_dir: str
            Checkpoint folder to log the model relevant files such as weight file. Once it's generated, one can use the
            directory for transformers.AutoModelForTokenClassification.from_pretrained as well as model sharing on
            transformers model hub.
        dataset: list or str
            List or str of dataset for training, alias of preset dataset (see tner.VALID_DATASET) or
            path to custom dataset
            eg) ['panx_dataset/en', 'conll2003', 'tests/custom_dataset_sample']
        transformers_model: str
            Model name from transformers model hub or path to local checkpoint directory, on which we perform
            finetunig or testing.
        random_seed: int
            Random seed through the experiment
        lr: float
            Learning rate
        total_step: int
            Total training step
        warmup_step: int
            Step for linear warmup
        weight_decay: float
            Parameter for weight decay
        batch_size: int
            Batch size for training
        max_seq_length: int
            Language model's maximum sequence length
        fp16: bool
            Training with mixture precision mode
        max_grad_norm: float
            Gradient clipping
        lower_case: bool
            Convert the training dataset into lowercase
        num_worker: int
            Number of worker for torch.Dataloader class
        cache_dir: str
            Cache directory for transformers
        """
        logging.info('*** initialize network ***')
        if num_worker <= 1:
            os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
        self.num_worker = num_worker
        self.cache_dir = cache_dir

        # checkpoint version
        self.args = Argument(
            checkpoint_dir=checkpoint_dir,
            dataset=dataset,
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
            lower_case=lower_case
        )

        # fix random seed
        random.seed(self.args.random_seed)
        transformers.set_seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

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
        self.label_to_id = None
        self.id_to_label = None
        self.__train_called = False

    def __setup_model_data(self, dataset, lower_case):
        """ set up data/language model """
        if self.model is not None:
            return
        if self.args.is_trained:
            self.model = transformers.AutoModelForTokenClassification.from_pretrained(self.args.transformers_model)
            self.transforms = Transforms(self.args.transformers_model, cache_dir=self.cache_dir)
            self.label_to_id = self.model.config.label2id
            self.dataset_split, self.label_to_id, self.language, self.unseen_entity_set = get_dataset_ner(
                dataset, label_to_id=self.label_to_id, fix_label_dict=True, lower_case=lower_case)
            self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}
        else:
            self.dataset_split, self.label_to_id, self.language, self.unseen_entity_set = get_dataset_ner(
                dataset, lower_case=lower_case)
            self.id_to_label = {v: str(k) for k, v in self.label_to_id.items()}
            config = transformers.AutoConfig.from_pretrained(
                self.args.transformers_model,
                num_labels=len(self.label_to_id),
                id2label=self.id_to_label,
                label2id=self.label_to_id,
                cache_dir=self.cache_dir)

            self.model = transformers.AutoModelForTokenClassification.from_pretrained(
                self.args.transformers_model, config=config)
            self.transforms = Transforms(self.args.transformers_model, cache_dir=self.cache_dir)

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
                logging.exception("Skip apex: please install apex from https://www.github.com/nvidia/apex to use fp16")

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

    def test(self,
             test_dataset: str = None,
             entity_span_prediction: bool = False,
             lower_case: bool = False,
             batch_size_validation: int = None,
             max_seq_length_validation: int = None):
        """ Test NER model on specific dataset

         Parameter
        -------------
        test_dataset: str
            Dataset to test, alias of preset dataset (see tner.VALID_DATASET) or path to custom dataset folder
        entity_span_prediction: bool
            Test without entity type (entity span detection)
        lower_case: bool
            Converting test data into lower-cased
        """
        # setup model/dataset/data loader
        assert self.args.is_trained, 'finetune model before'
        if test_dataset is None:
            assert len(self.args.dataset) == 1, "test dataset can not be determined"
            dataset = self.args.dataset[0]
        else:
            dataset = test_dataset
        filename = 'test_{}{}{}.json'.format(
            os.path.basename(dataset) if 'panx' not in dataset else dataset.replace('/', '-'),
            '_span' if entity_span_prediction else '',
            '_lower' if lower_case else '')
        filename = os.path.join(self.args.checkpoint_dir, filename)
        if os.path.exists(filename):
            return
        assert type(dataset) is str
        batch_size = batch_size_validation if batch_size_validation else self.args.batch_size
        max_seq_length = max_seq_length_validation if max_seq_length_validation else self.args.max_seq_length
        self.__setup_model_data(dataset, lower_case)

        if 'train' in self.dataset_split.keys():
            self.dataset_split.pop('train')

        data_loader = {k: self.__setup_loader(k, batch_size, max_seq_length) for k in self.dataset_split.keys()}

        logging.info('testing model on {}'.format(dataset))
        logging.info('data_loader: {}'.format(str(list(data_loader.keys()))))

        # run inference
        start_time = time()
        metrics = {}
        params = dict(entity_span_prediction=entity_span_prediction, unseen_entity_set=self.unseen_entity_set)
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
            Display validation result at the end of each epoch
        batch_size_validation: int
            Batch size for validation monitoring
        max_seq_length_validation: int
            Max seq length for validation monitoring
        """
        # setup model/dataset/data loader
        if self.__train_called:
            raise ValueError("`train` can be called once per instant")
        if self.args.is_trained:
            logging.warning('finetuning model, that has been already finetuned')
        self.__setup_model_data(self.args.dataset, self.args.lower_case)
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

        logging.info('[training completed, {} sec in total]'.format(time() - start_time))
        self.model.save_pretrained(self.args.checkpoint_dir)
        self.transforms.tokenizer.save_pretrained(self.args.checkpoint_dir)
        writer.close()
        logging.info('ckpt saved at {}'.format(self.args.checkpoint_dir))
        self.args.is_trained = True
        self.__train_called = True

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

    def __epoch_valid(self, data_loader, prefix, writer=None, unseen_entity_set: set = None,
                      entity_span_prediction: bool = False):
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
                    if entity_span_prediction:
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
