""" Checkpoint versioning based on hyperparameter file """
import os
import hashlib
import json
import shutil
import logging
from glob import glob
from logging.config import dictConfig

import torch

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()

__all__ = 'Argument'


class Argument:
    """ Model training arguments manager """

    def __init__(self, checkpoint: str = None, checkpoint_dir: str = None, **kwargs):
        """  Model training arguments manager

         Parameter
        -------------------
        prefix: prefix to filename
        checkpoint: existing checkpoint name if you want to load
        kwargs: model arguments
        """
        if checkpoint_dir is None:
            checkpoint_dir = '/'.join(checkpoint.split('/')[:-1])
            checkpoint = checkpoint.split('/')[-1]

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir, self.parameter, self.model_statistics, self.label_to_id = self.version(
            kwargs, checkpoint=checkpoint, checkpoint_dir=checkpoint_dir)
        LOGGER.info('checkpoint: %s' % self.checkpoint_dir)
        for k, v in self.parameter.items():
            LOGGER.info(' - [arg] %s: %s' % (k, str(v)))
        self.__dict__.update(self.parameter)

    def remove_ckpt(self):
        shutil.rmtree(self.checkpoint_dir)

    @staticmethod
    def load_ckpt(checkpoint_dir):
        """ load model statistics from pytorch model saved"""
        checkpoint_file = os.path.join(checkpoint_dir, 'model.pt')
        label_id_file = os.path.join(checkpoint_dir, 'label_to_id.json')
        if not os.path.exists(checkpoint_file):
            return None
        LOGGER.info('load ckpt from %s' % checkpoint_file)
        stats = torch.load(checkpoint_file, map_location='cpu')  # allocate stats on cpu
        if os.path.exists(label_id_file):
            label_to_id = json.load(open(label_id_file, 'r'))
            return stats, label_to_id
        else:
            return stats, None

    @staticmethod
    def md5(file_name):
        """ get MD5 checksum """
        hash_md5 = hashlib.md5()
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def version(self, parameter: dict = None, checkpoint: str = None, checkpoint_dir: str = './ckpt'):
        """ Checkpoint version

         Parameter
        ---------------------
        parameter: parameter configuration to find same setting checkpoint
        checkpoint: existing checkpoint to be loaded

         Return
        --------------------
        path_to_checkpoint: path to new checkpoint dir
        parameter: parameter
        """

        if checkpoint is None and parameter is None:
            raise ValueError('either of `checkpoint` or `parameter` is needed.')

        if checkpoint is None:
            LOGGER.info('issue new checkpoint id')
            # check if there are any checkpoints with same hyperparameters
            version_name = []
            for parameter_path in glob(os.path.join(checkpoint_dir, '*/parameter.json')):
                _dir = parameter_path.replace('/parameter.json', '')
                _dict = json.load(open(parameter_path))
                version_name.append(_dir.split('/')[-1])
                if parameter == _dict:
                    inp = input('found a checkpoint with same configuration\n'
                                'enter to delete the existing checkpoint %s\n'
                                'or exit by type anything but not empty' % _dir)
                    if inp == '':
                        shutil.rmtree(_dir)
                    else:
                        exit()

            with open(os.path.join(checkpoint_dir, 'tmp.json'), 'w') as _f:
                json.dump(parameter, _f)
            new_checkpoint = self.md5(os.path.join(checkpoint_dir, 'tmp.json'))
            if 'dataset' in parameter.keys():
                new_checkpoint = '_'.join([parameter['dataset'], new_checkpoint])
            new_checkpoint_dir = os.path.join(checkpoint_dir, new_checkpoint)
            os.makedirs(new_checkpoint_dir, exist_ok=True)
            shutil.move(os.path.join(checkpoint_dir, 'tmp.json'), os.path.join(new_checkpoint_dir, 'parameter.json'))
            return new_checkpoint_dir, parameter, None, None

        else:
            LOGGER.info('load existing checkpoint')
            checkpoints = glob(os.path.join(checkpoint_dir, checkpoint, 'parameter.json'))
            if len(checkpoints) >= 2:
                raise ValueError('Checkpoints are duplicated: %s' % str(checkpoints))
            elif len(checkpoints) == 0:
                raise ValueError('No checkpoint: %s' % os.path.join(checkpoint_dir, checkpoint))
            else:
                parameter = json.load(open(checkpoints[0]))
                target_checkpoints_path = checkpoints[0].replace('/parameter.json', '')
                model_statistics, label_dict = self.load_ckpt(target_checkpoints_path)
                return target_checkpoints_path, parameter, model_statistics, label_dict
