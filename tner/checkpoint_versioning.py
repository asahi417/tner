""" checkpoint versioning tool """
import os
import hashlib
import json
import shutil
import logging
import requests
from glob import glob

__all__ = 'Argument'


class Argument:
    """ Model training arguments manager """

    def __init__(self, checkpoint_dir: str, **kwargs):
        """ Model training arguments manager

         Parameter
        -------------------
        checkpoint_dir: str
            Directory to organize the checkpoint files
        kwargs: model arguments
        """
        assert type(checkpoint_dir) is str
        self.checkpoint_dir, self.parameter, self.is_trained = self.version(checkpoint_dir, parameter=kwargs)
        logging.info('checkpoint: {}'.format(self.checkpoint_dir))
        for k, v in self.parameter.items():
            logging.info(' - [arg] {}: {}'.format(k, str(v)))
        self.__dict__.update(self.parameter)

    @staticmethod
    def md5(file_name):
        """ get MD5 checksum """
        hash_md5 = hashlib.md5()
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def version(self, checkpoint_dir: str, parameter: dict = None):
        """ Checkpoint versioning """
        is_trained = True
        try:
            # load checkpoint on huggingface.transformers that trained with TNER
            url = 'https://huggingface.co/{}/raw/main/parameter.json'.format(parameter['transformers_model'])
            parameter['tner_config'] = json.loads(requests.get(url).content)
            logging.info('load TNER finetuned checkpoint: {}'.format(parameter['transformers_model']))
            checkpoint_dir = self.issue_new_checkpoint(checkpoint_dir, parameter)
            return checkpoint_dir, parameter, is_trained
        except json.JSONDecodeError:
            if os.path.exists(parameter['transformers_model']):
                # load local checkpoint that trained with TNER
                logging.info('load local checkpoint: {}'.format(parameter['transformers_model']))
            else:
                # new check point for finetuning
                is_trained = False
                logging.info('create new checkpoint')

            checkpoint_dir = self.issue_new_checkpoint(checkpoint_dir, parameter)
            return checkpoint_dir, parameter, is_trained

    def issue_new_checkpoint(self, checkpoint_dir, parameter):
        checkpoints = self.cleanup_checkpoint_dir(checkpoint_dir)
        if len(checkpoints) == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            # create a new checkpoint
            with open('{}/parameter.json'.format(checkpoint_dir), 'w') as f:
                json.dump(parameter, f)
            return checkpoint_dir
        else:
            if len(checkpoints) != 0:
                for _dir in checkpoints:
                    with open('{}/parameter.json'.format(_dir), 'r') as f:
                        if parameter == json.load(f):
                            exit('find same configuration at: {}'.format(_dir))
            # create a new checkpoint
            with open('{}/tmp.json'.format(checkpoint_dir), 'w') as f:
                json.dump(parameter, f)
            _id = self.md5('{}/tmp.json'.format(checkpoint_dir))
            new_checkpoint_dir = '{}_{}'.format(checkpoint_dir, _id)
            os.makedirs(new_checkpoint_dir, exist_ok=True)
            shutil.move('{}/tmp.json'.format(checkpoint_dir), '{}/parameter.json'.format(new_checkpoint_dir))
            return new_checkpoint_dir

    @staticmethod
    def cleanup_checkpoint_dir(checkpoint_dir):
        all_dir = glob('{}*'.format(checkpoint_dir))
        if len(all_dir) == 0:
            return []
        for _dir in all_dir:
            if os.path.exists('{}/parameter.json'.format(checkpoint_dir))\
                    and os.path.exists('{}/pytorch_model.bin'.format(checkpoint_dir))\
                    and os.path.exists('{}/tokenizer_config.json'.format(checkpoint_dir)):
                pass
            else:
                logging.info('removed incomplete checkpoint {}'.format(_dir))
                shutil.rmtree(_dir)
        return glob('{}*'.format(checkpoint_dir))
