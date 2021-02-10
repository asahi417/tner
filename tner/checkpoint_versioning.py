""" Checkpoint versioning based on hyperparameter file """
import os
import hashlib
import json
import shutil
import logging
from glob import glob
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

__all__ = 'Argument'


class Argument:
    """ Model training arguments manager """

    def __init__(self, checkpoint_dir: str = './ckpt/ner_model', **kwargs):
        """ Model training arguments manager

         Parameter
        -------------------
        checkpoint_dir: str
            Directory to organize the checkpoint files
        kwargs: model arguments
        """
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
        """ Checkpoint version

         Parameter
        ---------------------
        parameter: parameter configuration to find same setting checkpoint
        checkpoint: existing checkpoint to be loaded
        read: read mode, otherwise write

         Return
        --------------------
        path_to_checkpoint: path to new checkpoint dir
        parameter: parameter
        """

        if parameter is None or 'dataset' not in parameter.keys():
            assert os.path.exists(checkpoint_dir), 'no checkpoint at {}'.format(checkpoint_dir)
            logging.info('load existing checkpoint')
            config = '{}/{}'.format(checkpoint_dir, 'parameter.json')
            assert os.path.exists(config), 'config file not found at: {}'.format(config)
            with open(config, 'r') as f:
                parameter = json.load(f)
            return checkpoint_dir, parameter, True
        else:
            assert parameter is not None, 'no configurations are provided'
            logging.info('create new checkpoint')
            checkpoints = self.cleanup_checkpoint_dir(checkpoint_dir)
            if len(checkpoints) == 0:
                return checkpoint_dir, parameter, False
            for _dir in checkpoints:
                with open('{}/parameter.json'.format(_dir), 'r') as f:
                    if parameter == json.load(f):
                        exit('find same configuration at: {}'.format(_dir))
            # create a new checkpoint
            with open('{}/tmp.json'.format(checkpoint_dir), 'w') as f:
                json.dump(parameter, f)
            _id = self.md5('{}/tmp.json'.format(checkpoint_dir))
            new_checkpoint_dir = '{}_{}'.format(checkpoint_dir, _id)
            shutil.move('{}/tmp.json'.format(checkpoint_dir), '{}/parameter.json'.format(new_checkpoint_dir))
            return new_checkpoint_dir, parameter, False

    @staticmethod
    def cleanup_checkpoint_dir(checkpoint_dir):
        all_dir = glob('{}*'.format(checkpoint_dir))
        if len(all_dir) == 0:
            os.makedirs(checkpoint_dir)
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
