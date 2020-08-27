""" GCP wrapper """
import os
import zipfile
import logging
from logging.config import dictConfig
from google.cloud import storage


__all__ = ["upload", "download"]

dictConfig({
    "version": 1,
    "formatters": {'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}},
    "handlers": {'h': {'class': 'logging.StreamHandler', 'formatter': 'f', 'level': logging.DEBUG}},
    "root": {'handlers': ['h'], 'level': logging.DEBUG}})
LOGGER = logging.getLogger()


def upload(bucket_name: str, source_file_name: str, destination_blob_name: str):
    """Uploads a file to the gcloud bucket
     Parameter
    ------------
    bucket_name: str
        bucket name
    source_blob_name: str
        gstorage blob name
    destination_file_name: str
        file name to download
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    LOGGER.info("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def download(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """Downloads a blob from the gstorage bucket
     Parameter
    ------------
    source_blob_name: str
        gstorage blob name
    destination_file_name: str
        file name to download
    bucket_name: str
        bucket name
    """
    if os.path.exists(destination_file_name):
        LOGGER.debug('%s exists, will be overridden' % destination_file_name)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    LOGGER.debug('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))

    if destination_file_name.endswith('.zip'):
        target_dir = '/'.join(destination_file_name.split('/')[:-1])
        LOGGER.debug('unzip %s to %s' % (destination_file_name, target_dir))
        with zipfile.ZipFile(destination_file_name, "r") as zip_ref:
            zip_ref.extractall(target_dir)


if __name__ == '__main__':
    # upload(bucket_name='nlp-entity-recognition',
    #        source_file_name='./README.md',
    #        destination_blob_name='test/README.md')
    download(bucket_name='nlp-entity-recognition',
             destination_file_name='./cache/ner-cogent-en.zip',
             source_blob_name='ner-cogent-en.zip')