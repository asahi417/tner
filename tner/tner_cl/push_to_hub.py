""" Push Models to Modelhub"""
import json
import os
import argparse
import logging
import shutil
from distutils.dir_util import copy_tree
from os.path import join as pj
from huggingface_hub import create_repo

from tner import TransformersNER
from tner.tner_cl.readme_template import get_readme


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Push to Model hub')
    parser.add_argument('-m', '--model-checkpoint', required=True, type=str)
    parser.add_argument('-a', '--model-alias', required=True, type=str)
    parser.add_argument('-o', '--organization', required=True, type=str)
    parser.add_argument('--use-auth-token', help='Huggingface transformers argument of `use_auth_token`',
                        action='store_true')
    opt = parser.parse_args()

    assert os.path.exists(pj(opt.model_checkpoint, "pytorch_model.bin")), pj(opt.model_checkpoint, "pytorch_model.bin")
    logging.info(f"Upload {opt.model_checkpoint} to {opt.organization}/{opt.model_alias}")

    url = create_repo(opt.model_alias, organization=opt.organization, exist_ok=True)

    model = TransformersNER(opt.model_checkpoint)
    if model.parallel:
        model_ = model.model.module
    else:
        model_ = model.model

    args = {"use_auth_token": opt.use_auth_token, "repo_url": url, "organization": opt.organization}
    model_.push_to_hub(opt.model_alias, **args)
    model_.config.push_to_hub(opt.model_alias, **args)
    model.tokenizer.tokenizer.push_to_hub(opt.model_alias, **args)

    # config
    with open(pj(opt.model_checkpoint, "trainer_config.json")) as f:
        trainer_config = json.load(f)

    # metric
    with open(pj(opt.model_checkpoint, "eval", "metric.json")) as f:
        metric = json.load(f)
    with open(pj(opt.model_checkpoint, "eval", "metric_span.json")) as f:
        metric_span = json.load(f)

    readme = get_readme(
        model_name=f"{opt.organization}/{opt.model_alias}",
        metric=metric,
        metric_span=metric_span,
        config=trainer_config,
    )
    with open(pj(opt.model_checkpoint, "README.md"), 'w') as f:
        f.write(readme)

    # upload remaining files
    copy_tree(f"{opt.model_checkpoint}", f"{opt.model_alias}")
    os.system(f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
    shutil.rmtree(f"{opt.model_alias}")  # clean up the cloned repo