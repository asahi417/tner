""" Push Models to Modelhub"""
import json
import os
import argparse
import logging
import shutil
from distutils.dir_util import copy_tree
from os.path import join as pj
from tner import TransformersNER
from tner.tner_cl.readme_template import get_readme


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Push to Model hub')
    parser.add_argument('-m', '--model-checkpoint', required=True, type=str)
    parser.add_argument('-a', '--model-alias', required=True, type=str)
    parser.add_argument('-o', '--organization', default=None, type=str)
    opt = parser.parse_args()

    assert os.path.exists(pj(opt.model_checkpoint, "pytorch_model.bin")), pj(opt.model_checkpoint, "pytorch_model.bin")
    logging.info(f"Upload {opt.model_checkpoint} to {opt.organization}/{opt.model_alias}")
    model = TransformersNER(opt.model_checkpoint)
    if model.parallel:
        model_ = model.model.module
    else:
        model_ = model.model
    repo_id = f"{opt.organization}/{opt.model_alias}"
    if opt.organization is None:
        model_.push_to_hub(repo_id=repo_id)
        model_.config.push_to_hub(repo_id=repo_id)
        model.tokenizer.tokenizer.push_to_hub(repo_id=repo_id)
    else:
        model_.push_to_hub(repo_id=repo_id)
        model_.config.push_to_hub(repo_id=repo_id)
        model.tokenizer.tokenizer.push_to_hub(repo_id=repo_id)

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