""" Generate README for TweetNER7 fine-tuned models. """
import os
import json
import shutil
import subprocess

from typing import Dict
from glob import glob


from pprint import pprint
from huggingface_hub import ModelFilter, HfApi


api = HfApi()
filt = ModelFilter(author='tner')
models = [i.modelId for i in api.list_models(filter=filt) if i.modelId.startswith('tner')]
models = [i for i in models if 'tweetner7' in i]
models = sorted(models)
pprint(models)
sample = "Get the all-analog Classic Vinyl Edition of `Takin' Off` Album from {@herbiehancock@} via {@bluenoterecords@} link below: {{URL}}"
sample_raw = "Get the all-analog Classic Vinyl Edition of `Takin' Off` Album from @herbiehancock via @bluenoterecords link below: http://bluenote.lnk.to/AlbumOfTheWeek"
preprocess_function = """
def format_tweet(tweet):
    # mask web urls
    urls = extractor.find_urls(tweet)
    for url in urls:
        tweet = tweet.replace(url, "{{URL}}")
    # format twitter account
    tweet = re.sub(r"\\b(\s*)(@[\S]+)\\b", r'\\1{\\2@}', tweet)
    return tweet
"""


def safe_json_load(_file):
    if os.path.exists(_file):
        with open(_file) as f:
            return json.load(f)
    return None


bib = """
@inproceedings{ushio-camacho-collados-2021-ner,
    title = "{T}-{NER}: An All-Round Python Library for Transformer-based Named Entity Recognition",
    author = "Ushio, Asahi  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-demos.7",
    doi = "10.18653/v1/2021.eacl-demos.7",
    pages = "53--62",
    abstract = "Language model (LM) pretraining has led to consistent improvements in many NLP downstream tasks, including named entity recognition (NER). In this paper, we present T-NER (Transformer-based Named Entity Recognition), a Python library for NER LM finetuning. In addition to its practical utility, T-NER facilitates the study and investigation of the cross-domain and cross-lingual generalization ability of LMs finetuned on NER. Our library also provides a web app where users can get model predictions interactively for arbitrary text, which facilitates qualitative model evaluation for non-expert programmers. We show the potential of the library by compiling nine public NER datasets into a unified format and evaluating the cross-domain and cross- lingual performance across the datasets. The results from our initial experiments show that in-domain performance is generally competitive across datasets. However, cross-domain generalization is challenging even with a large pretrained LM, which has nevertheless capacity to learn domain-specific features if fine- tuned on a combined dataset. To facilitate future research, we also release all our LM checkpoints via the Hugging Face model hub.",
}
"""
bib_tweetner7 = """
@inproceedings{ushio-etal-2022-tweet,
    title = "{N}amed {E}ntity {R}ecognition in {T}witter: {A} {D}ataset and {A}nalysis on {S}hort-{T}erm {T}emporal {S}hifts",
    author = "Ushio, Asahi  and
        Neves, Leonardo  and
        Silva, Vitor  and
        Barbieri, Francesco. and
        Camacho-Collados, Jose",
    booktitle = "The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing",
    month = nov,
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
"""


def get_readme(model_name: str,
               metric_2020: Dict,
               metric_span_2020: Dict,
               metric_2021: Dict,
               metric_span_2021: Dict,
               config: Dict,
               year_sl: str = None,
               model_sl: str = None):
    language_model = config['model']
    dataset = config["dataset"][0]
    dataset_alias = config["dataset"][0]
    config_text = "\n".join([f" - {k}: {v}" for k, v in config.items()])
    ci_micro = '\n'.join([f'    - {k}%: {v}' for k, v in metric_2021["micro/f1_ci"].items()])
    ci_macro = '\n'.join([f'    - {k}%: {v}' for k, v in metric_2021["micro/f1_ci"].items()])
    per_entity_metric = '\n'.join([f'- {k}: {v["f1"]}' for k, v in metric_2021['per_entity_metric'].items()])
    dataset_link = f"[{dataset}](https://huggingface.co/datasets/{dataset})"
    extra_explain = ""
    if 'selflabel' in model_name:
        config["dataset"] = None
        extra_explain += f" This model is fine-tuned on self-labeled dataset which is the `extra_{year_sl}` split of the {dataset_link} annotated by " \
                         f"[{model_sl}](https://huggingface.co/{model_sl}-tweetner7-2020)). " \
                         f"Please check [https://github.com/asahi417/tner/tree/master/examples/tweetner7_paper#model-fine-tuning-self-labeling](https://github.com/asahi417/tner/tree/master/examples/tweetner7_paper#model-fine-tuning-self-labeling) " \
                         f"for more detail of reproducing the model. "
        if model_name.endswith('continuous'):
            extra_explain += " The model is first fine-tuned on `train_2020`, and then continuously fine-tuned on the self-labeled dataset. "
    else:
        if model_name.endswith('continuous'):
            extra_explain = " The model is first fine-tuned on `train_2020`, and then continuously fine-tuned on `train_2021`. "
    return f"""---
datasets:
- {dataset_alias}
metrics:
- f1
- precision
- recall
model-index:
- name: {model_name}
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: {dataset_alias}
      type: {dataset_alias}
      args: {dataset_alias}
    metrics:
    - name: F1 (test_2021)
      type: f1
      value: {metric_2021['micro/f1']}
    - name: Precision (test_2021)
      type: precision
      value: {metric_2021['micro/precision']}
    - name: Recall (test_2021)
      type: recall
      value: {metric_2021['micro/recall']}
    - name: Macro F1 (test_2021)
      type: f1_macro
      value: {metric_2021['macro/f1']}
    - name: Macro Precision (test_2021)
      type: precision_macro
      value: {metric_2021['macro/precision']}
    - name: Macro Recall (test_2021)
      type: recall_macro
      value: {metric_2021['macro/recall']}
    - name: Entity Span F1 (test_2021)
      type: f1_entity_span
      value: {metric_span_2021['micro/f1']}
    - name: Entity Span Precision (test_2020)
      type: precision_entity_span
      value: {metric_span_2021['micro/precision']}
    - name: Entity Span Recall (test_2021)
      type: recall_entity_span
      value: {metric_span_2021['micro/recall']}
    - name: F1 (test_2020)
      type: f1
      value: {metric_2020['micro/f1']}
    - name: Precision (test_2020)
      type: precision
      value: {metric_2020['micro/precision']}
    - name: Recall (test_2020)
      type: recall
      value: {metric_2020['micro/recall']}
    - name: Macro F1 (test_2020)
      type: f1_macro
      value: {metric_2020['macro/f1']}
    - name: Macro Precision (test_2020)
      type: precision_macro
      value: {metric_2020['macro/precision']}
    - name: Macro Recall (test_2020)
      type: recall_macro
      value: {metric_2020['macro/recall']}
    - name: Entity Span F1 (test_2020)
      type: f1_entity_span
      value: {metric_span_2020['micro/f1']}
    - name: Entity Span Precision (test_2020)
      type: precision_entity_span
      value: {metric_span_2020['micro/precision']}
    - name: Entity Span Recall (test_2020)
      type: recall_entity_span
      value: {metric_span_2020['micro/recall']}

pipeline_tag: token-classification
widget:
- text: "{sample}"
  example_title: "NER Example 1"
---
# {model_name}

This model is a fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) on the 
{dataset_link} dataset (`{config["dataset_split"]}` split).{extra_explain}
Model fine-tuning is done via [T-NER](https://github.com/asahi417/tner)'s hyper-parameter search (see the repository
for more detail). It achieves the following results on the test set of 2021:
- F1 (micro): {metric_2021['micro/f1']}
- Precision (micro): {metric_2021['micro/precision']}
- Recall (micro): {metric_2021['micro/recall']}
- F1 (macro): {metric_2021['macro/f1']}
- Precision (macro): {metric_2021['macro/precision']}
- Recall (macro): {metric_2021['macro/recall']}



The per-entity breakdown of the F1 score on the test set are below:
{per_entity_metric} 

For F1 scores, the confidence interval is obtained by bootstrap as below:
- F1 (micro): 
{ci_micro} 
- F1 (macro): 
{ci_macro} 

Full evaluation can be found at [metric file of NER](https://huggingface.co/{model_name}/raw/main/eval/metric.json) 
and [metric file of entity span](https://huggingface.co/{model_name}/raw/main/eval/metric_span.json).

### Usage
This model can be used through the [tner library](https://github.com/asahi417/tner). Install the library via pip.   
```shell
pip install tner
```
[TweetNER7](https://huggingface.co/datasets/tner/tweetner7) pre-processed tweets where the account name and URLs are 
converted into special formats (see the dataset page for more detail), so we process tweets accordingly and then run the model prediction as below.  

```python
import re
from urlextract import URLExtract
from tner import TransformersNER

extractor = URLExtract()
{preprocess_function}

text = "{sample_raw}"
text_format = format_tweet(text)
model = TransformersNER("{model_name}")
model.predict([text_format])
```
It can be used via transformers library but it is not recommended as CRF layer is not supported at the moment.

### Training hyperparameters

The following hyperparameters were used during training:
{config_text}

The full configuration can be found at [fine-tuning parameter file](https://huggingface.co/{model_name}/raw/main/trainer_config.json).

### Reference
If you use the model, please cite T-NER paper and TweetNER7 paper.
- T-NER
```
{bib}
```
- TweetNER7
```
{bib_tweetner7}
```


"""


for i in models:
    print(i)
    # os.system(f"git clone https://huggingface.co/{i}")
    returned_value = subprocess.call(f"git clone https://huggingface.co/{i}", shell=True)
    print(returned_value)
    for x in glob(f"{os.path.basename(i)}/eval/prediction.*.json"):
        os.remove(x)

    trainer_config = safe_json_load(f"{os.path.basename(i)}/trainer_config.json")
    if 'selflabel' in i:
        year = i.split('selflabel')[1][:4]
        _model_sl = i.split("-tweetner7")[0]
    else:
        year = None
        _model_sl = None

    if "dataset" in trainer_config:
        pass
    else:
        local_dataset = None
        datasets = ["tner/tweetner7"]
        if trainer_config['data_split'] == '2020.train':
            dataset_split = 'train_2020'
        elif trainer_config['data_split'] == '2020_2021.train':
            dataset_split = 'train_all'
        elif trainer_config['data_split'] == '2021.train':
            dataset_split = 'train_2021'
        elif trainer_config['data_split'] == 'random.train':
            dataset_split = 'train_random'
        elif 'selflabel' in i:

            if i.endswith('all'):
                local_dataset = {"train": f"tweet_ner/2020_{year}.extra.{_model_sl}-2020.txt",
                                 "validation": "tweet_ner/2020.dev.txt"}
            else:
                local_dataset = {"train": f"tweet_ner/{year}.extra.{_model_sl}-2020.txt",
                                 "validation": "tweet_ner/2020.dev.txt"}
            dataset_split = "train"
        else:
            raise ValueError(f"unknown split: {trainer_config['dataset_split']}")

        trainer_config = {
            "dataset": datasets,
            "dataset_split": dataset_split,
            "dataset_name": None,
            "local_dataset": local_dataset,
            "model": trainer_config["model"],
            "crf": trainer_config["crf"],
            "max_length": trainer_config["max_length"],
            "epoch": trainer_config["epoch"],
            "batch_size": trainer_config["batch_size"],
            "lr": trainer_config["lr"],
            "random_seed": trainer_config["random_seed"],
            "gradient_accumulation_steps": trainer_config["gradient_accumulation_steps"],
            "weight_decay": trainer_config["weight_decay"],
            "lr_warmup_step_ratio": trainer_config["lr_warmup_step_ratio"],
            "max_grad_norm": trainer_config["max_grad_norm"]
        }
        with open(f"{os.path.basename(i)}/trainer_config.json", "w") as f:
            json.dump(trainer_config, f)

    if os.path.exists(f"{os.path.basename(i)}/eval/metric.json"):
        metric_ = safe_json_load(f"{os.path.basename(i)}/eval/metric.json")
        os.remove(f"{os.path.basename(i)}/eval/metric.json")
        with open(f"{os.path.basename(i)}/eval/metric.test_2020.json", "w") as f:
            json.dump(metric_["2020.test"], f)
        with open(f"{os.path.basename(i)}/eval/metric.test_2021.json", "w") as f:
            json.dump(metric_["2021.test"], f)
        with open(f"{os.path.basename(i)}/eval/metric_span.test_2020.json", "w") as f:
            json.dump(metric_["2020.test (span detection)"], f)
        with open(f"{os.path.basename(i)}/eval/metric_span.test_2021.json", "w") as f:
            json.dump(metric_["2021.test (span detection)"], f)

    metric_2020_ = safe_json_load(f"{os.path.basename(i)}/eval/metric.test_2020.json")
    metric_span_2020_ = safe_json_load(f"{os.path.basename(i)}/eval/metric_span.test_2020.json")
    metric_2021_ = safe_json_load(f"{os.path.basename(i)}/eval/metric.test_2021.json")
    metric_span_2021_ = safe_json_load(f"{os.path.basename(i)}/eval/metric_span.test_2021.json")
    with open(f"{os.path.basename(i)}/README.md", "w") as f:
        readme = get_readme(
            model_name=i,
            metric_2020=metric_2020_,
            metric_span_2020=metric_span_2020_,
            metric_2021=metric_2021_,
            metric_span_2021=metric_span_2021_,
            config=trainer_config,
            year_sl=year,
            model_sl=_model_sl
        )
        f.write(readme)
    returned_value = subprocess.call(f"cd {os.path.basename(i)} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../", shell=True)
    print(returned_value)
    shutil.rmtree(os.path.basename(i))
    assert not os.path.exists(os.path.basename(i))

