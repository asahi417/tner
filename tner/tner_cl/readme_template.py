import os
from typing import Dict

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


def get_readme(model_name: str,
               metric: Dict,
               metric_span: Dict,
               config: Dict):
    language_model = config['model']
    dataset = None
    dataset_alias = "custom"
    if config["dataset"] is not None:
        dataset = sorted([i for i in config["dataset"]])
        dataset_alias = ','.join(dataset)
    config_text = "\n".join([f" - {k}: {v}" for k, v in config.items()])
    ci_micro = '\n'.join([f'    - {k}%: {v}' for k, v in metric["micro/f1_ci"].items()])
    ci_macro = '\n'.join([f'    - {k}%: {v}' for k, v in metric["micro/f1_ci"].items()])
    per_entity_metric = '\n'.join([f'- {k}: {v["f1"]}' for k, v in metric['per_entity_metric'].items()])
    if dataset is None:
        dataset_link = 'custom'
    else:
        dataset = [dataset] if type(dataset) is str else dataset
        dataset_link = ','.join([f"[{d}](https://huggingface.co/datasets/{d})" for d in dataset])
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
    - name: F1
      type: f1
      value: {metric['micro/f1']}
    - name: Precision
      type: precision
      value: {metric['micro/precision']}
    - name: Recall
      type: recall
      value: {metric['micro/recall']}
    - name: F1 (macro)
      type: f1_macro
      value: {metric['macro/f1']}
    - name: Precision (macro)
      type: precision_macro
      value: {metric['macro/precision']}
    - name: Recall (macro)
      type: recall_macro
      value: {metric['macro/recall']}
    - name: F1 (entity span)
      type: f1_entity_span
      value: {metric_span['micro/f1']}
    - name: Precision (entity span)
      type: precision_entity_span
      value: {metric_span['micro/precision']}
    - name: Recall (entity span)
      type: recall_entity_span
      value: {metric_span['micro/recall']}

pipeline_tag: token-classification
widget:
- text: "Jacob Collier is a Grammy awarded artist from England."
  example_title: "NER Example 1"
---
# {model_name}

This model is a fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) on the 
{dataset_link} dataset.
Model fine-tuning is done via [T-NER](https://github.com/asahi417/tner)'s hyper-parameter search (see the repository
for more detail). It achieves the following results on the test set:
- F1 (micro): {metric['micro/f1']}
- Precision (micro): {metric['micro/precision']}
- Recall (micro): {metric['micro/recall']}
- F1 (macro): {metric['macro/f1']}
- Precision (macro): {metric['macro/precision']}
- Recall (macro): {metric['macro/recall']}

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
This model can be used through the [tner library](https://github.com/asahi417/tner). Install the library via pip   
```shell
pip install tner
```
and activate model as below.
```python
from tner import TransformersNER
model = TransformersNER("{model_name}")
model.predict(["Jacob Collier is a Grammy awarded English artist from London"])
```
It can be used via transformers library but it is not recommended as CRF layer is not supported at the moment.

### Training hyperparameters

The following hyperparameters were used during training:
{config_text}

The full configuration can be found at [fine-tuning parameter file](https://huggingface.co/{model_name}/raw/main/trainer_config.json).

### Reference
If you use any resource from T-NER, please consider to cite our [paper](https://aclanthology.org/2021.eacl-demos.7/).

```
{bib}
```
"""
