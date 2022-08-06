# Work in Progress

# Released Model
We release 46 finetuned models on [transformers model hub](https://huggingface.co/models?search=asahi417/tner).
All the models are [XLM-R](https://arxiv.org/pdf/1911.02116.pdf), finetuned on named entity recognition task with T-NER.
*Please take a look our paper for the evaluation results including out-of-domain accuracy ([paper link](https://github.com/asahi417/tner/blob/master/asset/2021_01_EACL_TNER.pdf))*.

## Model Name
Model name is organized as `asahi417/tner-xlm-roberta-{model_type}-{dataset}`, where `model_type` is either `base` or `large` and `dataset` corresponds to 
the alias of [dataset](https://github.com/asahi417/tner/blob/master/README.md#datasets). In addition to each individual model, we train on the English merged dataset by 
concatenating all the English NER dataset, that denoted as `all-english`.
We also release model finetuned on lowercased dataset, which is called `asahi417/tner-xlm-roberta-{model_type}-uncased-{dataset}`.

For example
- `asahi417/tner-xlm-roberta-large-ontonotes5`: XLM-R large model finetuned on Ontonotes5 dataset
- `asahi417/tner-xlm-roberta-base-uncased-conll2003`: XLM-R base model finetuned on lowercased CoNLL2003 dataset
- `asahi417/tner-xlm-roberta-large-all-english`: XLM-R large model finetuned on all English datasets

The list of all public models can be checked [here](https://huggingface.co/models?search=asahi417/tner).
The training parameter used in TNER to finetune each model, is stored at `https://huggingface.co/{model-name}/blob/main/parameter.json`.
Eg) The training parameter of `asahi417/tner-xlm-roberta-large-all-english` is [here](https://huggingface.co/asahi417/tner-xlm-roberta-large-all-english/blob/main/parameter.json).

## Usage
### To use with TNER

```python
import tner
model = tner.TransformersNER("model-name")
```

### To use with transformers
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForTokenClassification.from_pretrained("model-name")
```

