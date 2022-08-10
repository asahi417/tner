

| Model (link)                                                                                      | Data                                                                | Language Model                                                                    |
|:--------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|:----------------------------------------------------------------------------------|
| [`tner/roberta-large-wnut2017`](https://huggingface.co/tner/roberta-large-wnut2017)               | [`wnut2017`](https://huggingface.co/datasets/tner/wnut2017)         | [`roberta-large`](https://huggingface.co/roberta-large)                           |
| [`tner/deberta-v3-large-wnut2017`](https://huggingface.co/tner/deberta-v3-large-wnut2017)         | [`wnut2017`](https://huggingface.co/datasets/tner/wnut2017)         | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) |
| [`tner/roberta-large-conll2003`](https://huggingface.co/tner/roberta-large-conll2003)             | [`conll2003`](https://huggingface.co/datasets/tner/conll2003)       | [`roberta-large`](https://huggingface.co/roberta-large)                           |
| [`tner/deberta-v3-large-conll2003`](https://huggingface.co/tner/deberta-v3-large-conll2003)       | [`conll2003`](https://huggingface.co/datasets/tner/conll2003)       | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) |
| [`tner/roberta-large-bc5cdr`](https://huggingface.co/tner/roberta-large-bc5cdr)                   | [`bc5cdr`](https://huggingface.co/datasets/tner/bc5cdr)             | [`roberta-large`](https://huggingface.co/roberta-large)                           |
| [`tner/deberta-v3-large-bc5cdr`](https://huggingface.co/tner/deberta-v3-large-bc5cdr)             | [`bc5cdr`](https://huggingface.co/datasets/tner/bc5cdr)             | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) |
| [`tner/roberta-large-tweebank_ner`](https://huggingface.co/tner/roberta-large-tweebank_ner)       | [`tweebank_ner`](https://huggingface.co/datasets/tner/tweebank_ner) | [`roberta-large`](https://huggingface.co/roberta-large)                           |
| [`tner/deberta-v3-large-tweebank_ner`](https://huggingface.co/tner/deberta-v3-large-tweebank_ner) | [`tweebank_ner`](https://huggingface.co/datasets/tner/tweebank_ner) | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) |
| [`tner/roberta-large-btc`](https://huggingface.co/tner/roberta-large-btc)                         | [`btc`](https://huggingface.co/datasets/tner/btc)                   | [`roberta-large`](https://huggingface.co/roberta-large)                           |
| [`tner/deberta-v3-large-btc`](https://huggingface.co/tner/deberta-v3-large-btc)                   | [`btc`](https://huggingface.co/datasets/tner/btc)                   | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) |
| [`tner/roberta-large-bionlp2004`](https://huggingface.co/tner/roberta-large-bionlp2004)           | [`bionlp2004`](https://huggingface.co/datasets/tner/bionlp2004)     | [`roberta-large`](https://huggingface.co/roberta-large)                           |
| [`tner/deberta-v3-large-bionlp2004`](https://huggingface.co/tner/deberta-v3-large-bionlp2004)     | [`bionlp2004`](https://huggingface.co/datasets/tner/bionlp2004)     | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) |
| [`tner/roberta-large-ontonotes5`](https://huggingface.co/tner/roberta-large-ontonotes5)           | [`ontonotes5`](https://huggingface.co/datasets/tner/ontonotes5)     | [`roberta-large`](https://huggingface.co/roberta-large)                           |
| [`tner/deberta-v3-large-ontonotes5`](https://huggingface.co/tner/deberta-v3-large-ontonotes5)     | [`ontonotes5`](https://huggingface.co/datasets/tner/ontonotes5)     | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) |

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
from itertools import product
import pandas as pd

datasets = ["wnut2017", "conll2003", "bc5cdr", "tweebank_ner", "btc", "bionlp2004", "ontonotes5"]
models = [(None, "roberta-large"), ("microsoft", "deberta-v3-large")]
output = []
for d, (org, m) in product(datasets, models):
    if org is not None:
        lm = f"{org}/m"
    else:
        lm = m
    tmp = {
        "Model (link)": f"[`tner/{m}-{d}`](https://huggingface.co/tner/{m}-{d})",
        "Data": f"[`{d}`](https://huggingface.co/datasets/tner/{d})",
        "Language Model": f"[`{lm}`](https://huggingface.co/{lm})",
    }
    output.append(tmp)
df = pd.DataFrame(output)
print(df.to_markdown(index=False))
```
