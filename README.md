[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/tner/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/tner.svg)](https://badge.fury.io/py/tner)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tner.svg)](https://pypi.python.org/pypi/tner/)
[![PyPI status](https://img.shields.io/pypi/status/tner.svg)](https://pypi.python.org/pypi/tner/)


<p align="left">
  <img src="https://github.com/asahi417/tner/blob/master/asset/tner_logo_horizontal.png" width="350">
</p>


# T-NER: An All-Round Python Library for Transformer-based Named Entity Recognition  

***T-NER*** is a python tool for language model finetuning on named-entity-recognition (NER) implemented in pytorch, available via [pip](https://pypi.org/project/tner/). 
It has an easy interface to finetune models and test on cross-domain and multilingual datasets. T-NER currently integrates 9 publicly available NER datasets and enables an easy integration of custom datasets.
All models finetuned with T-NER can be deployed on our web app for visualization.
[Our paper demonstrating T-NER](https://www.aclweb.org/anthology/2021.eacl-demos.7/) has been accepted to EACL 2021.
All the models and datasets are shared via [T-NER HuggingFace group](https://huggingface.co/tner).

- GitHub: [https://github.com/asahi417/tner](https://github.com/asahi417/tner)
- Paper: [https://aclanthology.org/2021.eacl-demos.7/](https://aclanthology.org/2021.eacl-demos.7/)
- HuggingFace: [https://huggingface.co/tner](https://huggingface.co/tner)
- Pypi: [https://pypi.org/project/tner](https://pypi.org/project/tner)

## Table of Contents  
1. **[Setup](#setup)**
2. **[Dataset](#dataset)**
   2.1 **[HuggingFace Dataset](#huggingface-dataset)**
   2.2 **[Custom Dataset](#custom-dataset)**
3. **[Model](#model)**
3. **[Pretrained Models](https://github.com/asahi417/tner/blob/master/MODEL_CARD.md)**
3. **[Model Finetuning](#model-finetuning)**
5. **[Model Evaluation](#model-evaluation)**
6. **[Model Inference](#model-inference)** 
7. **[Datasets](#datasets)**
2. **[Web API](#web-app)**
8. **[Reference](#reference-paper)**
9. **[Colab Examples](#google-colab-examples)**


## Setup
Install pip package.
```shell script
pip install tner
```
To install dependencies to run the web app, add option at installation.
```shell script
pip install tner[app]
```

## Dataset
An NER dataset contains a sequence of tokens and tags for each split (usually `train`/`validation`/`test`),
```python
{
    'train': {
        'tokens': [
            ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.'],
            ['From', 'Green', 'Newsfeed', ':', 'AHFA', 'extends', 'deadline', 'for', 'Sage', 'Award', 'to', 'Nov', '.', '5', 'http://tinyurl.com/24agj38'], ...
        ],
        'tags': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...
        ]
    },
    'validation': ...,
    'test': ...,
}
```
with a dictionary to map a label to its index (`label2id`) as below.
```python
{"O": 0, "B-ORG": 1, "B-MISC": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-ORG": 6, "I-MISC": 7, "I-LOC": 8}
```

### HuggingFace Dataset

| Dataset           | Alias (link)                                                                      | Domain                                  | Size (train/valid/test) | Language | Entity Size |
|-------------------|-----------------------------------------------------------------------------------|-----------------------------------------|-------------------------|----------|-------------|
| Ontonotes5        | [`tner/ontonotes5`](https://huggingface.co/datasets/tner/ontonotes5)              | News, Blog, Dialogue                    | 59,924/8,528/8,262      | en       | 18          |
| CoNLL2003         | [`tner/conll2003`](https://huggingface.co/datasets/tner/conll2003)                | News                                    | 14,041/3,250/3,453      | en       | 4           |
| WNUT2017          | [`tner/wnut2017`](https://huggingface.co/datasets/tner/wnut2017)                  | Twitter, Reddit, StackExchange, YouTube | 2,395/1,009/1,287       | en       | 6           |
| BioNLP2004        | [`tner/bionlp2004`](https://huggingface.co/datasets/tner/bionlp2004)              | Biochemical                             | 16,619/1,927/3,856      | en       | 5           |
| BioCreative V CDR | [`tner/bc5cdr`](https://huggingface.co/datasets/tner/bc5cdr)                      | Biomedical                              | 5,228/5,330/5,865       | en       | 2           |
| FIN               | [`tner/fin`](https://huggingface.co/datasets/tner/fin)                            | Financial News                          | 1,014/303/150           | en       | 4           |
| BTC               | [`tner/btc`](https://huggingface.co/datasets/tner/btc)                            | Twitter                                 | 1,014/303/150           | en       | 3           |
| Tweebank NER      | [`tner/tweebank_ner`](https://huggingface.co/datasets/tner/tweebank_ner)          | Twitter                                 | 1,639/710/1,201         | en       | 4           |
| MIT Movie         | [`tner/mit_movie_trivia` ](https://huggingface.co/datasets/tner/mit_movie_trivia) | Movie Review                            | 6,816/1,000/1,953       | en       | 12          |
| MIT Restaurant    | [`tner/mit_restaurant`](https://huggingface.co/datasets/tner/mit_restaurant)      | Restaurant Review                       | 6,900/760/1,521         | en       | 8           |

A variety of public NER datasets are shared on our [HuggingFace group](https://huggingface.co/tner), which can be used as below. 
```python
from tner import get_dataset
data, label2id = get_dataset("tner/wnut2017")
```
The idea is to share all the available NER datasets on the HuggingFace in a unified format, so let us know if you want any NER datasets to be added there!  

### Custom Dataset
To go beyond the public datasets, users can use their own datasets by formatting them into
the IOB format described in [CoNLL 2003 NER shared task paper](https://www.aclweb.org/anthology/W03-0419.pdf),
where all data files contain one word per line with empty lines representing sentence boundaries.
At the end of each line there is a tag which states whether the current word is inside a named entity or not.
The tag also encodes the type of named entity. Here is an example sentence:
```
EU B-ORG
rejects O
German B-MISC
call O
to O
boycott O
British B-MISC
lamb O
. O
```
Words tagged with O are outside of named entities and the I-XXX tag is used for words inside a
named entity of type XXX. Whenever two entities of type XXX are immediately next to each other, the
first word of the second entity will be tagged B-XXX in order to show that it starts another entity.
Please take a look [sample custom data](https://github.com/asahi417/tner/tree/master/examples/local_dataset_sample).
Those custom files can be loaded in a same way as HuggingFace dataset as below.
```python
from tner import get_dataset
data, label2id = get_dataset(local_dataset={
    "train": "examples/local_dataset_sample/train.txt",
    "valid": "examples/local_dataset_sample/train.txt",
    "test": "examples/local_dataset_sample/test.txt"
})
```

## Model
### HuggingFace Models

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

T-NER currently has shared more than 100 NER models on [HuggingFace group](https://huggingface.co/tner) (see above table for examples) and all the models can be used with `tner` as below.
```python
from tner import TransformersNER
model = TransformersNER("tner/roberta-large-wnut2017")  # provide model alias on huggingface
model.predict(['I live in United States, but Microsoft asks me to move to Japan.'.split(" ")])
```
These models can be used through the transformers library by 
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("tner/roberta-large-wnut2017")
model = AutoModelForTokenClassification.from_pretrained("tner/roberta-large-wnut2017")
```
but, since transformers do not support CRF layer, it is recommended to use the model via `T-NER` library.

### Model Fine-tuning

<p align="center">
  <img src="https://github.com/asahi417/tner/blob/master/asset/parameter_search.png" width="800">
</p>

Language model finetuning on NER can be done with a few lines:
```python
import tner
trainer = tner.Trainer(checkpoint_dir='./ckpt_tner', dataset="data-name", model="transformers-model")
trainer.train()
```
where `transformers_model` is a pre-trained model name from [transformers model hub](https://huggingface.co/models) and
`dataset` is a dataset alias or path to custom dataset explained [dataset section](#datasets).
[Model files](https://huggingface.co/transformers/model_sharing.html#check-the-directory-before-pushing-to-the-model-hub) will be generated at `checkpoint_dir`, and it can be uploaded to transformers model hub without any changes.

To show validation accuracy at the end of each epoch,
```python
trainer.train(monitor_validation=True)
```
and to tune training parameters such as batch size, epoch, learning rate, please take a look [the argument description](https://github.com/asahi417/tner/blob/master/tner/model.py#L47).

***Train on multiple datasets:*** Model can be trained on a concatenation of multiple datasets by providing a list of dataset names.
```python
trainer = tner.TrainTransformersNER(checkpoint_dir='./ckpt_merged', dataset=["ontonotes5", "conll2003"], transformers_model="xlm-roberta-base")
```
[Custom datasets](#custom-dataset) can be also added to it, e.g. `dataset=["ontonotes5", "./examples/custom_data_sample"]`.

***Command line tool:*** Finetune models with the command line (CL).
```shell script
tner-train [-h] [-c CHECKPOINT_DIR] [-d DATA] [-t TRANSFORMER] [-b BATCH_SIZE] [--max-grad-norm MAX_GRAD_NORM] [--max-seq-length MAX_SEQ_LENGTH] [--random-seed RANDOM_SEED] [--lr LR] [--total-step TOTAL_STEP] [--warmup-step WARMUP_STEP] [--weight-decay WEIGHT_DECAY] [--fp16] [--monitor-validation] [--lower-case]
```


## Web App

<p align="center">
  <img src="https://github.com/asahi417/tner/blob/master/asset/api.gif" width="500">
</p>

To start the web app, first clone the repository
```shell script
git clone https://github.com/asahi417/tner
cd tner
```
then launch the server by
```shell script
uvicorn app:app --reload --log-level debug --host 0.0.0.0 --port 8000
```
and open your browser http://0.0.0.0:8000 once ready.
You can specify model to deploy by an environment variable `NER_MODEL`, which is set as `asahi417/tner-xlm-roberta-large-ontonotes5` as a defalt. 
`NER_MODEL` can be either path to your local model checkpoint directory or model name on transformers model hub.

***Acknowledgement*** The App interface is heavily inspired by [this repository](https://github.com/renatoviolin/Multiple-Choice-Question-Generation-T5-and-Text2Text).



## Model Evaluation
Evaluation of NER models is easily done for in/out of domain settings.
```python
import tner
trainer = tner.TrainTransformersNER(checkpoint_dir='path-to-checkpoint', transformers_model="language-model-name")
trainer.test(test_dataset='data-name')
```

***Entity span prediction:***  For better understanding of out-of-domain accuracy, we provide the entity span prediction
pipeline, which ignores the entity type and compute metrics only on the IOB entity position.
```python
trainer.test(test_dataset='data-name', entity_span_prediction=True)
```

***Command line tool:*** Model evaluation with CL.
```shell script
tner-test [-h] -c CHECKPOINT_DIR [--lower-case] [--test-data TEST_DATA] [--test-lower-case] [--test-entity-span]
```

## Model Inference
If you just want a prediction from a finetuned NER model, here is the best option for you.
```python
import tner
classifier = tner.TransformersNER('transformers-model')
test_sentences = [
    'I live in United States, but Microsoft asks me to move to Japan.',
    'I have an Apple computer.',
    'I like to eat an apple.'
]
classifier.predict(test_sentences)
```
***Command line tool:*** Model inference with CL.
```shell script
tner-predict [-h] [-c CHECKPOINT]
```

## Google Colab Examples
| Description               | Link  |
|---------------------------|-------|
| Model Finetuning          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AlcTbEsp8W11yflT7SyT0L4C4HG6MXYr?usp=sharing) |
| Model Evaluation          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jHVGnFN4AU8uS-ozWJIXXe2fV8HUj8NZ?usp=sharing) |
| Model Prediction          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mQ_kQWeZkVs6LgV0KawHxHckFraYcFfO?usp=sharing) |
| Multilingual NER Workflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mq0UisC2dwlVMP9ar2Cf6h5b1T-7Tdwb?usp=sharing) |

## Reference paper
If you use any of these resources, please cite the following [paper](https://aclanthology.org/2021.eacl-demos.7/):
```
@inproceedings{ushio-camacho-collados-2021-ner,
    title = "{T}-{NER}: An All-Round Python Library for Transformer-based Named Entity Recognition",
    author = "Ushio, Asahi  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-demos.7",
    pages = "53--62",
    abstract = "Language model (LM) pretraining has led to consistent improvements in many NLP downstream tasks, including named entity recognition (NER). In this paper, we present T-NER (Transformer-based Named Entity Recognition), a Python library for NER LM finetuning. In addition to its practical utility, T-NER facilitates the study and investigation of the cross-domain and cross-lingual generalization ability of LMs finetuned on NER. Our library also provides a web app where users can get model predictions interactively for arbitrary text, which facilitates qualitative model evaluation for non-expert programmers. We show the potential of the library by compiling nine public NER datasets into a unified format and evaluating the cross-domain and cross- lingual performance across the datasets. The results from our initial experiments show that in-domain performance is generally competitive across datasets. However, cross-domain generalization is challenging even with a large pretrained LM, which has nevertheless capacity to learn domain-specific features if fine- tuned on a combined dataset. To facilitate future research, we also release all our LM checkpoints via the Hugging Face model hub.",
}
```
