[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/tner/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/tner.svg)](https://badge.fury.io/py/tner)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tner.svg)](https://pypi.python.org/pypi/tner/)
[![PyPI status](https://img.shields.io/pypi/status/tner.svg)](https://pypi.python.org/pypi/tner/)

# T-NER  
<p align="center">
  <img src="https://github.com/asahi417/tner/blob/master/asset/api.gif" width="600">
</p>


***T-NER*** is a python tool for language model finetuning on named-entity-recognition (NER), available via [pip](https://pypi.org/project/tner/). 
It has an easy interface to finetune models and test on cross-domain and multilingual datasets. T-NER currently integrates 9 publicly available NER datasets and enables an easy integration of custom datasets.
All models finetuned with T-NER can be deploy on our web app for visualization.

***Paper Accepted:*** Our paper demonstrating T-NER has been accepted to EACL 2021 🎉 
Paper [here](https://github.com/asahi417/tner/blob/master/asset/2021_01_EACL_TNER.pdf).

***PreTrained Models:*** We release 46 XLM-RoBERTa models finetuned on NER on the HuggingFace transformers model hub, [see here for more details and model cards](https://github.com/asahi417/tner/blob/master/MODEL_CARD.md).

### Table of Contents  
1. **[Setup](#get-started)**
2. **[Web API](#web-app)**
3. **[Pretrained Models](https://github.com/asahi417/tner/blob/master/MODEL_CARD.md)**
4. **[Model Finetuning](#model-finetuning)**
5. **[Model Evaluation](#model-evaluation)**
6. **[Model Inference](#model-inference)** 
7. **[Datasets](#datasets)**
8. **[Reference](#reference-paper)**

### Google Colab Examples
| Description               | Link  |
|---------------------------|-------|
| Model Finetuning          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AlcTbEsp8W11yflT7SyT0L4C4HG6MXYr?usp=sharing) |
| Model Evaluation          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jHVGnFN4AU8uS-ozWJIXXe2fV8HUj8NZ?usp=sharing) |
| Model Prediction          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mQ_kQWeZkVs6LgV0KawHxHckFraYcFfO?usp=sharing) |
| Multilingual NER Workflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mq0UisC2dwlVMP9ar2Cf6h5b1T-7Tdwb?usp=sharing) |

## Get Started
Install pip package
```shell script
pip install tner
```
or directly from the repository for the latest version.
```shell script
pip install git+https://github.com/asahi417/tner
```

## Web App
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


## Model Finetuning
Language model finetuning on NER can be done with a few lines:
```python
import tner
trainer = tner.TrainTransformersNER(checkpoint_dir='./ckpt_tner', dataset="data-name", transformers_model="transformers-model")
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

## Datasets
Public datasets that can be fetched with TNER are summarized here. Please cite the corresponding reference if using one of these datasets.

|                                   Name (`alias`)                                                                      |         Genre        |    Language   | Entity types | Data size (train/valid/test) | Note |
|:---------------------------------------------------------------------------------------------------------------------:|:--------------------:|:-------------:|:------------:|:--------------------:|:-----------:|
| OntoNotes 5 ([`ontonotes5`](https://www.aclweb.org/anthology/N06-2015.pdf))                                           | News, Blog, Dialogue | English       |           18 |   59,924/8,582/8,262 |  | 
| CoNLL 2003 ([`conll2003`](https://www.aclweb.org/anthology/W03-0419.pdf))                                             | News                 | English       |            4 |   14,041/3,250/3,453 |  |
| WNUT 2017 ([`wnut2017`](https://noisy-text.github.io/2017/pdf/WNUT18.pdf))                                            | SNS                  | English       |            6 |    1,000/1,008/1,287 |  |
| FIN ([`fin`](https://www.aclweb.org/anthology/U15-1010.pdf))                                                          | Finance              | English       |            4 |          1,164/-/303 |  |
| BioNLP 2004 ([`bionlp2004`](https://www.aclweb.org/anthology/W04-1213.pdf))                                           | Chemical             | English       |            5 |       18,546/-/3,856 |  |
| BioCreative V CDR ([`bc5cdr`](https://biocreative.bioinformatics.udel.edu/media/store/files/2015/BC5CDRoverview.pdf)) | Medical              | English       |            2 |    5,228/5,330/5,865 | split into sentences to reduce sequence length |
| WikiAnn ([`panx_dataset/en`, `panx_dataset/ja`, etc](https://www.aclweb.org/anthology/P17-1178.pdf))                  | Wikipedia            | 282 languages |            3 | 20,000/10,000/10,000 |  |
| Japanese Wikipedia ([`wiki_ja`](https://github.com/Hironsan/IOB2Corpus))                                              | Wikipedia            | Japanese      |            8 |              -/-/500 | test set only |
| Japanese WikiNews ([`wiki_news_ja`](https://github.com/Hironsan/IOB2Corpus))                                          | Wikipedia            | Japanese      |           10 |            -/-/1,000 | test set only |
| MIT Restaurant ([`mit_restaurant`](https://groups.csail.mit.edu/sls/downloads/))                                      | Restaurant review    | English       |            8 |        7,660/-/1,521 | lower-cased |
| MIT Movie ([`mit_movie_trivia`](https://groups.csail.mit.edu/sls/downloads/))                                         | Movie review         | English       |           12 |        7,816/-/1,953 | lower-cased |


To take a closer look into each dataset, one may want to use `tner.get_dataset_ner` as in
```python
import tner
data, label_to_id, language, unseen_entity_set = tner.get_dataset_ner('data-name')
```
where `data` consists of the following structured format.
```
{
    'train': {
        'data': [
            ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.'],
            ['From', 'Green', 'Newsfeed', ':', 'AHFA', 'extends', 'deadline', 'for', 'Sage', 'Award', 'to', 'Nov', '.', '5', 'http://tinyurl.com/24agj38'], ...
        ],
        'label': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...
        ]
    },
    'valid': ...
}
```

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
The custom dataset should have `train.txt` and `valid.txt` files in a same folder. 
Please take a look [sample custom data](https://github.com/asahi417/tner/tree/master/examples/custom_dataset_sample).


## Reference paper
If you use any of these resources, please cite the following [paper](https://github.com/asahi417/tner/blob/master/asset/2021_01_EACL_TNER.pdf):
```
@InProceedings{ushio2021tner,
  author    = "Ushio, Asahi and Camacho-Collados, Jose",
  title     = "T-NER: An All-Round Python Library for Transformer-based Named Entity Recognition",
  booktitle = "Proceedings of EACL: System Demonstrations",
  year      = "2021"
  }
```
