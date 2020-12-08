# T-NER: Transformers NER  

![](./asset/api.png)

***`T-NER`*** is a python tool to analyse language model finetuning for named-entity-recognition.  
 
### Table of Contents  
1. **[Setup](#get-started)**
2. **[Language Model Finetuning on NER](#language-model-finetuning-on-ner)**
    - *[Datasets](#datasets):* Built-in datasets and custom dataset
    - *[Model Finetuning](#model-finetuning):* Model training [colab notebook](https://colab.research.google.com/drive/1AlcTbEsp8W11yflT7SyT0L4C4HG6MXYr?usp=sharing)
    - *[Model Evaluation](#model-evaluation):* In/out of domain evaluation [colab notebook](https://colab.research.google.com/drive/1jHVGnFN4AU8uS-ozWJIXXe2fV8HUj8NZ?usp=sharing)
    - *[Model Inference API](#model-inference):* An API to get prediction from models
3. **[Experiment with XLM-R](#experiment-with-xlm-r):** Cross-domain analysis on XLM-R
4. **[Web API](#web-app):** Model deployment on a web-app   

## Get Started
Install via pip
```shell script
pip install git+https://github.com/asahi417/tner
```

or clone and install libraries.
```shell script
git clone https://github.com/asahi417/tner
cd tner
pip install -r requirement.txt
```

## Language Model Finetuning on NER

<p align="center">
  <img src="./asset/tb_valid.png" width="600">
  <br><i>Fig 1: Tensorboard visualization</i>
</p>

### Datasets
Following built-in NER datasets are available via `tner`.   

|                                   Name (`alias`)                                 |         Genre        |    Language   | Entity types |       Data size      | Lower-cased |
|:--------------------------------------------------------------------------------:|:--------------------:|:-------------:|:------------:|:--------------------:|:-----------:|
| OntoNote 5 ([`ontonote5`](https://www.aclweb.org/anthology/N06-2015.pdf))        | News, Blog, Dialogue |    English    |           18 |   59,924/8,582/8,262 | No | 
| CoNLL 2003 ([`conll2003`](https://www.aclweb.org/anthology/W03-0419.pdf))        |         News         |    English    |            4 |   14,041/3,250/3,453 | No |
| WNUT 2017 ([`wnut2017`](https://noisy-text.github.io/2017/pdf/WNUT18.pdf))       |         Tweet        |    English    |            6 |    1,000/1,008/1,287 | No |
| WikiAnn ([`panx_dataset/en`, `panx_dataset/ja`, etc](https://www.aclweb.org/anthology/P17-1178.pdf)) | Wikipedia | 282 languages |   3 | 20,000/10,000/10,000 | No |
| MIT Restaurant ([`mit_restaurant`](https://groups.csail.mit.edu/sls/downloads/)) |   Restaurant review  |    English    |            8 |          7,660/1,521 | Yes |
| MIT Movie ([`mit_movie_trivia`](https://groups.csail.mit.edu/sls/downloads/))    |     Movie review     |    English    |           12 |          7,816/1,953 | Yes |

One can specify cache directory by an environment variable `CACHE_DIR`, which set as `./cache` as default.

***WikiAnn dataset***  
All the dataset should be fetched automatically but not `panx_dataset/*` dataset, as you need 
first create the cache directory (`./cache` as the default but can be change through an environment variable `CACHE_DIR`)
and you then need to manually download data from
[here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN?_encoding=UTF8&%2AVersion%2A=1&%2Aentries%2A=0&mgh=1) 
(note that it will download as `AmazonPhotos.zip`) to the cache folder.

***Custom Dataset***  
To go beyond the public datasets, user can use their own dataset by formatting them into
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
The custom dataset should has `train.txt` and `valid.txt` file in a same folder.

Please take a look [sample custom data](./tests/sample_data). 

### Model Finetuning
Language model finetuning can be done with a few lines:
```python
import tner
trainer = tner.TrainTransformersNER(dataset="ontonote5", transformers_model="xlm-roberta-base")
trainer.train()
```
where `transformers_model` is a pre-trained model name from [pretrained LM list](https://huggingface.co/models) and
`dataset` is a dataset alias or path to custom dataset explained [dataset section](#datasets). 

In the end of each epoch, metrics on validation set are computed for monitoring purpose, but it can be turned off to reduce 
training time by
```python
trainer.train(skip_validation=True)
```

***Train on multiple datasets:*** Model can be trained on a concatenation of multiple datasets by 

```python
trainer = tner.TrainTransformersNER(dataset=["ontonote5", "conll2003"], transformers_model="xlm-roberta-base")
```
Custom dataset can be also added to built-in dataset eg) `dataset=["ontonote5", "./test/sample_data"]`.
For more information about the options, you may want to see [here](./tner/model.py#L3).

***Organize model weights (checkpoint files):*** Checkpoint files (model weight, training config, benchmark results, etc)
are stored under `checkpoint_dir`, which is `./ckpt` as default.
The folder names after `<MD5 hash of hyperparameter combination>` (eg, `./ckpt/6bb4fdb286b5e32c068262c2a413639e/`).
Each checkpoint consists of following files:
- `events.out.tfevents.*`: tensorboard file for monitoring the learning proecss
- `label_to_id.json`: dictionary to map prediction id to label
- `model.pt`: pytorch model weight file
- `parameter.json`: model hyperparameters

***Reference:***    
- [colab notebook](https://colab.research.google.com/drive/1AlcTbEsp8W11yflT7SyT0L4C4HG6MXYr?usp=sharing)
- [example_train_eval.py](examples/example_train_eval.py)

### Model Evaluation
To evaluate NER models, here we explain how to proceed in/out of domain evaluation by micro F1 score.
Supposing that your model's checkpoint is `./ckpt/xxx/`. 

```python
import tner
trainer = tner.TrainTransformersNER(checkpoint='./ckpt/xxx')
trainer.test(test_dataset='conll2003')
```
This gives you a accuracy summary.
Again, the `test_dataset` can be a path to custom dataset explained at [dataset section](#datasets).

***Entity position detection:***  For better understanding of out-of-domain accuracy, we provide entity position detection
accuracy, which ignores the entity type and compute metrics only on the IOB entity position.

```python
trainer.test(test_dataset='conll2003', ignore_entity_type=True)
```

***Reference:***    
- [colab notebook](https://colab.research.google.com/drive/1jHVGnFN4AU8uS-ozWJIXXe2fV8HUj8NZ?usp=sharing)
- [example_train_eval.py](helper/example_train_eval.py)

### Model Inference API
To work on model as a part of pipeline, we provide an API to get prediction from trained model.

```python
import tner
classifier = tner.TransformersNER(checkpoint='path-to-checkpoint-dir')
test_sentences = [
    'I live in United States, but Microsoft asks me to move to Japan.',
    'I have an Apple computer.',
    'I like to eat an apple.'
]
classifier.predict(test_sentences)
```
For more information about the module, you may want to see [here](./tner/model.py#L411).

## Experiment with XLM-R
We finetune [XLM-R](https://arxiv.org/pdf/1911.02116.pdf) (`xlm-roberta-base`) on each dataset and
evaluate it on in-domain/cross-domain/cross-lingual setting.

Firstly, we report our baseline on each dataset, where the metrics are quite close to, or even outperform current SoTA. 

|   Dataset              | F1 (val) | F1 (test) | SoTA F1 (test) |                    SoTA reference                    |
|:----------------------:|:--------:|:---------:|:--------------:|:----------------------------------------------------:|
| **`ontonote5`**        |     0.87 |      0.89 |           0.92 | [BERT-MRC-DSC](https://arxiv.org/pdf/1911.02855.pdf) |
| **`conll_2003`**       |     0.95 |      0.91 |           0.94 | [LUKE](https://arxiv.org/pdf/2010.01057v1.pdf)       |
| **`wnut_17`**          |     0.63 |      0.53 |           0.50 | [CrossWeigh](https://www.aclweb.org/anthology/D19-1519.pdf)  |
| **`panx_dataset/en`**  |     0.84 |      0.83 |           0.84 | [mBERT](https://arxiv.org/pdf/2005.00052.pdf)        |
| **`panx_dataset/ja`**  |     0.83 |      0.83 |           0.73 | [XLM-R](https://arxiv.org/pdf/2005.00052.pdf)        |
| **`panx_dataset/ru`**  |     0.89 |      0.89 |        -       |                           -                          |
| **`mit_restaurant`**   |     -    |      0.79 |        -       |                           -                          |
| **`mit_movie_trivia`** |     -    |      0.70 |        -       |                           -                          |

We also report models' entity-detection ability by ignoring entity-type and reduce the prediction/labels to uni-entity task.
If we break down NER task into *entity detection* and *type classification*,
these scores are the upper bound of entity classification for each model.
In any datasets, the discrepancy in between two scores is not large,
which implies model's high capacity of entity type classification.   

|   Dataset              | F1 (val, ignore type) | F1 (test, ignore type) |
|:----------------------:|:---------------------:|:----------------------:|
| **`ontonote5`**        |                  0.91 |                   0.91 |
| **`conll_2003`**       |                  0.98 |                   0.98 |
| **`wnut_17`**          |                  0.73 |                   0.63 | 
| **`panx_dataset/en`**  |                  0.93 |                   0.93 |
| **`panx_dataset/ja`**  |                  0.88 |                   0.88 |
| **`panx_dataset/ru`**  |                  0.94 |                   0.94 |
| **`mit_restaurant`**   | -                     |                   0.83 |
| **`mit_movie_trivia`** | -                     |                   0.73 |

Now, we are interested in how each model, trained on different dataset, differs in capturing entity given a sentence. 
Cross-domain comparison in NER is not so straightforward as in either number of entity type or definition of entity can 
be different depending on each dataset.
Instead of NER metric comparison, we focus on entity detection ability in different models.

|       Train\Test       | `ontonote5` | `conll_2003` | `wnut_17` | `panx_dataset/en` | `mit_movie_trivia` | `mit_restaurant` |
|:----------------------:|:-----------:|:------------:|:---------:|:-----------------:|:------------------:|:----------------:|
| **`ontonote5`**        |      _0.91_ |         0.58 |       0.5 |              0.46 |                0.2 |             0.01 |
| **`conll_2003`**       |        0.61 |       _0.96_ |       0.5 |              0.61 |                  0 |                0 |
| **`wnut_17`**          |        0.52 |         0.69 |    _0.63_ |              0.53 |                  0 |             0.09 |
| **`panx_dataset/en`**  |        0.41 |         0.73 |      0.34 |            _0.93_ |                  0 |             0.08 |
| **`mit_movie_trivia`** |        0.02 |            0 |         0 |                 0 |             _0.73_ |                0 |
| **`mit_restaurant`**   |        0.15 |          0.2 |      0.09 |              0.18 |                  0 |           _0.83_ |

Here, one can see that none of the models transfers well on the other dataset, which indicates the difficulty of domain transfer in NER task. 

Finally, we show cross-lingual transfer result on `panx_dataset`.

| Train\Test             | `panx_dataset/en` | `panx_dataset/ja` | `panx_dataset/ru` |
|:----------------------:|:-------:|:-------:|:-------:|
| **`panx_dataset/en`**  |  _0.83_ |    0.37 |    0.65 |
| **`panx_dataset/ja`**  |    0.53 |  _0.83_ |    0.53 |
| **`panx_dataset/ru`**  |    0.55 |    0.43 |  _0.88_ |


Notes:  
- Configuration can be found in [training script](examples/example_train_eval.py).
- SoTA reported at the time of Oct, 2020.
- F1 score is based on [seqeval](https://pypi.org/project/seqeval/) library, where is span based measure.
- For Japanese dataset, we process each sentence from a collection of characters into proper token by [mecab](https://pypi.org/project/mecab-python3/), so is not directly compatible with prior work. 

## Web App
To play around with NER model, we provide a quick [web App](./asset/api.gif). Please [clone and install the repo](#get-started) firstly.  
1. [Train a model](#train-model) or download [unified model checkpoint file](https://drive.google.com/drive/folders/1UOy_OU4qHyQCYX0QQi02lnCZj7mNFBak?usp=sharing),
`xlm-roberta-base` finetuned on all dataset except MIT corpora,
and put it under `./ckpt` (now you should have `./ckpt/all_english_no_lower_case`).
If you use your own checkpoint, set the path to the checkpoint folder by `export MODEL_CKPT=<path-to-your-checkpoint-folder>`.  

2. Run the app, and open your browser http://0.0.0.0:8000    

```shell script
uvicorn app:app --reload --log-level debug --host 0.0.0.0 --port 8000
```

### Acknowledgement
The App interface is heavily inspired by [Multiple-Choice-Question-Generation-T5-and-Text2Text](https://github.com/renatoviolin/Multiple-Choice-Question-Generation-T5-and-Text2Text).



