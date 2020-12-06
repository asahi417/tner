# T-NER: Transformers NER  

![](./asset/api.png)

***`tner`***, a python tool to inspect finetuning of pre-trained language model (LM) for Named-Entity-Recognition (NER). 
The following features are supported:
- [Modules to finetune LMs](#train-model) (see in [google colab](https://colab.research.google.com/drive/1AlcTbEsp8W11yflT7SyT0L4C4HG6MXYr?usp=sharing))
    - various dataset option
    - organizing checkpoints by hyperparameter configuration
    - tensorboard visualization
- [Script to evaluate model which enables](#evaluate-on-inout-of-domain-f1-score-withwithout-entity-type)  (see in [google colab](https://colab.research.google.com/drive/1jHVGnFN4AU8uS-ozWJIXXe2fV8HUj8NZ?usp=sharing)) 
    - **in-domain/cross-domain/cross-lingual** span-F1 score across datasets
    - **entity position F1 score** (reduce the prediction and true label to be entity-agnostic, to see entity position detection performance, 
    see the [baseline result](#result)) 
- [Web app](#web-app) to visualize model prediction.
- [Command line tool to get model prediction.](#model-inference-interface)
 
***Table of Contents***  
1. [Setup](#get-started)
2. [Model training](#model-trainingevaluation)

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

## Model Training/Evaluation

<p align="center">
  <img src="./asset/tb_valid.png" width="600">
  <br><i>Fig 1: Tensorboard visualization</i>
</p>

### Train model
Pick up a model from [pretrained LM list](https://huggingface.co/models), and run the following lines to finetune on NER! 

```python
import tner
trainer = tner.TrainTransformersNER(
        dataset="ontonote5",  # NER dataset name
        transformer="xlm-roberta-base",  # transformers model name
        checkpoint_dir="./ckpt",  
        random_seed=1234,
        lr=1e-5,
        total_step=13000,
        warmup_step=700,
        weight_decay=1e-7,
        batch_size=16,
        batch_size_validation=2,
        max_seq_length=128,
        fp16=False,
        max_grad_norm=1.0
    )
trainer.train()
```
As a choice of NER dataset, following data sources are supported.   

|                                   Name                                           |         Genre        |    Language   | Entity types |       Data size      | Lower-cased |
|:--------------------------------------------------------------------------------:|:--------------------:|:-------------:|:------------:|:--------------------:|:-----------:|
| OntoNote 5 ([`ontonote5`](https://www.aclweb.org/anthology/N06-2015.pdf))          | News, Blog, Dialogue |    English    |           18 |   59,924/8,582/8,262 | No | 
| CoNLL 2003 ([`conll_2003`](https://www.aclweb.org/anthology/W03-0419.pdf))         |         News         |    English    |            4 |   14,041/3,250/3,453 | No |
| WNUT 2017 ([`wnut_17`](https://noisy-text.github.io/2017/pdf/WNUT18.pdf))          |         Tweet        |    English    |            6 |       1,000/1,008/1,287 | No |
| WikiAnn ([`panx_dataset/en`, `panx_dataset/ja`, etc](https://www.aclweb.org/anthology/P17-1178.pdf)) |       Wikipedia      | 282 languages |            3 | 20,000/10,000/10,000 | No |
| MIT Restaurant ([`mit_restaurant`](https://groups.csail.mit.edu/sls/downloads/))   |   Restaurant review  |    English    |            8 |          7,660/1,521 | Yes |
| MIT Movie ([`mit_movie_trivia`](https://groups.csail.mit.edu/sls/downloads/))      |     Movie review     |    English    |           12 |          7,816/1,953 | Yes |

Checkpoints are stored under `checkpoint_dir`, called `<dataset>_<MD5 hash of hyperparameter combination>`
(eg, `./ckpt/ontonote5_6bb4fdb286b5e32c068262c2a413639e/`). Each checkpoint consists of following files:
- `events.out.tfevents.*`: tensorboard file for monitoring the learning proecss
- `label_to_id.json`: dictionary to map prediction id to label
- `model.pt`: pytorch model weight file
- `parameter.json`: model hyperparameters
- `logger_train.log`: training log

For more conclude examples, take a look below:  
- [colab notebook](https://colab.research.google.com/drive/1AlcTbEsp8W11yflT7SyT0L4C4HG6MXYr?usp=sharing)
- [example_train_eval.py](examples/example_train_eval.py)

***WikiAnn dataset***  
All the dataset should be fetched automatically but not `panx_dataset/*` dataset, as you need 
first create a cache folder with `mkdir -p ./cache` in the root of this project if it's not created yet.
You then need to manually download data from
[here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN?_encoding=UTF8&%2AVersion%2A=1&%2Aentries%2A=0&mgh=1) 
(note that it will download as `AmazonPhotos.zip`) to the cache folder.


### Evaluate on in/out of domain F1 score with/without entity type
Once a model was trained on any dataset, you can start test it on other dataset to see 
**cross-domain/cross-lingual** transferring ability as well as in domain accuracy.
In a same manner, **entity position accuracy** can be produced.
Here, let's suppose that your model was trained on `ontonote5`, and checkpoint files are in `./ckpt/ontonote5_6bb4fdb286b5e32c068262c2a413639e/`. 

```python
import tner
# model instance initialization with the checkpoint 
trainer = tner.TrainTransformersNER(checkpoint='./ckp/ontonote5_6bb4fdb286b5e32c068262c2a413639e')

# test in domain accuracy (just on the valid/test set of the dataset where the model trained on) 
trainer.test()

# test out of domain accuracy
trainer.test(test_dataset='conll_2003')

# test entity span accuracy
trainer.test(test_dataset='conll_2003', ignore_entity_type=True)
```

Evaluation process create `logger_test.<dataname>.log` file where includes all the report under the checkpoint directory.
For more conclude examples, take a look below:  
- [colab notebook](https://colab.research.google.com/drive/1jHVGnFN4AU8uS-ozWJIXXe2fV8HUj8NZ?usp=sharing)
- [example_train_eval.py](helper/example_train_eval.py)

### Result
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

### Model inference interface
To get an inference from finetuned model can be done as below.

```python
import tner
classifier = tner.TransformersNER(checkpoint='path-to-checkpoint-folder')
test_sentences = [
    'I live in United States, but Microsoft asks me to move to Japan.',
    'I have an Apple computer.',
    'I like to eat an apple.'
]
classifier.predict(test_sentences)
```


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

## Acknowledgement
The App interface is heavily inspired by [Multiple-Choice-Question-Generation-T5-and-Text2Text](https://github.com/renatoviolin/Multiple-Choice-Question-Generation-T5-and-Text2Text).



