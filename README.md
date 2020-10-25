# Transformer NER  

![](./asset/api.gif)

***Transformer NER***, a python tool to inspect finetuning of pre-trained language model (LM) for Named-Entity-Recognition (NER). 
The following features are supported:
- [Modules to finetune LMs](#train-model) (see in [google colab](https://colab.research.google.com/drive/1AlcTbEsp8W11yflT7SyT0L4C4HG6MXYr?usp=sharing))
    - various dataset option
    - organizing checkpoints by hyperparameter configuration
    - tensorboard visualization
- [Script to evaluate model which enables](#evaluate-on-inout-of-domain-f1-score-withwithout-entity-type)  (see in [google colab](https://colab.research.google.com/drive/1jHVGnFN4AU8uS-ozWJIXXe2fV8HUj8NZ?usp=sharing)) 
    - **in-domain/cross-domain/cross-lingual** span-F1 score across datasets
    - **entity position F1 score** (reduce the prediction and true label to be entity-agnostic, to see entity position detection performance, 
    see the [baseline result](#result)) 
- [Web app](#web-app) to visualize model prediction (shown above).
- Command line tool to get model prediction.
 
## Get Started
Clone and install libraries.
```shell script
git clone https://github.com/asahi417/transformers-ner
cd transformers-ner
pip install -r requirement.txt
```

## Model Training/Evaluation

<p align="center">
  <img src="./asset/tb_valid.png" width="800">
  <br><i>Fig 1: Tensorboard visualization</i>
</p>

### Train model
Pick up a model from [pretrained LM list](https://huggingface.co/models), and run the following lines to finetune on NER! 

```python
from src import TrainTransformerNER
trainer = TrainTransformerNER(
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

|                                   Name                                  |         Genre        |    Language   | Entity types |       Data size      |
|:-----------------------------------------------------------------------:|:--------------------:|:-------------:|:------------:|:--------------------:|
|        [ontonote5](https://www.aclweb.org/anthology/N06-2015.pdf)       | News, Blog, Dialogue |    English    |           18 |   59,924/8,582/8,262 |
|       [conll_2003](https://www.aclweb.org/anthology/W03-0419.pdf)       |         News         |    English    |            4 |   14,041/3,250/3,453 |
| [panx/en, panx/ja, etc](https://www.aclweb.org/anthology/P17-1178.pdf)  |       Wikipedia      | 282 languages |            3 | 20,000/10,000/10,000 |
|     [mit_restaurant](https://groups.csail.mit.edu/sls/downloads/)       |   Restaurant review  |    English    |            8 |          7,660/1,521 |
|       [mit_movie_trivia](https://groups.csail.mit.edu/sls/downloads/)   |     Movie review     |    English    |           12 |          7,816/1,953 |
|       [wnut_17](https://noisy-text.github.io/2017/pdf/WNUT18.pdf)       |         Tweet        |    English    |            6 |       1000/1008/1287 |


Checkpoints are stored under `checkpoint_dir`, called `<dataset>_<MD5 hash of hyperparameter combination>`
(eg, `./ckpt/ontonote5_6bb4fdb286b5e32c068262c2a413639e/`). Each checkpoint consists of following files:
- `events.out.tfevents.*`: tensorboard file for monitoring the learning proecss
- `label_to_id.json`: dictionary to map prediction id to label
- `model.pt`: pytorch model weight file
- `parameter.json`: model hyperparameters
- `*.log`: process log

For more conclude examples, take a look below:  
- [colab notebook](https://colab.research.google.com/drive/1AlcTbEsp8W11yflT7SyT0L4C4HG6MXYr?usp=sharing)
- [example_train_eval.py](example_train_eval.py)

***WikiAnn (panx) dataset***  
All the dataset should be fetched automatically but not `panx/*` dataset, as you need 
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
from src import TrainTransformerNER
# model instance initialization with the checkpoint 
trainer = TrainTransformerNER(checkpoint='./ckp/ontonote5_6bb4fdb286b5e32c068262c2a413639e')

# test in domain accuracy (just on the valid/test set of the dataset where the model trained on) 
trainer.test()

# test out of domain accuracy
trainer.test(test_dataset='conll_2003')

# test entity span accuracy
trainer.test(test_dataset='conll_2003', ignore_entity_type=True)
```

Evaluation process create `*.log` file where includes all the report under the checkpoint directory.
For more conclude examples, take a look below:  
- [colab notebook](https://colab.research.google.com/drive/1jHVGnFN4AU8uS-ozWJIXXe2fV8HUj8NZ?usp=sharing)
- [example_train_eval.py](example_train_eval.py)

### Result
As a baseline, we finetuned [XLM-R](https://arxiv.org/pdf/1911.02116.pdf) (`xlm-roberta-base`) on each dataset and
evaluate it on in-domain/cross-domain/cross-lingual setting.
We set the configuration used here as the default value in the [training script](example_train_eval.py).

***In-domain span F1 score***

|   Dataset  | F1 (val) | F1 (test) | SoTA F1 (test) |                    SoTA reference                    |
|:----------:|:--------:|:---------:|:--------------:|:----------------------------------------------------:|
| OntoNote 5 |     0.87 |      0.89 |           0.92 | [BERT-MRC-DSC](https://arxiv.org/pdf/1911.02855.pdf) |
| CoNLL 2003 |     0.95 |      0.91 |           0.94 | [LUKE](https://arxiv.org/pdf/2010.01057v1.pdf)       |
| PanX (en)  |     0.84 |      0.83 |           0.84 | [mBERT](https://arxiv.org/pdf/2005.00052.pdf)        |
| PanX (ja)  |     0.83 |      0.83 |           0.73 | [XLM-R](https://arxiv.org/pdf/2005.00052.pdf)        |
| PanX (ru)  |     0.89 |      0.89 |        -       |                           -                          |
| Restaurant |     -    |      0.79 |        -       |                           -                          |
| Movie      |     -    |      0.70 |        -       |                           -                          |

***In-domain span F1 score (ignore entity type)***

|   Dataset  | F1 (val, ignore type) | F1 (test, ignore type) |
|:----------:|:---------------------:|:----------------------:|
| OntoNote 5 |                  0.91 |                   0.91 |
| CoNLL 2003 |                  0.98 |                   0.98 |
| PanX (en)  |                  0.93 |                   0.93 |
| PanX (ja)  |                  0.88 |                   0.88 |
| PanX (ru)  |                  0.94 |                   0.94 |
| Restaurant | -                     |                   0.83 |
| Movie      | -                     |                   0.73 |

***Cross-domain span F1 score (ignore entity type)***

|       train\test      | OntoNote  (News, blog) | Conll (News) | wiki/en | Movie | Restaurant |
|:---------------------:|:----------------------:|:------------:|:-------:|:-----:|:----------:|
| OntoNote (News, blog) |                   0.91 |         0.58 |    0.46 |   0.2 |       0.01 |
|      Conll (News)     |                   0.61 |         0.96 |    0.61 |     0 |          0 |
|        wiki/en        |                   0.41 |         0.73 |    0.93 |     0 |       0.08 |
|         Movie         |                   0.02 |            0 |       0 |  0.73 |          0 |
|       Restaurant      |                   0.15 |          0.2 |    0.18 |     0 |       0.83 |

***Cross-lingual span F1 score***

| train\test | wiki/en | wiki/ja | wiki/ru |
|:----------:|:-------:|:-------:|:-------:|
|   wiki/en  |    0.83 |    0.37 |    0.65 |
|   wiki/ja  |    0.53 |    0.83 |    0.53 |
|   wiki/ru  |    0.55 |    0.43 |    0.88 |


Notes:  
- SOTA reported at the time of Oct, 2020.
- F1 score is based on [seqeval](https://pypi.org/project/seqeval/) library, where is span based measure.
- For Japanese dataset, we process each sentence from a collection of characters into proper token by [mecab](https://pypi.org/project/mecab-python3/), so is not directly compatible with prior work. 

## Web App
To play around with NER model, we provide a quick web App. 
1. [Train a model](#train-model) or download [default model checkpoint file](https://drive.google.com/file/d/19SLaL_KMDXvI15oPlNRd6ZCNEdmypU7s/view?usp=sharing),
`xlm-roberta-base` finetuned on `ontonote5`,
and unzip it (now you should have `./ckpt/default`).
If you use your own checkpoint, set the path to the checkpoint folder by `export MODEL_CKPT=<path-to-your-checkpoint-folder>`.  

2. Run the app, and open your browser http://0.0.0.0:8000    

```shell script
uvicorn app:app --reload --log-level debug --host 0.0.0.0 --port 8000
```

## Acknowledgement
The App interface is heavily inspired by [Multiple-Choice-Question-Generation-T5-and-Text2Text](https://github.com/renatoviolin/Multiple-Choice-Question-Generation-T5-and-Text2Text).



