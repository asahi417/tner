
# Transformer NER  

![](./asset/api.gif)

***Transformer NER*** is a python tool to inspect finetuning of pre-trained language model (LM) for Named-Entity-Recognition (NER) performance specifically. 
The following features are supported:
- Script to finetune LMs distributed by [transformers](https://huggingface.co/models)
    - [various dataset option](#model-training) 
    - organizing checkpoints by hyperparameter configuration
    - tensorboard visualization
- Script to produce benchmark of in-domain/cross-domain span-F1 score over the dataset.
- Interactive web app to visualize model prediction (shown above).
- Command line tool to get model prediction.
 
## Get Started
Clone and install libraries.
```shell script
git clone https://github.com/asahi417/transformers-ner
cd transformers-ner
pip install -r requirement.txt
```

## Model Training/Evaluation

![](asset/tb_valid.png)

### Training
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


Checkpoints are stored under `checkpoint_dir`, called `<dataset>_<MD5 hash of hyperparameter combination>` (eg, `./ckpt/ontonote5_6bb4fdb286b5e32c068262c2a413639e/`).
Each checkpoint consists of following files:
- `events.out.tfevents.*`: tensorboard file for monitoring the learning proecss
- `label_to_id.json`: dictionary to map prediction id to label
- `model.pt`: pytorch model weight file
- `parameter.json`: model hyperparameters
- `*.log`: process log

### Evaluation
Once a model was trained on any dataset, you can start  

## Model
## App
Default checkpoint is fine-tuned on [XLM-R](https://arxiv.org/pdf/1911.02116.pdf), so can be tested on any language.

## Application
1. Download [default model checkpoint file](https://drive.google.com/file/d/19SLaL_KMDXvI15oPlNRd6ZCNEdmypU7s/view?usp=sharing), 
and unzip the file, so that you have a default checkpoint folder `./ckpt/default`.
2. Run the app, and open your browser http://0.0.0.0:8000    

```shell script
uvicorn app:app --reload --log-level debug --host 0.0.0.0 --port 8000
```
One can also specify model checkpoint by `export MODEL_CKPT={path to checkpoint directory}`, which produced by following training script.

## Model Training
Here's a benchmark, where all the models are trained on [XLM-R](https://arxiv.org/pdf/1911.02116.pdf) (`xlm-roberta-base`) for 3 epochs.

| Dataset    | Language | # Type | # Sent (train/val/test) | F1 (val) | F1 (test) | SoTA F1 (test) | 
|------------|----------|--------|-------------------------|----------|-----------|-----------------| 
| OntoNote 5 | English  | 18     | 59,924/8,582/8,262      | 0.87     | 0.89      | 0.9207 ([BERT-MRC-DSC](https://arxiv.org/pdf/1911.02855.pdf)) |
| CoNLL 2003 | English  | 4      | 14,041/3,250/3,453      | 0.95     | 0.91      | 0.943 ([LUKE](https://arxiv.org/pdf/2010.01057v1.pdf)) |
| PanX (en)  | English  | 4      | 20,000/10,000/10,000    | 0.84     | 0.83      | 0.848 ([mBERT](https://arxiv.org/pdf/2005.00052.pdf)) | 
| PanX (ja)  | Japanese | 4      | 20,000/10,000/10,000    | 0.83     | 0.83      | 0.733 ([XLM-R](https://arxiv.org/pdf/2005.00052.pdf)) |
| PanX (ru)  | Russian  | 4      | 20,000/10,000/10,000    | 0.89     | 0.89      | - |
| Restaurant | English  | 8      | 7,660/1,521             | 0.79     | -         | - |
| Movie      | English  | 12     | 7,816/1,953             | 0.7      | -         | - |

- SOTA reported at the time of Oct, 2020.
- F1 score is based on [seqeval](https://pypi.org/project/seqeval/) library, where is span based measure.

You can train a model on various public dataset such as
[OntoNote5](https://www.aclweb.org/anthology/N06-2015.pdf),
[CoNLL 2003](https://www.aclweb.org/anthology/W03-0419.pdf),
[WikiAnn (PanX dataset)](https://www.aclweb.org/anthology/P17-1178.pdf),
[Restaurant Rating](https://groups.csail.mit.edu/sls/downloads/),
[Movie Review](https://groups.csail.mit.edu/sls/downloads/), and
[WNUT2017](https://noisy-text.github.io/2017/pdf/WNUT18.pdf) 
by following script. 

```shell script
usage: example_train.py [-h] [-c CHECKPOINT] [--checkpoint-dir CHECKPOINT_DIR]
                        [-d DATA] [-t TRANSFORMER]
                        [--max_grad_norm MAX_GRAD_NORM]
                        [--max-seq-length MAX_SEQ_LENGTH] [-b BATCH_SIZE]
                        [--random-seed RANDOM_SEED] [--lr LR]
                        [--total-step TOTAL_STEP]
                        [--batch-size-validation BATCH_SIZE_VALIDATION]
                        [--warmup-step WARMUP_STEP]
                        [--weight-decay WEIGHT_DECAY]
                        [--early-stop EARLY_STOP] [--fp16] [--test]
                        [--test-data TEST_DATA] [--test-ignore-entity]

Fine-tune transformers on NER dataset

optional arguments:
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        checkpoint to load
  --checkpoint-dir CHECKPOINT_DIR
                        checkpoint directory
  -d DATA, --data DATA  dataset: ['panx_dataset/*', 'conll_2003', 'wnut_17', 'ontonote5', 'mit_movie_trivia', 'mit_restaurant']
  -t TRANSFORMER, --transformer TRANSFORMER
                        pretrained language model
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm.
  --max-seq-length MAX_SEQ_LENGTH
                        max sequence length (use same length as used in pre-training if not provided)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size
  --random-seed RANDOM_SEED
                        random seed
  --lr LR               learning rate
  --total-step TOTAL_STEP
                        total training step
  --batch-size-validation BATCH_SIZE_VALIDATION
                        batch size for validation (smaller size to save memory)
  --warmup-step WARMUP_STEP
                        warmup step (6 percent of total is recommended)
  --weight-decay WEIGHT_DECAY
                        weight decay
  --early-stop EARLY_STOP
                        value of accuracy drop for early stop
  --fp16                fp16
  --test                test mode
  --test-data TEST_DATA
                        test dataset (if not specified, use trained set)
  --test-ignore-entity  test with ignoring entity type
```

***Model training examples***  
You can reproduce the default checkpoint by 
```shell script
python ./example_train.py \
    -t xlm-roberta-base \
    -d ontonote5 \
    --max-seq-length 128
```
Checkpoint files are automatically organized depends on the hyperparameters.

***WikiAnn dataset***  
To train a model on [WikiAnn dataset](https://www.aclweb.org/anthology/P17-1178.pdf),
first create a download folder with `mkdir -p ./cache` in the root of this project if it's not created yet.
You then need to manually download panx_dataset (for NER) from
[here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN?_encoding=UTF8&%2AVersion%2A=1&%2Aentries%2A=0&mgh=1) 
(note that it will download as AmazonPhotos.zip) to the download directory. You now can train a model on it by

```shell script
python ./example_train.py -d panx_dataset/ja
```

## Acknowledgement
The App interface is heavily inspired by [Multiple-Choice-Question-Generation-T5-and-Text2Text](https://github.com/renatoviolin/Multiple-Choice-Question-Generation-T5-and-Text2Text).



