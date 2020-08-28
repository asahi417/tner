# Named Entity Recognition
Finetuning [transformers](https://github.com/huggingface/transformers) on Named Entity Recognition (NER).

```bash
git clone https://github.com/asahi417/transformers-ner
cd transformers-ner
pip install -r requirement.txt
```


## Train model

```bash
python example_train.py \
    --checkpoint-dir ./ckpt \
    -d conll_2013
```

## Test model on dataset
```bash
python example_train.py \
    --test \
    -c {path-to-checkpoint-dir} \
```

## Run APP
```
uvicorn app:app --reload --log-level debug
```


```bash

```