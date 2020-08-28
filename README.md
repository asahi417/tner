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
python example_train.py --test --test-data ner-cogent-ja-mecab -c ./ckpt/conll_2003_15db7244e38c1c4ab75e28a5c9419031 --test-ignore-entity
python example_train.py --test --test-data ner-cogent-en -c ./ckpt/panx-ja-mecab_56cf76d3c6a509f0775209e288a10365 --test-ignore-entity

python example_train.py --test --test-data ner-cogent-en -c ./ckpt/ner-cogent-ja-mecab_d7b46e0da8b5605b5f6b6f257438f3bf --test-ignore-entity
python example_train.py --test --test-data conll_2003 -c ./ckpt/ner-cogent-ja-mecab_d7b46e0da8b5605b5f6b6f257438f3bf --test-ignore-entity

python example_train.py --test --test-data ner-cogent-ja-mecab -c ./ckpt/panx-ja-mecab_56cf76d3c6a509f0775209e288a10365 --test-ignore-entity
python example_train.py --test --test-data conll_2003 -c ./ckpt/panx-ja-mecab_56cf76d3c6a509f0775209e288a10365 --test-ignore-entity
python example_train.py --test --test-data ner-cogent-en -c ./ckpt/panx-ja-mecab_56cf76d3c6a509f0775209e288a10365 --test-ignore-entity

python example_train.py --test --test-data panx-ja-mecab -c ./ckpt/conll_2003_15db7244e38c1c4ab75e28a5c9419031 --test-ignore-entity
python example_train.py --test --test-data panx-ja-mecab -c ./ckpt/ner-cogent-ja-mecab_d7b46e0da8b5605b5f6b6f257438f3bf --test-ignore-entity
python example_train.py --test --test-data panx-ja-mecab -c ./ckpt/ner-cogent-en_3864f79d1ed1b1fa0a76f9a7a9c9d58e --test-ignore-entity


```