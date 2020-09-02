# Named Entity Recognition
Finetuning [transformers](https://github.com/huggingface/transformers) on Named Entity Recognition (NER).

```bash
git clone https://github.com/asahi417/transformers-ner
cd transformers-ner
pip install -r requirement.txt
```

## Wikiann dataset
Download [raw dataset]()
First create a download folder with `mkdir -p ./cache` in the root of this project.
You then need to manually download panx_dataset (for NER) from [here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN?_encoding=UTF8&%2AVersion%2A=1&%2Aentries%2A=0&mgh=1) 
(note that it will download as AmazonPhotos.zip) to the download directory. Finally, run the following command to download the remaining datasets:

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

## Result


## Misc

```bash
python example_train.py --test -c ./ckpt/panx_dataset_en_base 
python example_train.py --test -c ./ckpt/panx_dataset_en_base --test-ignore-entity
python example_train.py --test -c ./ckpt/panx_dataset_en_base --test-data panx_dataset/ja
python example_train.py --test -c ./ckpt/panx_dataset_en_base --test-ignore-entity --test-data panx_dataset/ja
python example_train.py --test -c ./ckpt/panx_dataset_en_base --test-data panx_dataset/ru
python example_train.py --test -c ./ckpt/panx_dataset_en_base --test-ignore-entity --test-data panx_dataset/ru
python example_train.py --test -c ./ckpt/panx_dataset_en_base --test-data conll_2003
python example_train.py --test -c ./ckpt/panx_dataset_en_base --test-ignore-entity --test-data conll_2003
python example_train.py --test -c ./ckpt/panx_dataset_en_base --test-data wnut_17
python example_train.py --test -c ./ckpt/panx_dataset_en_base --test-ignore-entity --test-data wnut_17



python example_train.py --test -c ./ckpt/panx_dataset_ja_base 
python example_train.py --test -c ./ckpt/panx_dataset_ja_base --test-ignore-entity
python example_train.py --test -c ./ckpt/panx_dataset_ja_base --test-data panx_dataset/en
python example_train.py --test -c ./ckpt/panx_dataset_ja_base --test-ignore-entity --test-data panx_dataset/en
python example_train.py --test -c ./ckpt/panx_dataset_ja_base --test-data panx_dataset/ru
python example_train.py --test -c ./ckpt/panx_dataset_ja_base --test-ignore-entity --test-data panx_dataset/ru
python example_train.py --test -c ./ckpt/panx_dataset_ja_base --test-data conll_2003
python example_train.py --test -c ./ckpt/panx_dataset_ja_base --test-ignore-entity --test-data conll_2003
python example_train.py --test -c ./ckpt/panx_dataset_ja_base --test-data wnut_17
python example_train.py --test -c ./ckpt/panx_dataset_ja_base --test-ignore-entity --test-data wnut_17




python example_train.py --test -c ./ckpt/panx_dataset_ru_base 
python example_train.py --test -c ./ckpt/panx_dataset_ru_base --test-ignore-entity
python example_train.py --test -c ./ckpt/panx_dataset_ru_base --test-data panx_dataset/ja
python example_train.py --test -c ./ckpt/panx_dataset_ru_base --test-ignore-entity --test-data panx_dataset/ja
python example_train.py --test -c ./ckpt/panx_dataset_ru_base --test-data panx_dataset/en
python example_train.py --test -c ./ckpt/panx_dataset_ru_base --test-ignore-entity --test-data panx_dataset/en
python example_train.py --test -c ./ckpt/panx_dataset_ru_base --test-data conll_2003
python example_train.py --test -c ./ckpt/panx_dataset_ru_base --test-ignore-entity --test-data conll_2003
python example_train.py --test -c ./ckpt/panx_dataset_ru_base --test-data wnut_17
python example_train.py --test -c ./ckpt/panx_dataset_ru_base --test-ignore-entity --test-data wnut_17




python example_train.py --test --test-data ner-cogent-ja-mecab -c ./ckpt/conll_2003_15db7244e38c1c4ab75e28a5c9419031 --test-ignore-entity
python example_train.py --test --test-data ner-cogent-en -c ./ckpt/panx-ja-mecab_56cf76d3c6a509f0775209e288a10365 --test-ignore-entity

python example_train.py --test --test-data ner-cogent-en -c ./ckpt/ner-cogent-ja-mecab_d7b46e0da8b5605b5f6b6f257438f3bf --test-ignore-entity
python example_train.py --test --test-data conll_2003 -c ./ckpt/ner-cogent-ja-mecab_d7b46e0da8b5605b5f6b6f257438f3bf --test-ignore-entity

python example_train.py --test --test-data ner-cogent-ja-mecab -c ./ckpt/panx-ja-mecab_56cf76d3c6a509f0775209e288a10365 --test-ignore-entity
python example_train.py --test --test-data conll_2003 -c ./ckpt/panx-ja-mecab_56cf76d3c6a509f0775209e288a10365 --test-ignore-entity
python example_train.py --test --test-data ner-cogent-en -c ./ckpt/panx-ja-mecab_56cf76d3c6a509f0775209e288a10365 --test-ignore-entity

python example_train.py --test --test-data panx_dataset-mecab -c ./ckpt/conll_2003_15db7244e38c1c4ab75e28a5c9419031 --test-ignore-entity
python example_train.py --test --test-data panx-ja-mecab -c ./ckpt/ner-cogent-ja-mecab_d7b46e0da8b5605b5f6b6f257438f3bf --test-ignore-entity
python example_train.py --test --test-data panx-ja-mecab -c ./ckpt/ner-cogent-en_3864f79d1ed1b1fa0a76f9a7a9c9d58e --test-ignore-entity


```