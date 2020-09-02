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

| Dataset    | OntoNote5   | Conll 2003  | Pan-en       | Movie | Restaurant |
|------------|-------------|-------------|--------------|-------|------------|
| OntoNote5  | 0.87 (0.89) |             |              |       |            |
| Conll 2003 |             | 0.95 (0.91) |              |       |            |
| Pan-en     |             |             | 0.84 (0.83)  |       |            |
| Movie      |             |             |              | 0.70  |            |
| Restaurant |             |             |              |       | 0.79       |

## Misc
- Cross-domain transfer test
```bash
python example_train.py --test -c ./ckpt/ontonote5 
python example_train.py --test -c ./ckpt/ontonote5 --test-ignore-entity
python example_train.py --test -c ./ckpt/conll_2003 
python example_train.py --test -c ./ckpt/conll_2003 --test-ignore-entity
python example_train.py --test -c ./ckpt/mit_movie_trivia 
python example_train.py --test -c ./ckpt/mit_movie_trivia --test-ignore-entity
python example_train.py --test -c ./ckpt/mit_restaurant
python example_train.py --test -c ./ckpt/mit_restaurant --test-ignore-entity
python example_train.py --test -c ./ckpt/panx_dataset_en
python example_train.py --test -c ./ckpt/panx_dataset_en --test-ignore-entity

python example_train.py --test -c ./ckpt/ontonote5 --test-data conll_2003 
python example_train.py --test -c ./ckpt/ontonote5 --test-data mit_movie_trivia
python example_train.py --test -c ./ckpt/ontonote5 --test-data mit_restaurant
python example_train.py --test -c ./ckpt/ontonote5 --test-data panx_dataset_en 

python example_train.py --test -c ./ckpt/conll_2003 --test-data ontonote5 
python example_train.py --test -c ./ckpt/conll_2003 --test-data mit_movie_trivia
python example_train.py --test -c ./ckpt/conll_2003 --test-data mit_restaurant
python example_train.py --test -c ./ckpt/conll_2003 --test-data panx_dataset_en 

python example_train.py --test -c ./ckpt/mit_movie_trivia --test-data ontonote5 
python example_train.py --test -c ./ckpt/mit_movie_trivia --test-data conll_2003
python example_train.py --test -c ./ckpt/mit_movie_trivia --test-data mit_restaurant
python example_train.py --test -c ./ckpt/mit_movie_trivia --test-data panx_dataset_en 

python example_train.py --test -c ./ckpt/mit_restaurant --test-data ontonote5 
python example_train.py --test -c ./ckpt/mit_restaurant --test-data conll_2003
python example_train.py --test -c ./ckpt/mit_restaurant --test-data mit_movie_trivia
python example_train.py --test -c ./ckpt/mit_restaurant --test-data panx_dataset_en 

python example_train.py --test -c ./ckpt/panx_dataset_en --test-data ontonote5 
python example_train.py --test -c ./ckpt/panx_dataset_en --test-data conll_2003
python example_train.py --test -c ./ckpt/panx_dataset_en --test-data mit_movie_trivia
python example_train.py --test -c ./ckpt/panx_dataset_en --test-data mit_restaurant 


python example_train.py --test -c ./ckpt/ontonote5 --test-data conll_2003 --test-ignore-entity 
python example_train.py --test -c ./ckpt/ontonote5 --test-data mit_movie_trivia --test-ignore-entity
python example_train.py --test -c ./ckpt/ontonote5 --test-data mit_restaurant --test-ignore-entity
python example_train.py --test -c ./ckpt/ontonote5 --test-data panx_dataset_en --test-ignore-entity

python example_train.py --test -c ./ckpt/conll_2003 --test-data ontonote5 --test-ignore-entity
python example_train.py --test -c ./ckpt/conll_2003 --test-data mit_movie_trivia --test-ignore-entity
python example_train.py --test -c ./ckpt/conll_2003 --test-data mit_restaurant --test-ignore-entity
python example_train.py --test -c ./ckpt/conll_2003 --test-data panx_dataset_en --test-ignore-entity

python example_train.py --test -c ./ckpt/mit_movie_trivia --test-data ontonote5 --test-ignore-entity
python example_train.py --test -c ./ckpt/mit_movie_trivia --test-data conll_2003 --test-ignore-entity
python example_train.py --test -c ./ckpt/mit_movie_trivia --test-data mit_restaurant --test-ignore-entity
python example_train.py --test -c ./ckpt/mit_movie_trivia --test-data panx_dataset_en --test-ignore-entity

python example_train.py --test -c ./ckpt/mit_restaurant --test-data ontonote5 --test-ignore-entity
python example_train.py --test -c ./ckpt/mit_restaurant --test-data conll_2003 --test-ignore-entity
python example_train.py --test -c ./ckpt/mit_restaurant --test-data mit_movie_trivia --test-ignore-entity
python example_train.py --test -c ./ckpt/mit_restaurant --test-data panx_dataset_en --test-ignore-entity

python example_train.py --test -c ./ckpt/panx_dataset_en --test-data ontonote5 --test-ignore-entity
python example_train.py --test -c ./ckpt/panx_dataset_en --test-data conll_2003 --test-ignore-entity
python example_train.py --test -c ./ckpt/panx_dataset_en --test-data mit_movie_trivia --test-ignore-entity
python example_train.py --test -c ./ckpt/panx_dataset_en --test-data mit_restaurant --test-ignore-entity
```

```bash
python example_train.py --test -c ./ckpt/panx_dataset_ja
python example_train.py --test -c ./ckpt/panx_dataset_ja --test-ignore-entity
python example_train.py --test -c ./ckpt/ner-cogent-ja
python example_train.py --test -c ./ckpt/ner-cogent-ja --test-ignore-entity
```

- Cross-lingual transfer test
```bash
python example_train.py --test -c ./ckpt/panx_dataset_ru
python example_train.py --test -c ./ckpt/panx_dataset_ru --test-ignore-entity
python example_train.py --test -c ./ckpt/ner-cogent-ja
python example_train.py --test -c ./ckpt/ner-cogent-ja --test-ignore-entity

python example_train.py --test -c ./ckpt/panx_dataset_ja --test-data panx_dataset_en
python example_train.py --test -c ./ckpt/panx_dataset_ja --test-data panx_dataset_ru
python example_train.py --test -c ./ckpt/panx_dataset_ja --test-data panx_dataset_ru

python example_train.py --test -c ./ckpt/panx_dataset_ja --test-data panx_dataset_en --test-ignore-entity
```