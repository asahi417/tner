# TweetNER7
This is an official repository of TweetNER7, an NER dataset on Twitter with 7 entity labels. Each instance of TweetNER7 comes with
a timestamp which distributes from September 2019 to August 2021. 
The dataset is available on the huggingface [https://huggingface.co/datasets/tner/tweetner7](https://huggingface.co/datasets/tner/tweetner7), and 
basic usage follows below. 
This repository explains how to reproduce the experimental results on our paper. Please visit the huggingface page of the dataset
[https://huggingface.co/datasets/tner/tweetner7](https://huggingface.co/datasets/tner/tweetner7) to know more about the dataset.

- ***Dataset***

```python
from datasets import load_dataset
dataset = load_dataset("tner/tweetner7")
```

- ***Split***

| split             | number of instances | description |
|:------------------|------:|------:|
| train_2020        |  4616 | training dataset from September 2019 to August 2020 |
| train_2021        |  2495 | training dataset from September 2020 to August 2021 |
| train_all         |  7111 | combined training dataset of `train_2020` and `train_2021` |
| validation_2020   |   576 | validation dataset from September 2019 to August 2020 |
| validation_2021   |   310 | validation dataset from September 2020 to August 2021 | 
| test_2020         |   576 | test dataset from September 2019 to August 2020 |
| test_2021         |  2807 | test dataset from September 2020 to August 2021 |
| train_random      |  4616 | randomly sampled training dataset with the same size as `train_2020` from `train_all` |
| validation_random |   576 | randomly sampled training dataset with the same size as `validation_2020` from `validation_all` |
| extra_2020        | 87880 | extra tweet without annotations from September 2019 to August 2020 |
| extra_2021        | 93594 | extra tweet without annotations from September 2020 to August 2021 |

- ***Models:*** Following models are fine-tuned on `train_all` and validated on `validation_2021` of `tner/tweetner7`.  See full model list [here](https://github.com/asahi417/tner/blob/master/MODEL_CARD.md#models-for-tweetner7).

| Model (link)                                                                                                                                | Data                                                          | Language Model                                                                          |
|:--------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|:----------------------------------------------------------------------------------------|
| [`tner/roberta-large-tweetner7-all`](https://huggingface.co/tner/roberta-large-tweetner7-all)                                               | [`tweetner7`](https://huggingface.co/datasets/tner/tweetner7) | [`roberta-large`](https://huggingface.co/roberta-large)                                 |
| [`tner/roberta-base-tweetner7-all`](https://huggingface.co/tner/roberta-base-tweetner7-all)                                                 | [`tweetner7`](https://huggingface.co/datasets/tner/tweetner7) | [`roberta-base`](https://huggingface.co/roberta-base)                                   |
| [`tner/twitter-roberta-base-2019-90m-tweetner7-all`](https://huggingface.co/tner/twitter-roberta-base-2019-90m-tweetner7-all)               | [`tweetner7`](https://huggingface.co/datasets/tner/tweetner7) | [`twitter-roberta-base-2019-90m`](https://huggingface.co/twitter-roberta-base-2019-90m) |
| [`tner/twitter-roberta-base-dec2020-tweetner7-all`](https://huggingface.co/tner/twitter-roberta-base-dec2020-tweetner7-all)                 | [`tweetner7`](https://huggingface.co/datasets/tner/tweetner7) | [`twitter-roberta-base-dec2020`](https://huggingface.co/twitter-roberta-base-dec2020)   |
| [`tner/twitter-roberta-base-dec2021-tweetner7-all`](https://huggingface.co/tner/twitter-roberta-base-dec2021-tweetner7-all)                 | [`tweetner7`](https://huggingface.co/datasets/tner/tweetner7) | [`twitter-roberta-base-dec2021`](https://huggingface.co/twitter-roberta-base-dec2021)   |

Basic usage of those models follows below.

```python
import re
from urlextract import URLExtract
from tner import TransformersNER

extractor = URLExtract()

def format_tweet(tweet):
    # mask web urls
    urls = extractor.find_urls(tweet)
    for url in urls:
        tweet = tweet.replace(url, "{{URL}}")
    # format twitter account
    tweet = re.sub(r"\b(\s*)(@[\S]+)\b", r'\1{\2@}', tweet)
    return tweet


text = "Get the all-analog Classic Vinyl Edition of `Takin' Off` Album from @herbiehancock via @bluenoterecords link below: http://bluenote.lnk.to/AlbumOfTheWeek"
text_format = format_tweet(text)
model = TransformersNER("tner/roberta-large-tweetner7-all")
model.predict([text_format])
```

## Reference
If you use the dataset or any of these resources, please cite the following paper:
```
@inproceedings{ushio-etal-2022-tweet,
    title = "{N}amed {E}ntity {R}ecognition in {T}witter: {A} {D}ataset and {A}nalysis on {S}hort-{T}erm {T}emporal {S}hifts",
    author = "Ushio, Asahi  and
        Neves, Leonardo  and
        Silva, Vitor  and
        Barbieri, Francesco. and
        Camacho-Collados, Jose",
    booktitle = "The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing",
    month = nov,
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```

## Model Fine-tuning
- Finetuning on 2020 training/validation set
```shell
evaluate_model () {
  tner-evaluation -m "cner_output/model/${1}/${2}/best_model" -e "cner_output/model/${1}/${2}/best_model/eval/metric.test_2021.json" --dataset-split "test_2021" --return-ci
  tner-evaluation -m "cner_output/model/${1}/${2}/best_model" -e "cner_output/model/${1}/${2}/best_model/eval/metric.test_2020.json" --dataset-split "test_2020" --return-ci
  tner-evaluation -m "cner_output/model/${1}/${2}/best_model" -e "cner_output/model/${1}/${2}/best_model/eval/metric_span.test_2021.json" --dataset-split "test_2021" --span-detection-mode
  tner-evaluation -m "cner_output/model/${1}/${2}/best_model" -e "cner_output/model/${1}/${2}/best_model/eval/metric_span.test_2020.json" --dataset-split "test_2020" --span-detection-mode
}

finetuning (){
    HF_MODEL=${1}
    MODEL_NAME=${2}
    ALIAS=${3}
    tner-train-search -m "${HF_MODEL}" -c "cner_output/model/baseline/${MODEL_NAME}" -d "tner/tweetner7" -e 30 --epoch-partial 10 -b 32 \
      --max-length 128 --lr-warmup-step-ratio 0.15 0.3 --crf 0 1 -g 1 --weight-decay 1e-7 \
      --dataset-split-train 'train_2020' --dataset-split-valid 'validation_2020'
    evaluate_model "baseline" "${MODEL_NAME}"
    tner-push-to-hub -o "tner" -m "cner_output/model/baseline/${MODEL_NAME}/best_model" -a "${ALIAS}"
}

finetuning 'bert-base-cased' 'bert_base' 'bert-base-tweetner7-2020'
finetuning 'bert-large-cased' 'bert_large' 'bert-large-tweetner7-2020'
finetuning 'vinai/bertweet-base' 'bertweet_base' 'bertweet-base-tweetner7-2020'
finetuning 'vinai/bertweet-large' 'bertweet_large' 'bertweet-large-tweetner7-2020'
finetuning 'roberta-base' 'roberta_base' 'roberta-base-tweetner7-2020'
finetuning 'roberta-large' 'roberta_large' 'roberta-large-tweetner7-2020'
finetuning 'cardiffnlp/twitter-roberta-base-2019-90m' 't_roberta_base_2019' 'twitter-roberta-base-2019-90m-tweetner7-2020'
finetuning 'cardiffnlp/twitter-roberta-base-dec2020' 't_roberta_base_dec2020' 'twitter-roberta-base-dec2020-tweetner7-2020'
finetuning 'cardiffnlp/twitter-roberta-base-dec2021' 't_roberta_base_dec2021' 'twitter-roberta-base-dec2021-tweetner7-2020'
```

- Finetuning on random training/validation set

```shell
finetuning_random_split (){
    HF_MODEL=${1}
    MODEL_NAME=${2}
    ALIAS=${3}
    tner-train-search -m "${HF_MODEL}" -c "cner_output/model/random_split/${MODEL_NAME}" -d "tner/tweetner7" -e 30 --epoch-partial 10 -b 32 \
      --max-length 128 --lr-warmup-step-ratio 0.15 0.3 --crf 0 1 -g 1 --weight-decay 1e-7 \
      --dataset-split-train 'train_random' --dataset-split-valid 'validation_random'
    evaluate_model "random_split" "${MODEL_NAME}"
    tner-push-to-hub -o "tner" -m "cner_output/model/random_split/${MODEL_NAME}/best_model" -a "${ALIAS}"
}

finetuning_random_split 'vinai/bertweet-base' 'bertweet_base' 'bertweet-base-tweetner7-random'
finetuning_random_split 'vinai/bertweet-large' 'bertweet_large' 'bertweet-large-tweetner7-random'
finetuning_random_split 'bert-base-cased' 'bert_base' 'bert-base-tweetner7-random'
finetuning_random_split 'bert-large-cased' 'bert_large' 'bert-large-tweetner7-random'
finetuning_random_split 'roberta-base' 'roberta_base' 'roberta-base-tweetner7-random'
finetuning_random_split 'roberta-large' 'roberta_large' 'roberta-large-tweetner7-random'
finetuning_random_split 'cardiffnlp/twitter-roberta-base-2019-90m' 't_roberta_base_2019' 'twitter-roberta-base-2019-90m-tweetner7-random'
finetuning_random_split 'cardiffnlp/twitter-roberta-base-dec2020' 't_roberta_base_dec2020' 'twitter-roberta-base-dec2020-tweetner7-random'
finetuning_random_split 'cardiffnlp/twitter-roberta-base-dec2021' 't_roberta_base_dec2021' 'twitter-roberta-base-dec2021-tweetner7-random'
```

- Finetuning on 2021 training/validation set
```shell
finetuning_2021 () {
    HF_MODEL=${1}
    MODEL_NAME=${2}
    ALIAS=${3}
    
    # Fine-tuning on 2021
    tner-train-search -m "${HF_MODEL}" -c "cner_output/model/baseline_2021/${MODEL_NAME}" -d "tner/tweetner7" -e 30 --epoch-partial 10 -b 32 \
      --max-length 128 --lr-warmup-step-ratio 0.15 0.3 --crf 0 1 -g 1 --weight-decay 1e-7 \
      --dataset-split-train 'train_2021' --dataset-split-valid 'validation_2021' 
    evaluate_model "baseline_2021" "${MODEL_NAME}"
    tner-push-to-hub -o "tner" -m "cner_output/model/baseline_2021/${MODEL_NAME}/best_model" -a "${ALIAS}-2021"
    
    # Fine-tuning on 2021 + 2020
    tner-train-search -m "${HF_MODEL}" -c "cner_output/model/baseline_2021/${MODEL_NAME}_concat" -d "tner/tweetner7" -e 30 --epoch-partial 10 -b 32 \
      --max-length 128 --lr-warmup-step-ratio 0.15 0.3 --crf 0 1 -g 1 --weight-decay 1e-7 \
      --dataset-split-train 'train_all' --dataset-split-valid 'validation_2021' 
    evaluate_model "baseline_2021" "${MODEL_NAME}_concat"
    tner-push-to-hub -o "tner" -m "cner_output/model/baseline_2021/${MODEL_NAME}_concat/best_model" -a "${ALIAS}-2020-2021-concat"
    
    # Fine-tuning on 2020 --> 2021 
    tner-train-search -m "tner/${ALIAS}-2020" -c "cner_output/model/baseline_2021/${MODEL_NAME}_continuous" -d "tner/tweetner7" -e 30 --epoch-partial 10 -b 32 \
      --max-length 128 --lr-warmup-step-ratio 0.15 0.3 --crf 0 1 -g 1 --weight-decay 1e-7 \
      --dataset-split-train 'train_2021' --dataset-split-valid 'validation_2021' 
    evaluate_model "baseline_2021" "${MODEL_NAME}_continuous"
    tner-push-to-hub -o "tner" -m "cner_output/model/baseline_2021/${MODEL_NAME}_continuous/best_model" -a "${ALIAS}-2020-2021-continuous"
}

finetuning_2021 'bert-base-cased' 'bert_base' 'bert-base-tweetner'
finetuning_2021 'bert-large-cased' 'bert_large' 'bert-large-tweetner'
finetuning_2021 'vinai/bertweet-base' "bertweet_base" "bertweet-base-tweetner"
finetuning_2021 'vinai/bertweet-large' "bertweet_large" "bertweet-large-tweetner"
finetuning_2021 "roberta-base" "roberta_base" "roberta-base-tweetner"
finetuning_2021 "roberta-large" "roberta_large" "roberta-large-tweetner"
finetuning_2021 'cardiffnlp/twitter-roberta-base-2019-90m' 't_roberta_base_2019' 'twitter-roberta-base-2019-90m-tweetner'
finetuning_2021 'cardiffnlp/twitter-roberta-base-dec2020' 't_roberta_base_dec2020' 'twitter-roberta-base-dec2020-tweetner'
finetuning_2021 'cardiffnlp/twitter-roberta-base-dec2021' 't_roberta_base_dec2021' 'twitter-roberta-base-dec2021-tweetner'
```

## Model Fine-tuning (self-labeling)
- Label generation
Generate pseudo label on the extra sets
```shell
# generate self-labels
mkdir tweet_ner
ALIAS="twitter-roberta-base-dec2021"
for ALIAS in "roberta-large" "roberta-base" "bertweet-base" "bertweet-large" "bert-base" "bert-large" "twitter-roberta-base-2019-90m" "twitter-roberta-base-dec2020" "twitter-roberta-base-dec2021" 
  do
  python ner_model_labeling.py -m "tner/${ALIAS}-tweetner7-2020" -e "tweet_ner/2020.extra.${ALIAS}-2020.txt" --split "extra_2020" -b 64
  python ner_model_labeling.py -m "tner/${ALIAS}-tweetner7-2020" -e "tweet_ner/2021.extra.${ALIAS}-2020.txt" --split "extra_2021" -b 64
  done
```
or download the cached self-labeled data.
```shell
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/tweet_ner_extra_selflabeled.tar.gz
tar -xzf tweet_ner_extra_selflabeled.tar.gz
rm -rf tweet_ner_extra_selflabeled.tar.gz
mv tweet_ner_extra_selflabeled/* tweet_ner
```
Download the IOB-formatted TweetNER7 dataset too.
```shell
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/tweet_ner.tar.gz
tar -xzf tweet_ner.tar.gz
rm -rf tweet_ner.tar.gz
```

- Model fine-tuning
```shell

finetuning_selftraining () {
    HF_MODEL=${1}
    MODEL_NAME=${2}
    ALIAS=${3}
    SL_YEAR=${4}
    
    # Fine-tuning on 2021
    DATA_PATH='{"train": "' "tweet_ner/${SL_YEAR}.extra.${ALIAS}-2020.txt" '", "validation": "tweet_ner/2020.dev.txt"}' 
    tner-train-search -m "${HF_MODEL}" -c "cner_output/model/self_training_${SL_YEAR}/${MODEL_NAME}" -e 30 --epoch-partial 10 -b 32 \
      --max-length 128 --lr-warmup-step-ratio 0.15 0.3 --crf 0 1 -g 1 --weight-decay 1e-7 -l "${DATA_PATH}"
    evaluate_model "self_training_${SL_YEAR}" "${MODEL_NAME}"
    tner-push-to-hub -o "tner" -m "cner_output/model/self_training_${SL_YEAR}/${MODEL_NAME}/best_model" -a "${ALIAS}-tweetner7-selflabel${SL_YEAR}"
    
    # Fine-tuning on 2021 + 2020
    cat "tweet_ner/2020.train.txt" "tweet_ner/${SL_YEAR}.extra.${ALIAS}-2020.txt" > "tweet_ner/2020_${SL_YEAR}.extra.${ALIAS}-2020.txt"
    DATA_PATH='{"train": "' "tweet_ner/2020_${SL_YEAR}.extra.${ALIAS}-2020.txt" '", "validation": "tweet_ner/2020.dev.txt"}'
    tner-train-search -m "${HF_MODEL}" -c "cner_output/model/self_training_${SL_YEAR}/${MODEL_NAME}_concat" -e 30 --epoch-partial 10 -b 32 \
      --max-length 128 --lr-warmup-step-ratio 0.15 0.3 --crf 0 1 -g 1 --weight-decay 1e-7 -l "${DATA_PATH}"      
    evaluate_model "self_training_${SL_YEAR}" "${MODEL_NAME}_concat"
    tner-push-to-hub -o "tner" -m "cner_output/model/self_training_${SL_YEAR}/${MODEL_NAME}_concat/best_model" -a "${ALIAS}-tweetner7-2020-selflabel${SL_YEAR}-concat"
    
    # Fine-tuning on 2020 --> 2021 
    DATA_PATH='{"train": "' "tweet_ner/${SL_YEAR}.extra.${ALIAS}-2020.txt" '", "validation": "tweet_ner/2020.dev.txt"}'
    tner-train-search -m "tner/${ALIAS}-tweetner7-2020" -c "cner_output/model/self_training_${SL_YEAR}/${MODEL_NAME}_concat" -e 30 --epoch-partial 10 -b 32 \
      --max-length 128 --lr-warmup-step-ratio 0.15 0.3 --crf 0 1 -g 1 --weight-decay 1e-7  -l "${DATA_PATH}"      
    evaluate_model "self_training_${SL_YEAR}" "${MODEL_NAME}_continuous"
    tner-push-to-hub -o "tner" -m "cner_output/model/self_training_${SL_YEAR}/${MODEL_NAME}_continuous/best_model" -a "${ALIAS}-tweetner7-2020-selflabel${SL_YEAR}-continuous"
}
finetuning_selftraining "roberta-large" "roberta_large" "roberta-large" "2020"
finetuning_selftraining "roberta-large" "roberta_large" "roberta-large" "2021"
finetuning_selftraining "roberta-large" "roberta_large" "roberta-large" "2020"
finetuning_selftraining "roberta-large" "roberta_large" "roberta-large" "2021"
finetuning_selftraining 'cardiffnlp/twitter-roberta-base-2019-90m' 'twitter_roberta_base_2019_90m' 'twitter-roberta-base-2019-90m' "2020"
finetuning_selftraining 'cardiffnlp/twitter-roberta-base-2019-90m' 'twitter_roberta_base_2019_90m' 'twitter-roberta-base-2019-90m' "2021"
finetuning_selftraining 'cardiffnlp/twitter-roberta-base-dec2020' 'twitter_roberta_base_dec2020' 'twitter-roberta-base-dec2020' "2020"
finetuning_selftraining 'cardiffnlp/twitter-roberta-base-dec2020' 'twitter_roberta_base_dec2020' 'twitter-roberta-base-dec2020' "2021"
finetuning_selftraining 'cardiffnlp/twitter-roberta-base-dec2021' 'twitter_roberta_base_dec2021' 'twitter-roberta-base-dec2021' "2020"
finetuning_selftraining 'cardiffnlp/twitter-roberta-base-dec2021' 'twitter_roberta_base_dec2021' 'twitter-roberta-base-dec2021' "2021"
```

## Summarize Result
Get summary of model training.
```shell
python model_finetuning_results.py
```
Update README of huggingface uploaded models.
```shell
python update_readme.py
```

## Contextual Prediction Analysis
Following command runs the contextual prediction analysis (see the paper for more detail).
All the output given by the contextual prediction analysis experiment can be found at [output/contextual_prediction_analysis](output/contextual_prediction_analysis).
- Index Data
Create search index with RoBERTa Large fine-tuned NER model from scratch.
```shell
# index search pool
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/extra_tweets.csv.tar.gz
tar -xzf extra_tweets.csv.tar.gz
rm -rf extra_tweets.csv.tar.gz
python text_retriever.py -b 32 -i 'cner_output/index_extra/roberta_large' -n 'tner/roberta-large-tweetner7-2020' -s 'cambridgeltl/mirror-roberta-base-sentence-drophead' -f 'extra_tweets.csv'
# index search pool
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/tweet_ner_test2021_tweet_id_date.csv
python text_retriever.py -b 32 -i 'cner_output/index_test2021/roberta_large' -n 'tner/roberta-large-tweetner7-2020' -s 'cambridgeltl/mirror-roberta-base-sentence-drophead' -f 'tweet_ner_test2021_tweet_id_date.csv'
```

Or download the index from the repository.
```shell
mkdir -p cner_output/index_extra/roberta_large
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/extra_tweets_roberta_large_ner_prediction.json.tar.gz
tar -xzf extra_tweets_roberta_large_ner_prediction.json.tar.gz
mv ner_prediction.json cner_output/index_extra/roberta_large
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/extra_tweets_search_index.tar.gz
tar -xzf extra_tweets_search_index.tar.gz
mv search_index cner_output/index_extra/roberta_large/
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/extra_tweets_embedding.json.tar.gz
tar -xzf extra_tweets_embedding.json.tar.gz
mv embedding.json cner_output/index_extra/roberta_large/

mkdir -p cner_output/index_test2021/roberta_large
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/test2021_roberta_large_ner_prediction.json.tar.gz
tar -xzf test2021_roberta_large_ner_prediction.json.tar.gz
mv ner_prediction.json cner_output/index_test2021/roberta_large
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/test2021_search_index.tar.gz
tar -xzf test2021_search_index.tar.gz
mv search_index cner_output/index_test2021/roberta_large/
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/test2021_embedding.json.tar.gz
tar -xzf test2021_embedding.json.tar.gz
mv embedding.json cner_output/index_test2021/roberta_large/

rm -rf *.tar.gz
```

- Compute Context Features
Compute features with a query tweet, and the retrieved tweets.
```shell
python contextual_prediction_analysis.py
```
Or download the features and run the above script.
```shell
wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/roberta_large.search_cache.test2021.json.tar.gz
tar -xzf roberta_large.search_cache.test2021.json.tar.gz
mv search_cache.test2021.json cner_output/index_extra/roberta_large

wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/roberta_large.search_feature.test2021.json.tar.gz
tar -xzf roberta_large.search_feature.test2021.json.tar.gz
mv search_feature.test2021.json cner_output/index_extra/roberta_large

wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/roberta_large.analysis_td_cache.test2021.json.tar.gz
tar -xzf roberta_large.analysis_td_cache.test2021.json.tar.gz
mv analysis_td_cache.test2021.json cner_output/index_extra/roberta_large

wget https://huggingface.co/datasets/tner/label2id/resolve/main/tweetner7/roberta_large.analysis_td_cache.test2021.json.tar.gz
tar -xzf roberta_large.analysis_td_cache.test2021.json.tar.gz
mv analysis_td_cache.test2021.json cner_output/index_extra/roberta_large

rm -rf *.tar.gz
```


