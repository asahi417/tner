# Released Model
We release 46 finetuned models on [transformers model hub](https://huggingface.co/models?search=asahi417/tner).
All the models are [XLM-R](https://arxiv.org/pdf/1911.02116.pdf), finetuned on named entity recognition task with TNER.

## Model Name
Model name is organized as `asahi417/tner-xlm-roberta-{model_type}-{dataset}`, where `model_type` is either `base` or `large` and `dataset` corresponds to 
the alias of [dataset](https://github.com/asahi417/tner/blob/master/README.md#datasets). In addition to individual model, we train on the English merged dataset by 
concatenating all the English NER dataset, that denoted as `all-english`.
We also release model finetuned on lowercased dataset, which is called `asahi417/tner-xlm-roberta-{model_type}-uncased-{dataset}`.

For example
- `asahi417/tner-xlm-roberta-large-ontonotes5`: XLM-R large model finetuned on Ontonotes5 dataset
- `asahi417/tner-xlm-roberta-base-uncased-conll2003`: XLM-R base model finetuned on lowercased CoNLL2003 dataset
- `asahi417/tner-xlm-roberta-large-all-english`: XLM-R large model finetuned on all English datasets

The training parameter used in TNER to finetune each model, is stored at `https://huggingface.co/{model-name}/blob/main/parameter.json`.
Eg) The training parameter of `asahi417/tner-xlm-roberta-large-all-english` is [here](https://huggingface.co/asahi417/tner-xlm-roberta-large-all-english/blob/main/parameter.json).

## Usage
### To use with TNER

```python
import tner
model = tner.TransformersNER("model-name")
```

### To use with transformers
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForTokenClassification.from_pretrained("model-name")
```


## Experimental Results
Here we show a few experimental results on our released XLM-R models with in-domain/cross-domain/cross-lingual setting. Firstly, we report in-domain baseline on each dataset, where the metrics are quite close to, or even outperform current SoTA (Oct, 2020).
Through the section, we use test F1 score. 

| Dataset            | Recall | Precision | F1    |  SoTA F1  |                    SoTA reference                    |
|:------------------:|:------:|:---------:|:-----:|:---------:|:----------------------------------------------------:|
| `ontonotes5`       | 90.56  | 87.75     | 89.13 | 92.07     | [BERT-MRC-DSC](https://arxiv.org/pdf/1911.02855.pdf) |
| `wnut2017`         | 51.53  | 67.85     | 58.58 | 50.03     | [CrossWeigh](https://www.aclweb.org/anthology/D19-1519.pdf)  |
| `conll2003`        | 93.86  | 92.09     | 92.97 | 94.30     | [LUKE](https://arxiv.org/pdf/2010.01057v1.pdf)       | 
| `panx_dataset/en`  | 84.78  | 83.27     | 84.02 | 84.8      | [mBERT](https://arxiv.org/pdf/2005.00052.pdf)        |
| `panx_dataset/ja`  | 87.96  | 85.17     | 86.54 | - | - |
| `panx_dataset/ru`  | 90.7   | 89.45     | 90.07 | - | - |
| `fin`              | 82.56  | 71.24     | 76.48 | - | - |  
| `bionlp2004`       | 79.63  | 69.78     | 74.38 | - | - |
| `bc5cdr`           | 90.36  | 87.02     | 88.66 | - | - |
| `mit_restaurant`   | 80.64  | 78.64     | 79.63 | - | - |
| `mit_movie_trivia` | 73.14  | 69.42     | 71.23 | - | - |

Then, we run evaluation of each model on different dataset to see its domain adaptation capacity in English.
As the entities are different among those dataset, we can't compare them by ordinary entity-type F1 score like above.
Due to that, we employ entity-span f1 score for our metric of domain adaptation. 

|  Train\Test        | `ontonotes5` | `conll2003` | `wnut2017` | `panx_dataset/en` | `bionlp2004` | `bc5cdr` | `fin`   | `mit_restaurant` | `mit_movie_trivia` | 
|:------------------:|:----------:|:---------:|:--------:|:---------------:|:----------:|:------:|:-----:|:--------------:|:----------------:| 
| `ontonotes5`       | _91.69_    | 65.45     | 53.69    | 47.57           | 0.0        | 0.0    | 18.34 | 2.47           | 88.87            | 
| `conll2003`        | 62.24      | _96.08_   | 69.13    | 61.7            | 0.0        | 0.0    | 22.71 | 4.61           | 0.0              | 
| `wnut2017`         | 41.89      | 85.7      | _68.32_  | 54.52           | 0.0        | 0.0    | 20.07 | 15.58          | 0.0              | 
| `panx_dataset/en`  | 32.81      | 73.37     | 53.69    | _93.41_         | 0.0        | 0.0    | 12.25 | 1.16           | 0.0              | 
| `bionlp2004`       | 0.0        | 0.0       | 0.0      | 0.0             | _79.04_    | 0.0    | 0.0   | 0.0            | 0.0              | 
| `bc5cdr`           | 0.0        | 0.0       | 0.0      | 0.0             | 0.0        | _88.88_| 0.0   | 0.0            | 0.0              | 
| `fin`              | 48.25      | 73.21     | 60.99    | 58.99           | 0.0        | 0.0    | _82.05_| 19.73         | 0.0              | 
| `mit_restaurant`   | 5.68       | 18.37     | 21.2     | 24.07           | 0.0        | 0.0    | 18.06 | _83.4_         | 0.0              | 
| `mit_movie_trivia` | 11.97      | 0.0       | 0.0      | 0.0             | 0.0        | 0.0    | 0.0   | 0.0            | _73.1_           | 


One can see that none of the models transfers well on the other dataset, which indicates the difficulty of domain transfer in NER task.
Now, we train NER model on all the dataset and report the result.
Each models were trained on all datasets for `5000`, `10000`, and `15000` steps.
As you can see, the accuracy is altogether close to what attained from from single dataset model, indicating `xlm-roberta-large` at least can learn all the features in each domain.  

|                 | `ontonotes5` | `conll2003` | `wnut2017` | `panx_dataset/en` | `bionlp2004` | `bc5cdr` | `fin`   | `mit_restaurant` | `mit_movie_trivia` | 
|:---------------:|:------------:|:-----------:|:----------:|:-----------------:|:------------:|:--------:|:-------:|:----------------:|:------------------:| 
| `all_english`   | 87.91        | 89.8        | 55.48      | 82.29             | 73.76        | 84.25    | 74.77   | 81.44            | 72.33              | 

Finally, we show cross-lingual transfer metrics over a few `WikiAnn` datasets.

|  Train\Test       | `panx_dataset/en` | `panx_dataset/ja` | `panx_dataset/ru` | 
|:-----------------:|:-----------------:|:-----------------:|:-----------------:| 
| `panx_dataset/en` | 84.02             | 46.37             | 73.18             | 
| `panx_dataset/ja` | 53.6              | 86.54             | 45.75             | 
| `panx_dataset/ru` | 60.49             | 53.38             | 90.07             | 
