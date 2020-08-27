# Transformer Finetunings
Finetuning [transformers](https://github.com/huggingface/transformers) on supervisions.

```bash
git clone
cd 
pip install -r requirement.txt
```

## Question & Answering (Squad v1, v2)


```bash
python ./transformers_qa.py -d squad-v2 --with-negatives
```


## Named-Entity-Recognition
```
uvicorn app_ner:app --reload --log-level debug
```


## Text Classification
