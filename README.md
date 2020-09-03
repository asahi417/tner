# Named Entity Recognition
Finetuning [transformers](https://github.com/huggingface/transformers) on Named Entity Recognition (NER).

## Get started
```bash
git clone https://github.com/asahi417/transformers-ner
cd transformers-ner
pip install -r requirement.txt
```

- Wikiann dataset
First create a download folder with `mkdir -p ./cache` in the root of this project.
You then need to manually download panx_dataset (for NER) from [here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN?_encoding=UTF8&%2AVersion%2A=1&%2Aentries%2A=0&mgh=1) 
(note that it will download as AmazonPhotos.zip) to the download directory. Finally, run the following command to download the remaining datasets:

## Scripts
### Train model

```bash
python example_train.py \
    --checkpoint-dir ./ckpt \
    -d ontonote5
```

### Test model
```bash
python example_train.py \
    --test \
    -c {path-to-checkpoint}
```

### Run APP
```bash
export MODEL_CKPT={path-to-checkpoint}
uvicorn app:app --reload --log-level debug --host 0.0.0.0 --port 8000
```

## Result
Rows are test dataset and columns are training dataset.

### Domain transfer 
- English

| train\test | OntoNote5   | Conll 2003  | Pan-en       | Movie | Restaurant |
|------------|-------------|-------------|--------------|-------|------------|
| OntoNote5  | 0.87 (0.89) | 0.52 (0.52) | 0.40 (0.41)  | 0.19  | 0.01       |
| Conll 2003 | 0.34 (0.34) | 0.95 (0.91) | 0.53 (0.52)  | 0.00  | 0.00       |
| Pan-en     | 0.19 (0.20) | 0.51 (0.52) | 0.84 (0.83)  | 0.00  | 0.02       |
| Movie      | 0.02 (0.02) | 0.00 (0.00) | 0.00 (0.00)  | 0.70  | 0.00       |
| Restaurant | 0.02 (0.02) | 0.13 (0.14) | 0.14 (0.12)  | 0.00  | 0.79       |

- English (ignore entity type)

| train\test | OntoNote5   | Conll 2003  | Pan-en       | Movie | Restaurant |
|------------|-------------|-------------|--------------|-------|------------|
| OntoNote5  | 0.91 (0.91) | 0.58 (0.58) | 0.46 (0.46)  | 0.20  | 0.01       |
| Conll 2003 | 0.61 (0.61) | 0.98 (0.96) | 0.62 (0.61)  | 0.00  | 0.00       |
| Pan-en     | 0.41 (0.41) | 0.72 (0.73) | 0.93 (0.93)  | 0.00  | 0.08       |
| Movie      | 0.02 (0.02) | 0.00 (0.00) | 0.00 (0.00)  | 0.73  | 0.00       |
| Restaurant | 0.15 (0.15) | 0.20 (0.20) | 0.19 (0.18)  | 0.00  | 0.83       |

- Japanese

| train\test | Pan-ja      | cogent-ja   |
|------------|-------------|-------------|
| Pan-ja     | 0.83 (0.83) | 0.29 (0.28) |
| cogent-ja  | 0.50 (0.50) | 0.78 (0.80) |

- Japanese (ignore entity type)

| train\test | Pan-ja      | cogent-ja   |
|------------|-------------|-------------|
| Pan-ja     | 0.88 (0.88) | 0.34 (0.33) |
| cogent-ja  | 0.60 (0.60) | 0.87 (0.87) |

### Cross-lingual transfer

- wikipedia 

| train\test | Pan-en      | Pan-ja      | Pan-ru      |
|------------|-------------|-------------|-------------|
| Pan-en     | 0.84 (0.83) | 0.37 (0.37) | 0.66 (0.65) |
| Pan-ja     | 0.54 (0.53) | 0.83 (0.83) | 0.54 (0.53) |
| Pan-ru     | 0.55 (0.55) | 0.43 (0.43) | 0.89 (0.88) |

- wikipedia (ignore entity type)

| train\test | Pan-en      | Pan-ja      | Pan-ru      |
|------------|-------------|-------------|-------------|
| Pan-en     | 0.93 (0.93) | 0.44 (0.44) | 0.82 (0.81) |
| Pan-ja     | 0.62 (0.62) | 0.88 (0.88) | 0.62 (0.62) |
| Pan-ru     | 0.71 (0.71) | 0.57 (0.57) | 0.94 (0.94) |

- News

| train\test | cogent-en   | cogent-ja   |
|------------|-------------|-------------|
| cogent-en  | 0.83 (0.89) | 0.25 (0.25) |
| cogent-ja  | 0.51 (0.62) | 0.79 (0.80) |

- News (ignore entity type)

| train\test | cogent-en   | cogent-ja   |
|------------|-------------|-------------|
| cogent-en  | 0.84 (0.93) | 0.30 (0.31) |
| cogent-ja  | 0.62 (0.73) | 0.87 (0.87) |
