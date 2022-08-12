
model_development () {
  DATA=${1}
  MODEL=${2}
  BATCH=${3}
  GRAD_1=${4}
  GRAD_2=${5}
  export MODEL_ALIAS="${MODEL##*/}"
  tner-train-search -m "${MODEL}" -c "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}" -d "tner/${DATA}" -e 15 --epoch-partial 5 --n-max-config 3 -b "${BATCH}" -g "${GRAD_1}" "${GRAD_2}" --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
  tner-evaluate -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -e "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model/eval/metric.json" -d "tner/${DATA}" -b "${BATCH}" --return-ci
  tner-evaluate -m "tner_ckpt/${DATA}_roberta_large/best_model" -e "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model/eval/metric_span.json" -d "tner/${DATA}" -b "${BATCH}" --return-ci --span-detection-mode
#  tner-push-to-hub -o "tner" -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -a "${MODEL_ALIAS}-${DATA//_/-}"
}


model_development "conll2003" "roberta-large" 64 1 2
model_development "ontonotes5" "roberta-large" 64 1 2
model_development "bionlp2004" "roberta-large" 64 1 2
model_development "bc5cdr" "roberta-large" 64 1 2
model_development "fin" "roberta-large" 64 1 2
model_development "wnut2017" "roberta-large" 64 1 2
model_development "tweebank_ner" "roberta-large" 64 1 2
model_development "btc" "roberta-large" 64 1 2
model_development "ttc" "roberta-large" 64 1 2
model_development "mit_restaurant" "roberta-large" 64 1 2
model_development "mit_movie_trivia" "roberta-large" 64 1 2

model_development "conll2003" "microsoft/deberta-v3-large" 16 4 8
model_development "ontonotes5" "microsoft/deberta-v3-large" 16 4 8
model_development "bionlp2004" "microsoft/deberta-v3-large" 16 4 8
model_development "bc5cdr" "microsoft/deberta-v3-large" 16 4 8
model_development "fin" "microsoft/deberta-v3-large" 16 4 8
model_development "wnut2017" "microsoft/deberta-v3-large" 16 4 8
model_development "tweebank_ner" "microsoft/deberta-v3-large" 16 4 8
model_development "btc" "microsoft/deberta-v3-large" 16 4 8
model_development "ttc" "microsoft/deberta-v3-large" 16 4 8
model_development "mit_restaurant" "microsoft/deberta-v3-large" 16 4 8
model_development "mit_movie_trivia" "microsoft/deberta-v3-large" 16 4 8


git clone https://huggingface.co/tner/deberta-v3-large-bc5cdr
tner-push-to-hub -o "tner" -m "deberta-v3-large-bc5cdr" -a "deberta-v3-large-bc5cdr"
git clone https://huggingface.co/tner/deberta-v3-large-conll2003
tner-push-to-hub -o "tner" -m "deberta-v3-large-conll2003" -a "deberta-v3-large-conll2003"
git clone https://huggingface.co/tner/bertweet-large-wnut2017
tner-push-to-hub -o "tner" -m "bertweet-large-wnut2017" -a "bertweet-large-wnut2017"
git clone https://huggingface.co/tner/roberta-large-conll2003
tner-push-to-hub -o "tner" -m "roberta-large-conll2003" -a "roberta-large-conll2003"
git clone https://huggingface.co/tner/roberta-large-bc5cdr
tner-push-to-hub -o "tner" -m "roberta-large-bc5cdr" -a "roberta-large-bc5cdr"
git clone https://huggingface.co/tner/deberta-large-wnut2017
tner-push-to-hub -o "tner" -m "deberta-large-wnut2017" -a "deberta-large-wnut2017"
git clone https://huggingface.co/tner/deberta-v3-large-wnut2017
tner-push-to-hub -o "tner" -m "deberta-v3-large-wnut2017" -a "deberta-v3-large-wnut2017"
git clone https://huggingface.co/tner/roberta-large-wnut2017
tner-push-to-hub -o "tner" -m "roberta-large-wnut2017" -a "roberta-large-wnut2017"
git clone https://huggingface.co/tner/deberta-v3-large-tweebank-ner
tner-push-to-hub -o "tner" -m "deberta-v3-large-tweebank-ner" -a "deberta-v3-large-tweebank-ner"
git clone https://huggingface.co/tner/roberta-large-tweebank-ner
tner-push-to-hub -o "tner" -m "roberta-large-tweebank-ner" -a "roberta-large-tweebank-ner"

export DATA="conll2003"
export DATA="ontonotes5"
export DATA="wnut2017"
export DATA="bc5cdr"
export DATA="btc"
export DATA="tweebank_ner"

export MODEL="roberta-large"
export MODEL="microsoft/deberta-large"

export MODEL="microsoft/deberta-v3-large"
export MODEL="vinai/bertweet-large"


export MODEL_ALIAS="${MODEL##*/}"
tner-push-to-hub -o "tner" -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -a "${MODEL_ALIAS}-${DATA//_/-}"

tner-train-search -m "${MODEL}" -c "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}" -d "tner/${DATA}" -b 64 -e 15 --epoch-partial 5 --n-max-config 3 -b 64 -g 1 2 --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
tner-evaluate -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -e "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model/eval/metric.json" -d "tner/${DATA}" -b 64 --return-ci
tner-evaluate -m "tner_ckpt/${DATA}_roberta_large/best_model" -e "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model/eval/metric_span.json" -d "tner/${DATA}" -b 64 --return-ci --span-detection-mode

tner-push-to-hub -o "tner" -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -a "${MODEL_ALIAS}-${DATA//_/-}"




tner-train-search -m "microsoft/deberta-v3-large" -c "tner_ckpt/${DATA}_deberta_large" -d "tner/${DATA}" -e 15 --epoch-partial 5 --n-max-config 3 -b 16 -g 4 8 --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
tner-evaluate -m "tner_ckpt/${DATA}_deberta_large/best_model" -e "tner_ckpt/${DATA}_deberta_large/best_model/eval/metric.json" -d "tner/${DATA}" -b 32 --return-ci
tner-evaluate -m "tner_ckpt/${DATA}_deberta_large/best_model" -e "tner_ckpt/${DATA}_deberta_large/best_model/eval/metric_span.json" -d "tner/${DATA}" -b 32 --return-ci --span-detection-mode

tner-train-search -m "microsoft/deberta-large" -c "tner_ckpt/${DATA}_deberta_large_v1" -d "tner/${DATA}" -e 15 --epoch-partial 5 --n-max-config 3 -b 16 -g 4 8 --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
tner-evaluate -m "tner_ckpt/${DATA}_deberta_large_v1/best_model" -e "tner_ckpt/${DATA}_deberta_large_v1/best_model/eval/metric.json" -d "tner/${DATA}" -b 32 --return-ci
tner-evaluate -m "tner_ckpt/${DATA}_deberta_large_v1/best_model" -e "tner_ckpt/${DATA}_deberta_large_v1/best_model/eval/metric_span.json" -d "tner/${DATA}" -b 32 --return-ci --span-detection-mode

tner-train-search -m "vinai/bertweet-large" -c "tner_ckpt/${DATA}_bertweet_large" -d "tner/${DATA}" -e 15 --epoch-partial 5 --n-max-config 3 -b 16 -g 4 8 --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
tner-evaluate -m "tner_ckpt/${DATA}_bertweet_large/best_model" -e "tner_ckpt/${DATA}_bertweet_large/best_model/eval/metric.json" -d "tner/${DATA}" -b 32 --return-ci
tner-evaluate -m "tner_ckpt/${DATA}_bertweet_large/best_model" -e "tner_ckpt/${DATA}_bertweet_large/best_model/eval/metric_span.json" -d "tner/${DATA}" -b 32 --return-ci --span-detection-mode

