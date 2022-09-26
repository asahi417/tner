
model_development () {
  DATA=${1}
  MODEL=${2}
  BATCH=${3}
  GRAD_1=${4}
  GRAD_2=${5}
  export MODEL_ALIAS="${MODEL##*/}"
  tner-train-search -m "${MODEL}" -c "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}" -d "tner/${DATA}" -e 15 --epoch-partial 5 --n-max-config 3 -b "${BATCH}" -g "${GRAD_1}" "${GRAD_2}" --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
  tner-evaluate -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -e "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model/eval/metric.json" -d "tner/${DATA}" -b "${BATCH}" --return-ci
  tner-evaluate -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -e "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model/eval/metric_span.json" -d "tner/${DATA}" -b "${BATCH}" --return-ci --span-detection-mode
  tner-push-to-hub -o "tner" -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -a "${MODEL_ALIAS}-${DATA//_/-}"
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

model_development "wnut2017" "vinai/bertweet-large" 16 4 8
model_development "wnut2017" "microsoft/deberta-large" 16 4 8
