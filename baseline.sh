python3 src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \

python3 src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json

python3 src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json

python3 src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python3 src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json

python3 src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
