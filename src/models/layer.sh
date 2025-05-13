python _layer.py \
  --model_dir ./pruned-gpt2 \
  --prefill_tokens 128 \
  --decode_length 10 \
  --num_threads 16 \
  --device cpu \
  --perf
