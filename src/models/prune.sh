#!/bin/bash

# python _prune.py mistral-7b-instruct-v0.1.Q4_K_S.gguf pruned-mistral-7b-instruct-v0.1.Q4_K_S.gguf.gguf \
#     --method unstructured --amount 0.5


python _prune.py \
  --model_name gpt2 \
  --output_dir ./pruned-gpt2 \
  --method unstructured \
  --amount 0.3
