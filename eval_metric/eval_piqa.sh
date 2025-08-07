#!/bin/bash

MODEL_NAME="/data0/butao/cmpLlama/checkpoint/loralike_llama"
BASELINE_MODEL_NAME="/data0/butao/cmpLlama/checkpoint/baseline_llama"
META_MODEL_NAME="/data1/model/llama3/meta-llama/Llama-3.2-1B"
DTYPE="float"
TASKS="piqa"
DEVICE="cuda:0"
BATCH_SIZE="auto:4"
OUTPUT_PATH="/data0/butao/cmpLlama/eval_metric/results/piqa"


export PYTHONPATH=/data0/butao/cmpLlama:$PYTHONPATH

python eval_entry.py \
    --model hf \
    --model_args pretrained=$META_MODEL_NAME,dtype=$DTYPE \
    --tasks $TASKS \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --wandb_args project="amp-llama-test",name="eval-meta-llama-piqa"