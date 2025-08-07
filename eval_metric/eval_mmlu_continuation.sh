#!/bin/bash

MODEL_NAME="/data0/butao/cmpLlama/checkpoint/loralike_llama"
BASE_MODEL_NAME="/data1/model/llama3/meta-llama/Llama-3.2-1B"
BASELINE_MODEL_NAME="/data0/butao/cmpLlama/checkpoint/baseline_llama"
DTYPE="float"
TASKS="mmlu_continuation"
DEVICE="cuda:0"
BATCH_SIZE="auto:4"
OUTPUT_PATH="/data0/butao/cmpLlama/eval_metric/results/mmlu_continuation"

export PYTHONPATH=/data0/butao/cmpLlama:$PYTHONPATH

python eval_entry.py \
    --model hf \
    --model_args pretrained=$MODEL_NAME,dtype=$DTYPE \
    --tasks $TASKS \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_PATH \
    --wandb_args project="amp-llama-test",name="eval-loralike-llama-mmlu_continuation"