# loralike-llama

1. Train
    - Train Dataset
        - EleutherAI/pile
    - Base Model
        - meta-llama/Llama-3.2-1B
    - `python train.py`
2. Evaluation
    - Use [lm-evaluate-harness](https://github.com/EleutherAI/lm-evaluation-harness.git)
    - Benchmark Datasets
        - allenai/ai2_arc
        - Rowan/hellaswag
        - hails/mmlu_no_train
        - baber/piqa
        - cais/mmlu
        - EleutherAI/lambada_openai
    - Modify lm-evaluate-harness/lm_eval/tasks/*.ymal `dataset_path` to use local dataset
        - arc/arc_easy.yaml
        - hellaswag/hellaswag.yaml
        - lambada/lambada_openai.yaml
        - mmlu/continuation/_continuation_template_yaml
        - piqa/piqa.yaml
        - winogrande/default.yaml
    - `bash eval_*.sh`