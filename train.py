import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import wandb
import torch
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import math

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from llama.configuration_llama_lp import LoraLikeLlamaConfig
from llama.modeling_llama_lp import LoraLikeLlamaForCausalLM

from transformers.models.llama import LlamaForCausalLM, LlamaConfig

def load_wiki_dataset():
    train_dataset = load_dataset("/data1/datasets/wikitext103/wikitext-103-raw-v1", split="train")
    val_dataset = load_dataset("/data1/datasets/wikitext103/wikitext-103-raw-v1", split="validation")
    test_dataset = load_dataset("/data1/datasets/wikitext103/wikitext-103-raw-v1", split="test")

    raw_wiki_datasets = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
            "valid": val_dataset,
        }
    )
    return raw_wiki_datasets

def load_pile_dataset():

    # Pile-CC 0.2964
    # Github 0.1021
    # StackExchange 0.1668
    # Wikipedia (en) 0.0956
    # PubMed Abstracts 0.1656
    # USPTO Backgrounds 0.0627
    # FreeLaw 0.0287
    # PubMed Central 0.0322
    # Enron Emails 0.0052
    # HackerNews 0.0089
    # NIH ExPorter 0.0101
    # ArXiv 0.0134
    # DM Mathematics 0.0108
    # Ubuntu IRC 0.0001
    # EuroParl 0.0007
    # PhilPapers 0.0004
    # Gutenberg (PG-19) 0.0004

    TOTAL_TRAIN_SAMPLES = 5899215
    TOTAL_VALID_SAMPLES = 179996
    TOTAL_TEST_SAMPLES = 180378

    RATIO = 0.1
    VALID_TEST_RATIO = RATIO / 30

    data_files = {
        "train": "/data1/datasets/lm-pretrain-corpus/Pile/pile-uncopyright/train-set/00.jsonl",
        "validation": "/data1/datasets/lm-pretrain-corpus/Pile/pile-uncopyright/val.jsonl",
        "test": "/data1/datasets/lm-pretrain-corpus/Pile/pile-uncopyright/test.jsonl"
    }

    train_dataset = load_dataset("json", data_files=data_files, split="train", streaming=False)
    val_dataset = load_dataset("json", data_files=data_files, split="validation", streaming=False)
    test_dataset = load_dataset("json", data_files=data_files, split="test", streaming=False)

    train_sample_size = RATIO * TOTAL_TRAIN_SAMPLES
    val_sample_size = VALID_TEST_RATIO * TOTAL_VALID_SAMPLES
    test_sample_size = VALID_TEST_RATIO * TOTAL_TEST_SAMPLES

    train_dataset = train_dataset.select(range(int(train_sample_size)))
    val_dataset = val_dataset.select(range(int(val_sample_size)))
    test_dataset = test_dataset.select(range(int(test_sample_size)))

    raw_pile_datasets = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
            "valid": val_dataset,
        }
    )
    return raw_pile_datasets


def count_subset_ratio():
    raw_pile_datasets=load_pile_dataset()
    print(raw_pile_datasets)
    classes = {}
    for sample in raw_pile_datasets['train']:
        if sample['meta']['pile_set_name'] not in classes:
            classes[sample['meta']['pile_set_name']] = 1
        else:
            classes[sample['meta']['pile_set_name']] += 1
    
    total = sum(classes.values())
    for k,v in classes.items():
        print(k, "{:.4f}".format(v/total))


def tokenize(element, tokenizer, context_length = 512):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        # stride=128,
        padding=False,
        return_tensors=None
    )

    input_batch = []    
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length >= 32:
            input_batch.append(input_ids)    

    return {"input_ids": input_batch}

def packed_tokenize(element, tokenizer, context_length = 512):
    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or tokenizer.sep_token_id
    assert eos_token_id is not None, "Need eos_token_id or equivalent"

    outputs = tokenizer(
        element["text"],
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    all_tokens = []
    for input_ids in outputs["input_ids"]:
        all_tokens.extend(input_ids + [eos_token_id])

    # if len(all_tokens) > 131072:
    #     all_tokens = all_tokens[:131072]

    total_length = len(all_tokens)
    input_ids = [
        all_tokens[i : i + context_length]
        for i in range(0, total_length - context_length + 1, context_length)
    ]

    return {"input_ids": input_ids}

def get_custom_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr, base_lr):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            warmup_lr = float(current_step) / float(max(1, num_warmup_steps))
            return warmup_lr
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        scaled = cosine_decay * (1 - min_lr / base_lr) + (min_lr / base_lr)
        return scaled

    return LambdaLR(optimizer, lr_lambda)

class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_custom_cosine_schedule_with_min_lr(
                optimizer or self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
                min_lr=5e-5,
                base_lr=self.args.learning_rate
            )
        return self.lr_scheduler

def main():

    context_length = 512
    model_path = "/data1/model/llama3/meta-llama/Llama-3.2-1B"
    model_save_path = "/data0/butao/cmpLlama/checkpoint/total_loralike_llama"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    partial_tokenize = partial(packed_tokenize, tokenizer=tokenizer, context_length=context_length)
    raw_pile_datasets = load_pile_dataset()
    tokenized_datasets = raw_pile_datasets.map(partial_tokenize, batched=True, batch_size=512, remove_columns=raw_pile_datasets["train"].column_names)
    # tokenized_datasets = tokenized_datasets.shuffle(seed=42)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    config = LoraLikeLlamaConfig.from_pretrained(model_path)
    config.architectures = ["LoraLikeLlamaForCausalLM"]
    config.model_type = "loralike_llama"
    model = LoraLikeLlamaForCausalLM(config)

    # config = LlamaConfig.from_pretrained(model_path)
    # model = LlamaForCausalLM(config)

    wandb.init(
        project="amp-llama-test",
        name="pile-total-loralike-llama-test-training",
        resume="never"
    )

    args = TrainingArguments(
        output_dir="/data0/butao/cmpLlama/output/test",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=200,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        bf16=True,
        push_to_hub=False,
        save_strategy="no",
        report_to="wandb",
        run_name="pile-total-loralike-llama-test-training",
    )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )


    trainer.train()

    trainer.model.save_pretrained(model_save_path, safe_serialization=False)
    tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    main()