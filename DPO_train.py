import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import wandb
from datasets import Dataset
from transformers import HfArgumentParser, TrainingArguments

from trl import DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
PatchDPOTrainer()

from prepare_DPO_reject import load_dataframes


@dataclass()
class ModelParameters():
    model_name: str
    max_seq_length: int
    dtype: Optional[str]
    load_in_4bit: bool


@dataclass()
class PeftConfig():
    r: int
    target_modules: list[str]
    lora_alpha: int
    lora_dropout: int
    bias: str
    use_gradient_checkpointing: str
    random_state: int
    use_rslora: bool
    loftq_config: None


@dataclass()
class DatasetConfig():
    mimic_version: str
    heading_type: str
    ref_model_name: str
    tokenizer_name: str


@dataclass()
class TrainingConfig():
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    num_train_epochs: int
    eval_strategy: str
    save_strategy: str
    group_by_length: bool
    learning_rate: float
    logging_steps: int
    optim: str
    weight_decay: float
    lr_scheduler_type: str
    seed: int
    max_grad_norm: float
    output_dir: str
    fp16: bool = field(default=not is_bfloat16_supported())
    bf16: bool = field(default=is_bfloat16_supported())


@dataclass()
class DPOConfig():
    max_prompt_length : int
    max_response_length : int
    ref_model : Optional[str]
    beta : float


@dataclass()
class WandbConfig():
    wandb_project: str
    wandb_log_model: str
    run_name: str


def get_DPO_prompts_func(EOS_TOKEN: str, prompt: str):
    def DPO_prompts_func(examples):
        documents = examples["DOCUMENT"]
        summaries = examples["SUMMARY"]
        rejected_summaries = examples["REJECTED_SUMMARY"]
        texts = []
        chosen_texts = []
        rejected_texts = []
        for document, summary, rejected_summary in zip(documents, summaries, rejected_summaries):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = prompt.format(document)
            texts.append(text)
            chosen_texts.append(summary + EOS_TOKEN)
            rejected_texts.append(rejected_summary + EOS_TOKEN)
        return {
            "prompt": texts,
            "chosen": chosen_texts,   # rated better than k
            "rejected": rejected_texts, # rated worse than j
        }
    return DPO_prompts_func


def get_count_tokens_func(tokenizer, max_token_length=4096):
    def count_tokens(example):
        return len(tokenizer(example['prompt'], return_tensors='pt')["input_ids"][0]) < max_token_length
    return count_tokens


def main():

    parser = HfArgumentParser((ModelParameters, PeftConfig, DatasetConfig, TrainingConfig, DPOConfig, WandbConfig))
    args: tuple[ModelParameters, PeftConfig, DatasetConfig, TrainingConfig, DPOConfig, WandbConfig] = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    model_parameters, peft_config, dataset_config,training_config, dpo_config, wandb_config = args

    os.environ["WANDB_PROJECT"] = wandb_config.wandb_project  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = wandb_config.wandb_log_model  # log all model checkpoints

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_parameters.model_name, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = model_parameters.max_seq_length,
        dtype = model_parameters.dtype,
        load_in_4bit = model_parameters.load_in_4bit,
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_config.r,
        target_modules=peft_config.target_modules,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout, # Supports any, but = 0 is optimized
        bias=peft_config.bias, # Supports any, but = "none" is optimized
        use_gradient_checkpointing=peft_config.use_gradient_checkpointing, # True or "unsloth" for very long context
        random_state=peft_config.random_state,
        use_rslora=peft_config.use_rslora, # We support rank stabilized LoRA
        loftq_config=peft_config.loftq_config, # And LoftQ
        max_seq_length=model_parameters.max_seq_length,
    )

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    prompt = "extract the hospital courses of the following reports and summarize a brief hospital course section: {}\n the following is the golden brief hospital course section:"

    DPO_prompts_func = get_DPO_prompts_func(EOS_TOKEN, prompt)
    train_rejected_df, val_rejected_df = load_dataframes(dataset_config.mimic_version, dataset_config.heading_type, dataset_config.ref_model_name)

    train_df = pd.read_csv(os.path.join("dataset", dataset_config.mimic_version, f"by_{dataset_config.heading_type}", dataset_config.tokenizer_name, "train.csv"))
    train_dataset = pd.merge(train_rejected_df, train_df, on='PAIR_ID', how='left')
    train_dataset = Dataset.from_pandas(train_dataset)

    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        DPO_prompts_func,
        batched=True,
        remove_columns=original_columns
    )

    val_df = pd.read_csv(os.path.join("dataset", dataset_config.mimic_version, f"by_{dataset_config.heading_type}", dataset_config.tokenizer_name, "val.csv"))
    val_dataset = pd.merge(val_rejected_df, val_df, on='PAIR_ID', how='left')
    val_dataset = Dataset.from_pandas(val_dataset)

    original_columns = val_dataset.column_names
    val_dataset = val_dataset.map(
        DPO_prompts_func,
        batched=True,
        remove_columns=original_columns
    )

    train_dataset = train_dataset.filter(get_count_tokens_func(tokenizer))
    val_dataset = val_dataset.filter(get_count_tokens_func(tokenizer))

    training_args = TrainingArguments(
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            warmup_ratio=training_config.warmup_ratio,
            num_train_epochs=training_config.num_train_epochs, # Set this for 1 full training run.
            eval_strategy=training_config.eval_strategy,
            save_strategy=training_config.save_strategy,
            learning_rate=training_config.learning_rate,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            logging_steps=training_config.logging_steps,
            optim=training_config.optim,
            weight_decay=training_config.weight_decay,
            lr_scheduler_type=training_config.lr_scheduler_type,
            seed=training_config.seed,
            max_grad_norm=training_config.max_grad_norm,
            output_dir=training_config.output_dir,
            report_to="wandb",
            run_name=wandb_config.run_name,
        )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=dpo_config.ref_model,
        args=training_args,
        beta=dpo_config.beta,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_length=dpo_config.max_prompt_length+dpo_config.max_response_length,
        max_prompt_length=dpo_config.max_prompt_length,
    )

    dpo_trainer.train()

    model.save_pretrained(os.path.join("unsloth_DPO_models", wandb_config.run_name)) # Local saving
    tokenizer.save_pretrained(os.path.join("unsloth_DPO_models", wandb_config.run_name))

if __name__ == "__main__":
    main()