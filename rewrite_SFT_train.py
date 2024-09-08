import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
import pandas as pd
from datasets import Dataset
from transformers import (
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    TrainingArguments
)

from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


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
class WandbConfig():
    wandb_project: str
    wandb_log_model: str
    run_name: str


def add_special_token(token: str, 
                      description: str, 
                      tokenizer: PreTrainedTokenizerFast, 
                      model: PreTrainedModel) -> None:
    tokenizer.add_special_tokens({'additional_special_tokens':[token]})
    model.resize_token_embeddings(len(tokenizer))

    # Get the index of the new token
    new_token_id = tokenizer.convert_tokens_to_ids(token)

    # Tokenize the text
    encoded_dict = tokenizer(description, return_tensors='pt')
    # Get the special token IDs
    special_tokens = tokenizer.all_special_ids
    # Filter out the special token IDs
    filtered_input_ids = [token_id for token_id in encoded_dict['input_ids'].squeeze().tolist() if token_id not in special_tokens]

    with torch.no_grad():
        # Initialize the new embedding
        embeddings_layer = model.get_input_embeddings()
        new_embedding = 0
        for token_id in filtered_input_ids:
            new_embedding += embeddings_layer.weight[token_id]
        new_embedding /= len(filtered_input_ids)

        # Update the embedding matrix of the model
        embeddings_layer.weight[new_token_id] = new_embedding


def get_formatting_prompts_func(EOS_TOKEN: str, prompt: str):
    def formatting_prompts_func(examples):
        documents = examples["DOCUMENT"]
        summaries = examples["MASKED_SUMMARY"]
        answers_list = examples["ANSWERS"]
        texts = []
        for document, summary, answers in zip(documents, summaries, answers_list):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            answers = answers.replace(" WORD_SEP ", ", ")
            text = prompt.format(document, summary, answers) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    return formatting_prompts_func


def main():
    parser = HfArgumentParser((ModelParameters, PeftConfig, TrainingConfig, WandbConfig))
    args: tuple[ModelParameters, PeftConfig, TrainingConfig, WandbConfig] = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    model_parameters, peft_config, training_config, wandb_config = args

    os.environ["WANDB_PROJECT"] = wandb_config.wandb_project  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = wandb_config.wandb_log_model  # log all model checkpoints

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_parameters.model_name,
        max_seq_length=model_parameters.max_seq_length,
        dtype=model_parameters.dtype,
        load_in_4bit=model_parameters.load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    # add_special_token("DOC_SEP", "seperation of two documents", tokenizer, model)

    model: PreTrainedModel = FastLanguageModel.get_peft_model(
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
    )

    response_template: str = "the MASK words should be"
    response_template_ids: list[int] = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    EOS_TOKEN: str = tokenizer.eos_token # Must add EOS_TOKEN
    prompt: str = "the following are a collection of clinical reports seperated by DOC_SEP and an incomplete brief hospital course summary. the MASK word in the summary represents the missing information. the following is the clinical reports: {}\n the following is the imcomplete brief hospital course section: {}\n the MASK words should be {}"

    train_df: pd.DataFrame = pd.read_csv(os.path.join("dataset", "mimic-iii", "by_hpc", "Meta-Llama-3.1-8B_hpc1_32768", "medcat_extraction_train_mask_all.csv"))
    eval_df: pd.DataFrame = pd.read_csv(os.path.join("dataset", "mimic-iii", "by_hpc", "Meta-Llama-3.1-8B_hpc1_32768", "medcat_extraction_val_mask_all.csv"))
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    os.makedirs(training_config.output_dir, exist_ok=True)
    training_args = TrainingArguments(
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            warmup_ratio=training_config.warmup_ratio,
            num_train_epochs=training_config.num_train_epochs,
            eval_strategy=training_config.eval_strategy,
            save_strategy=training_config.save_strategy,
            group_by_length=training_config.group_by_length,
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

    formatting_prompts_func = get_formatting_prompts_func(EOS_TOKEN, prompt)
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=model_parameters.max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    trainer_stats = trainer.train()

    # save model
    model.save_pretrained(os.path.join("unsloth_rewriting_SFT_models", wandb_config.run_name))
    tokenizer.save_pretrained(os.path.join("unsloth_rewriting_SFT_models", wandb_config.run_name))


if __name__ == "__main__":
    main()