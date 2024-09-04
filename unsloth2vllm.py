
import argparse
import os

from typing import Optional

from unsloth import FastLanguageModel


def parse_args() -> tuple[str, str, str, Optional[str], bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str, help="name of the model to use")
    parser.add_argument("--vllm_save_path", required=True, type=str, help="location for saving vllm-form model")
    parser.add_argument("--max_seq_length", required=False, type=str, default=32768, help="maximum sequence length to use for the model")
    parser.add_argument("--dtype", required=False, type=str, default=None, help="data type to use for the model")
    parser.add_argument("--load_in_4bit", required=False, type=bool, default=True, help="whether to load the model in 4-bit precision")
    args = parser.parse_args()
    return args.model_name, args.vllm_save_path, args.max_seq_length, args.dtype, args.load_in_4bit


def main():

    model_name, vllm_save_path, max_seq_length, dtype, load_in_4bit = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    os.makedirs(os.path.dirname(vllm_save_path), exist_ok=True)
    model.save_pretrained_merged(vllm_save_path, tokenizer, save_method="merged_16bit",)


if __name__ == "__main__":
    main()