import os
import json
from typing import Callable

import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast


def load_datadict(mimic_version: str, 
                  heading_type: str, 
                  tokenizer: PreTrainedTokenizerFast, 
                  tokenizer_name: str,
                  max_token_length: int,
                  prompt_func: Callable[[str, str], str],
                  prompt_name: str,
                  prompt: str,
                  ratio: tuple[float, float] = (0.9, 0.1),
                  safe_token_space: int = 10,
                  ) -> DatasetDict:

    dataset_dir: str = os.path.join("dataset", mimic_version, f"by_{heading_type}")
    constant_dir: str = os.path.join("constants", mimic_version, heading_type, "safe_pair_ids")
    datadict_dir: str = os.path.join(dataset_dir, f"{tokenizer_name}_{prompt_name}_{max_token_length}")
    os.makedirs(datadict_dir, exist_ok=True)
    os.makedirs(constant_dir, exist_ok=True)

    if (os.path.exists(os.path.join(datadict_dir, "train.csv"))
        and os.path.exists(os.path.join(datadict_dir, "val.csv"))
        and os.path.exists(os.path.join(datadict_dir, "test.csv"))
        and os.path.exists(os.path.join(datadict_dir, "prompt.txt"))):

        with open(os.path.join(datadict_dir, "prompt.txt"), 'r') as file:
            prev_prompt: str = file.read()
            if prev_prompt == prompt:
                train_df: pd.DataFrame = pd.read_csv(os.path.join(dataset_dir, f"{tokenizer_name}_{prompt_name}_{max_token_length}", "train.csv"))
                val_df: pd.DataFrame = pd.read_csv(os.path.join(dataset_dir, f"{tokenizer_name}_{prompt_name}_{max_token_length}", "val.csv"))
                test_df: pd.DataFrame = pd.read_csv(os.path.join(dataset_dir, f"{tokenizer_name}_{prompt_name}_{max_token_length}", "test.csv"))
            else:
                raise ValueError(f"Prompt name doesn't match the prompt.")

    else:
        # prepare train_eval and test.csv
        if (not os.path.exists(os.path.join(dataset_dir, "train_val.csv"))
            or not os.path.exists(os.path.join(dataset_dir, "test.csv"))):

            if not os.path.exists(os.path.join(dataset_dir, "all.csv")):
                raise ValueError(f"Dataset for {mimic_version} {heading_type} not found.")
            else:
                all_df: pd.DataFrame = pd.read_csv(os.path.join(dataset_dir, "all.csv"))
                all_df = all_df.sample(frac=1)

                train_val_ratio: float = 0.9
                test_ratio: float = 0.1

                train_val_size: int = int(len(all_df) * train_val_ratio)
                train_val_df: pd.DataFrame = all_df[:train_val_size]
                test_df: pd.DataFrame = all_df[train_val_size:]

                train_val_df.to_csv(os.path.join(dataset_dir, "train_val.csv"), index=False)
                test_df.to_csv(os.path.join(dataset_dir, "test.csv"), index=False)
        
        # split train and val
        train_val_df: pd.DataFrame = pd.read_csv(os.path.join(dataset_dir, "train_val.csv"))
        test_df: pd.DataFrame = pd.read_csv(os.path.join(dataset_dir, "test.csv"))
        assert sum(ratio) == 1
        train_ratio, val_ratio = ratio

        # extract examples that won't exceed max token limit
        safe_pair_ids: list[str] = []
        for index, row in tqdm(train_val_df.iterrows(), total=len(train_val_df)):
            text: str = prompt_func(row["DOCUMENT"], row["SUMMARY"])
            tokens: int = len(tokenizer(text, return_tensors='pt')["input_ids"][0]) + safe_token_space
            if tokens < max_token_length:
                safe_pair_ids.append(row["PAIR_ID"])
        filtered_data_df: pd.DataFrame = train_val_df[train_val_df["PAIR_ID"].isin(safe_pair_ids)]

        # save examples that won't exceed max token limit
        with open(os.path.join(os.path.join(constant_dir, f"{tokenizer_name}_{prompt_name}_{max_token_length}.json")), 'w') as json_file:
            json.dump(safe_pair_ids, json_file, indent=4)
        # save prompt
        with open(os.path.join(datadict_dir, "prompt.txt"), 'w') as file:
            file.write(prompt)

        filtered_data_df = filtered_data_df.sample(frac=1)
        train_size: int = int(len(filtered_data_df) * train_ratio)
        train_df: pd.DataFrame = filtered_data_df[:train_size]
        val_df: pd.DataFrame = filtered_data_df[train_size:]

        # save
        os.makedirs(os.path.join(dataset_dir, f"{tokenizer_name}_{prompt_name}_{max_token_length}"), exist_ok=True)
        train_df.to_csv(os.path.join(dataset_dir, f"{tokenizer_name}_{prompt_name}_{max_token_length}", "train.csv"), index=False)
        val_df.to_csv(os.path.join(dataset_dir, f"{tokenizer_name}_{prompt_name}_{max_token_length}", "val.csv"), index=False)
        test_df.to_csv(os.path.join(dataset_dir, f"{tokenizer_name}_{prompt_name}_{max_token_length}", "test.csv"), index=False)

    dataDict = {
        "train": Dataset.from_pandas(train_df[["DOCUMENT", "SUMMARY"]], preserve_index=False),
        "val": Dataset.from_pandas(val_df[["DOCUMENT", "SUMMARY"]], preserve_index=False),
        "test": Dataset.from_pandas(test_df[["DOCUMENT", "SUMMARY"]], preserve_index=False),
    }

    return DatasetDict(dataDict)
