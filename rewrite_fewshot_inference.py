import argparse
import os
import shutil
from typing import Any

import json
import numpy as np
import pandas as pd
import ray
import re

from tqdm import tqdm
from vllm import LLM, SamplingParams

from rewrite_cloze_data_prep import save_mask_dfs


def parse_args() -> tuple[int, str, str, str, str, str, str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", required=False, type=int, default=0, help="how many shots to use for rewriting, default zero-shot")
    parser.add_argument("--gpus", required=True, type=str, help="the gpu indices for this script to run on")
    parser.add_argument("--ref_pairs_csv", required=True, type=str, help="location for document-summary pairs, should be a csv file")
    parser.add_argument("--ref_summary_path", required=True, type=str, help="location for summaries, should be folder containing txt files named after the pair_ids")
    parser.add_argument("--save_path", required=True, type=str, help="location for saving generated responses, should be a directory")
    parser.add_argument("--save_medcat_extraction_path", required=True, type=str, help="location for saving medcat extractions, should be a directory")
    parser.add_argument("--save_mask_dfs_save_path", required=True, type=str, help="location for saving joined document and masked summary, should be a directory")
    parser.add_argument("--save_mask_dfs_save_name", required=False, type=str, default="test", help="name for saving joined document and masked summary")
    parser.add_argument("--save_rewrite_response_name", required=True, type=str, default="test", help="name for saving generated answers for rewriting")
    args = parser.parse_args()
    return args.shots, args.gpus, args.ref_pairs_csv, args.ref_summary_path, args.save_path, args.save_medcat_extraction_path, args.save_mask_dfs_save_path, args.save_mask_dfs_save_name, args.save_rewrite_response_name


def get_LLMPredictor(tensor_parallel_size, system_prompt, sampling_params, save_path):

    class LLMPredictor:

        def __init__(self):
            self.llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        tensor_parallel_size=tensor_parallel_size)
            self.tokenizer = self.llm.get_tokenizer()
            self.system_prompt = system_prompt
            self.sampling_params = sampling_params
            self.save_path = save_path

        def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            conversations = []
            for text in batch["text"]:
                conversation = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text},
                    ],
                    tokenize=False,
                )
                conversations.append(conversation)
            # change that so conversation takes the batch[text]
            outputs = self.llm.generate(conversations, self.sampling_params)

            prompt: list[str] = []
            generated_text: list[str] = []
            for output in outputs:
                prompt.append(output.prompt)
                response = ' '.join([o.text for o in output.outputs])
                generated_text.append('\n'.join(response.split('\n')[2:]))

            # Save the generated text to a file named after the pair_id.
            pair_ids = batch["PAIR_ID"]
            for pair_id, one_generated_text in zip(pair_ids, generated_text):
                filename = os.path.join(self.save_path, f"{pair_id}.txt")
                with open(filename, "w") as f:
                    f.write(one_generated_text)

            return {
                "prompt": prompt,
                "generated_text": generated_text,
            }

    return LLMPredictor


def get_create_prompt(prompt_template):
    def create_prompt(row: dict[str, Any]) -> dict[str, Any]:
        answers = row["ANSWERS"]
        mask_no = answers.count(" WORD_SEP ") + 1
        assert mask_no == row["MASKED_SUMMARY"].count("MASK"), f"The number of MASK in the masked summary should be {mask_no}"
        row["text"] = prompt_template.format(mask_no, row["DOCUMENT"], row["MASKED_SUMMARY"])
        return row
    return create_prompt


def is_valid_cuda_visible_devices(cuda_str: str) -> bool:
    if cuda_str == "":
        return True  # An empty string is a valid value

    devices = cuda_str.split(',')

    try:
        # Convert to integers and check for duplicates
        device_numbers = list(map(int, devices))
    except ValueError:
        return False  # Non-integer value present

    # Check for non-negative integers and duplicates
    if any(d < 0 for d in device_numbers) or len(device_numbers) != len(set(device_numbers)):
        return False

    return True


def main():

    shots, gpus, ref_pairs_csv, ref_summary_path, save_path, save_medcat_extraction_path, save_mask_dfs_save_path, save_mask_dfs_save_name, save_rewrite_response_name = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    assert is_valid_cuda_visible_devices, "Invalid CUDA_VISIBLE_DEVICES value"

    # TODO uncomment it
    if len(os.listdir(save_medcat_extraction_path)) != len(os.listdir(ref_summary_path)):
        raise ValueError(f"The number of files in the reference summary path and the save_medcat_extraction_path should be the same, but {len(os.listdir(save_medcat_extraction_path))} is not {len(os.listdir(ref_summary_path))}. Run the medcat_extraction.py first.")

    if not (os.path.exists(os.path.join(save_mask_dfs_save_path, f"medcat_extraction_{save_mask_dfs_save_name}_mask_one.csv"))
            and os.path.exists(os.path.join(save_mask_dfs_save_path, f"medcat_extraction_{save_mask_dfs_save_name}_mask_all.csv"))):
        save_mask_dfs(ref_pairs_csv, save_medcat_extraction_path, save_mask_dfs_save_path, save_mask_dfs_save_name, summary_path=ref_summary_path)

    # mask_one_df = pd.read_csv(os.path.join(save_mask_dfs_save_path, f"medcat_extraction_{save_mask_dfs_save_name}_mask_one.csv"))
    mask_all_df = pd.read_csv(os.path.join(save_mask_dfs_save_path, f"medcat_extraction_{save_mask_dfs_save_name}_mask_all.csv"))

    # Create a directory to save the generated text.
    os.makedirs(save_rewrite_response_name, exist_ok=True)

    system_prompt = """
    You are a clinical writing assistant. You are tasked with filling a incomplete brief hospital course summary based on provided clinical notes. Clinical notes are seperated by the indicator [DOC_SEP] to indicate each document. The incomplete brief hospital course summary has mutiple words MASK that represents the missing part. The goal is to fill in all missing parts of the summary based on the provided clinical notes. The output should be the words to replace every MASK word. If you cannot find the correct words, use NOT_FOUND to indicate it.

    Steps for generating the hallucinated summary: 
    Step 1: Find every MASK word in the incomplete brief hospital course summary and make sure that the number of MASK is correct. If you find the number of MASK is different from the number I provided, then you must count it wrong.
    Step 2: Use the provided clinical notes to find the words to replace every MASK word. 

    The output should strictly be in the following JSON object format and remember to include the {} at the begining and end of the JSON object:
    {1: "your answer for the first missing part", 2: "your answer for the second missing part", ...}
    """

    user_prompt = """
    Now, let's start. Find the correct words to replace MASK. There are exactly {} number of MASK words in the summary.
    Clinical Notes- {} 
    Incomplete Brief Hospital Course Summary- {}
    """

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
    # Set tensor parallelism per instance.
    tensor_parallel_size = 1
    # Set number of instances. Each instance will use tensor_parallel_size GPUs.
    num_instances = len(gpus.split(','))

    ds = ray.data.from_pandas(mask_all_df)
    ds = ds.map(get_create_prompt(user_prompt))
    ds = ds.repartition(2)

    resources_kwarg: dict[str, Any] = {}
    resources_kwarg["num_gpus"] = 1

    llm_predictor = get_LLMPredictor(tensor_parallel_size, system_prompt, sampling_params, save_rewrite_response_name)

    # Apply batch inference for all input data.
    ds = ds.map_batches(
        llm_predictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=32,
        **resources_kwarg,
    )

    os.makedirs("temp_dir", exist_ok=True)
    ds.write_parquet("temp_dir")
    shutil.rmtree("temp_dir")

    # process answers to insert them into the masked summary
    os.makedirs(save_path, exist_ok=True)
    for file_name in tqdm(os.listdir(save_rewrite_response_name)):
        pair_id = file_name.split(".")[0]
        with open(os.path.join(save_rewrite_response_name, file_name), "r") as f:
            response = f.read()
        left_bracket_index = response.find("{")
        right_bracket_index = response.find("}")
        if left_bracket_index == -1 or right_bracket_index == -1:
            continue
        response_json = response[left_bracket_index:right_bracket_index+1]

        try:
            response_json = re.sub(r'(\d+):', r'"\1":', response_json)
            answers = json.loads(response_json)
            answers = {int(k): v for k, v in answers.items()}
            answers_keys = list(answers.keys())
            original_answers = mask_all_df[mask_all_df["PAIR_ID"] == pair_id].iloc[0]["ANSWERS"].split(" WORD_SEP ")
            mask_no = len(original_answers)
            assert set(answers_keys) == set(range(1, mask_no+1))
        except:
            continue
        answers_list = [answers[i] if answers[i] != "NOT_FOUND" else original_answers[i-1] for i in range(1, mask_no+1)]
        masked_summary = mask_all_df[mask_all_df["PAIR_ID"] == pair_id].iloc[0]["MASKED_SUMMARY"]
        # write inference file
        summary = masked_summary.replace("MASK", "{}").format(*answers_list)
        with open(os.path.join(save_path, f"{pair_id}.txt"), "w") as f:
            f.write(summary)


if __name__ == "__main__":
    main()