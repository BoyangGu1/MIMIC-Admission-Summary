import argparse
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import ray

from tqdm import tqdm
from vllm import LLM, SamplingParams

from rewrite_cloze_data_prep import save_mask_dfs


def parse_args() -> tuple[str, str, str, str, str, str, str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite_model", required=True, type=str, help="the rewriting model used for inference")
    parser.add_argument("--gpus", required=True, type=str, help="the gpu indices for this script to run on")
    parser.add_argument("--ref_pairs_csv", required=True, type=str, help="location for document-summary pairs, should be a csv file")
    parser.add_argument("--ref_summary_path", required=True, type=str, help="location for summaries, should be folder containing txt files named after the pair_ids")
    parser.add_argument("--save_path", required=True, type=str, help="location for saving generated responses, should be a directory")
    parser.add_argument("--save_medcat_extraction_path", required=True, type=str, help="location for saving medcat extractions, should be a directory")
    parser.add_argument("--save_mask_dfs_save_path", required=True, type=str, help="location for saving joined document and masked summary, should be a directory")
    parser.add_argument("--save_mask_dfs_save_name", required=False, type=str, default="test", help="name for saving joined document and masked summary")
    parser.add_argument("--save_rewrite_response_name", required=True, type=str, default="test", help="name for saving generated answers for rewriting")
    args = parser.parse_args()
    return args.rewrite_model, args.gpus, args.ref_pairs_csv, args.ref_summary_path, args.save_path, args.save_medcat_extraction_path, args.save_mask_dfs_save_path, args.save_mask_dfs_save_name, args.save_rewrite_response_name


def get_LLMPredictor(model_name, tensor_parallel_size, sampling_params, save_path):

    class LLMPredictor:

        def __init__(self):
            # Create an LLM.
            self.llm = LLM(model=model_name,
                        tensor_parallel_size=tensor_parallel_size)
            self.sampling_params = sampling_params
            self.save_path = save_path

        def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            outputs = self.llm.generate(batch["text"], self.sampling_params)

            prompt: list[str] = []
            generated_text: list[str] = []
            for output in outputs:
                prompt.append(output.prompt)
                generated_text.append(' '.join([o.text for o in output.outputs]))

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

    rewrite_model, gpus, ref_pairs_csv, ref_summary_path, save_path, save_medcat_extraction_path, save_mask_dfs_save_path, save_mask_dfs_save_name, save_rewrite_response_name = parse_args()
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

    prompt_template = "the following are a collection of clinical reports seperated by DOC_SEP and an incomplete brief hospital course summary. the MASK word in the summary represents the missing information. the following is the clinical reports: {}\n the following is the imcomplete brief hospital course section: {}\n the MASK words should be "

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
    # Set tensor parallelism per instance.
    tensor_parallel_size = 1
    # Set number of instances. Each instance will use tensor_parallel_size GPUs.
    num_instances = len(gpus.split(','))

    ds = ray.data.from_pandas(mask_all_df)
    ds = ds.map(get_create_prompt(prompt_template))
    ds = ds.repartition(2)

    resources_kwarg: dict[str, Any] = {}
    resources_kwarg["num_gpus"] = 1

    llm_predictor = get_LLMPredictor(rewrite_model, tensor_parallel_size, sampling_params, save_rewrite_response_name)

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
        answers = response.split(",")
        answers = [answer.strip() for answer in answers]

        original_answers = mask_all_df[mask_all_df["PAIR_ID"] == pair_id].iloc[0]["ANSWERS"].split(" WORD_SEP ")
        mask_no = len(original_answers)

        if len(answers) > mask_no:
            answers = answers[:mask_no]

        if len(answers) < mask_no:
            answers = [answers[i] if i < len(answers) else original_answers[i] for i in range(mask_no)]

        assert len(answers) == mask_no

        masked_summary = mask_all_df[mask_all_df["PAIR_ID"] == pair_id].iloc[0]["MASKED_SUMMARY"]
        # write inference file
        summary = masked_summary.replace("MASK", "{}").format(*answers)
        with open(os.path.join(save_path, f"{pair_id}.txt"), "w") as f:
            f.write(summary)


if __name__ == "__main__":
    main()