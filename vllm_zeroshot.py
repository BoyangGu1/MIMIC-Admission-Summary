import argparse
import os
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd
import ray

from vllm import LLM, SamplingParams


def parse_args() -> tuple[str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, type=str, help="the gpu indices for this script to run on")
    parser.add_argument("--csv_path", required=True, type=str, help="location for document-summary pairs, should be a csv file")
    parser.add_argument("--save_path", required=True, type=str, help="location for saving generated responses, should be a directory")
    args = parser.parse_args()
    return args.gpus, args.csv_path, args.save_path


def get_LLMPredictor(tensor_parallel_size, system_prompt, sampling_params, save_path):

    class LLMPredictor:

        def __init__(self):
            # Create an LLM.
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


def get_create_prompt(user_prompt: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def create_prompt(row: dict[str, Any]) -> dict[str, Any]:
        row["text"] = user_prompt.format(row["DOCUMENT"])
        return row
    return create_prompt


def main():

    gpus, csv_path, save_path = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    assert is_valid_cuda_visible_devices, "Invalid CUDA_VISIBLE_DEVICES value"

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
    # Set tensor parallelism per instance.
    tensor_parallel_size = 1
    # Set number of instances. Each instance will use tensor_parallel_size GPUs.
    num_instances = len(gpus.split(','))

    system_prompt = """
    You are a clinical writing assistant. You are tasked with generating a brief hospital course summary based on provided clinical notes. Clinical notes are seperated by the indicator [DOC_SEP] to indicate each document. The output should be the summarized brief hospital course. Make sure to output with no introduction, no explaintation, only the summarized brief hospital course.
    """

    user_prompt = """
    Now, let's start. Generate the brief hospital course summary.
    Clinical Notes- {} 
    """

    df = pd.read_csv(csv_path)
    ds = ray.data.from_pandas(df)
    ds = ds.map(get_create_prompt(user_prompt))
    ds = ds.repartition(128)

    resources_kwarg: dict[str, Any] = {}
    resources_kwarg["num_gpus"] = 1

    llm_predictor = get_LLMPredictor(tensor_parallel_size, system_prompt, sampling_params, save_path)
    os.makedirs(save_path, exist_ok=True)

    ds = ds.map_batches(
        llm_predictor,
        concurrency=num_instances,
        batch_size=32,
        **resources_kwarg,
    )

    os.makedirs("temp_dir", exist_ok=True)
    ds.write_parquet("temp_dir")
    shutil.rmtree("temp_dir")


if __name__ == "__main__": 
    main()