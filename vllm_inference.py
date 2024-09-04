import argparse
import os
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd
import ray

from vllm import LLM, SamplingParams


class LLMPredictor:

    def __init__(self, model_name, tensor_parallel_size, sampling_params, save_path):
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


def parse_args() -> tuple[str, str, str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str, help="the name of the model to use")
    parser.add_argument("--gpus", required=True, type=str, help="the gpu indices for this script to run on")
    parser.add_argument("--csv_path", required=True, type=str, help="location for document-summary pairs, should be a csv file")
    parser.add_argument("--prompt_path", required=True, type=str, help="location for prompt template, should be a txt file")
    parser.add_argument("--save_path", required=True, type=str, help="location for saving generated responses, should be a directory")
    args = parser.parse_args()
    return args.model_name, args.gpus, args.csv_path, args.prompt_path, args.save_path


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


def get_create_prompt(prompt_template: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def create_prompt(row: dict[str, Any]) -> dict[str, Any]:
        row["text"] = prompt_template.format(row["DOCUMENT"])
        return row
    return create_prompt


def main():

    model_name, gpus, csv_path, prompt_path, save_path = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    assert is_valid_cuda_visible_devices, "Invalid CUDA_VISIBLE_DEVICES value"

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
    # Set tensor parallelism per instance.
    tensor_parallel_size = 1
    # Set number of instances. Each instance will use tensor_parallel_size GPUs.
    num_instances = len(gpus.split(','))

    with open(prompt_path, 'r') as file:
        prompt_template = file.read()
    prompt_template = prompt_template[:-2]

    df = pd.read_csv(csv_path)
    ds = ray.data.from_pandas(df)
    ds = ds.map(get_create_prompt(prompt_template))
    ds = ds.repartition(128)

    resources_kwarg: dict[str, Any] = {}
    resources_kwarg["num_gpus"] = 1

    os.makedirs(save_path, exist_ok=True)
    ds = ds.map_batches(
        LLMPredictor(model_name, tensor_parallel_size, sampling_params, save_path),
        concurrency=num_instances,
        batch_size=32,
        **resources_kwarg,
    )

    os.makedirs("temp_dir", exist_ok=True)
    ds.write_parquet("temp_dir")
    shutil.rmtree("temp_dir")


if __name__ == "__main__":
    main()