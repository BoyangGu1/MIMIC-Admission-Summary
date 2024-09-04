import argparse
import os
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd
import ray

from vllm import LLM, SamplingParams


class LLMPredictor:

    def __init__(self, tensor_parallel_size, system_prompt, sampling_params, save_path):
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

def parse_args() -> tuple[str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, type=str, help="the gpu indices for this script to run on")
    parser.add_argument("--csv_path", required=True, type=str, help="location for document-summary pairs, should be a csv file")
    parser.add_argument("--save_path", required=True, type=str, help="location for saving generated responses, should be a directory")
    args = parser.parse_args()
    return args.gpus, args.csv_path, args.save_path


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
        row["text"] = user_prompt.format(row["DOCUMENT"], row["SUMMARY"])
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
    You are a clinical writing assistant who is in edit mode. You are tasked with generating hallucinated brief hospital course summary based on provided clinical notes and a reference summary for the clinical notes. Clinical notes are seperated by the indicator [DOC_SEP] to indicate each document. The goal is to edit the reference summary to generate a hallucinated summary that sounds plausible but includes edits introduced through an edit operation which can be one of the following: 
    Add Operation: Intentionally add medico-legally essential words from the clinical notes not required for accurate diagnosis and treatment documentation. 
    Omit Operation: Intentionally omit medico-legally essential words in the reference summary required for accurate diagnosis and treatment documentation. 

    For these operations focus on words that, if missing or incorrect in the hallucinated summary, could lead to wrong diagnoses and treatments in the future. Maintain coherence while excluding essential terms. The hallucinated summary should be concise and contain no more than FIVE EXTRA WORDS compared to the reference summary and should have an equal number of Add/Omit operations. 

    Steps for generating the hallucinated summary: 
    Step 1: List the proposed edit operations to introduce hallucination on the reference summary. 
    Step 2: Use the proposed edit operations to edit the reference summary.

    The output should strictly be in the following format: 
    Numbererd List hallucination edits made: {Edit 1}, {Edit 2}, {Edit 3} ... 
    Hallucinated Summary:
    """

    user_prompt = """
    Now, let's start. Generate the hallucinated summary: 
    Clinical Notes- {} 
    Reference Summary- {}
    """

    df = pd.read_csv(csv_path)
    ds = ray.data.from_pandas(df)
    ds = ds.map(get_create_prompt(user_prompt))
    ds = ds.repartition(128)

    resources_kwarg: dict[str, Any] = {}
    resources_kwarg["num_gpus"] = 1

    os.makedirs(save_path, exist_ok=True)

    ds = ds.map_batches(
        LLMPredictor(tensor_parallel_size, system_prompt, sampling_params, save_path),
        concurrency=num_instances,
        batch_size=32,
        **resources_kwarg,
    )

    os.makedirs("temp_dir", exist_ok=True)
    ds.write_parquet("temp_dir")
    shutil.rmtree("temp_dir")

if __name__ == "__main__":
    main()