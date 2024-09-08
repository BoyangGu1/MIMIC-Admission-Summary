import os

import pandas as pd
from tqdm import tqdm

def get_rejected_summary(text: str):
    lines: list[str] = text.split("\n")
    summary_indicator = False
    rejected_summary = ""
    for line in lines:
        if summary_indicator and ("hallucinate" in line.lower() or "hallucination" in line.lower()):
            break
        if summary_indicator:
            rejected_summary += line + "\n"
        if "hallucinated summary" in line.lower() and len(line) <= 30:
            summary_indicator = True
    return rejected_summary.strip('\n')

def load_dataframes(mimic_version: str, heading_type: str, ref_model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    inference_dir = os.path.join("DPO_rejected_summary", mimic_version, f"by_{heading_type}", f"{ref_model_name}")
    if os.path.exists(os.path.join(inference_dir, "train_rejected_summary.csv")) and os.path.exists(os.path.join(inference_dir, "val_rejected_summary.csv")):
        train_rejected_df = pd.read_csv(os.path.join(inference_dir, "train_rejected_summary.csv"))
        val_rejected_df = pd.read_csv(os.path.join(inference_dir, "val_rejected_summary.csv"))

    else:

        train_rejected_df = pd.DataFrame(columns=["PAIR_ID", "rejected"])
        train_inference_dir = os.path.join(inference_dir, "train")
        for file in tqdm(os.listdir(train_inference_dir), desc="Processing train files"):
            pair_id = file.split(".")[0]
            with open(os.path.join(train_inference_dir, file), "r") as f:
                text = f.read()

            rejected_summary = get_rejected_summary(text)
            if rejected_summary != "":
                new_row = pd.DataFrame({'PAIR_ID': [pair_id], 'REJECTED_SUMMARY': [rejected_summary]})
                train_rejected_df = pd.concat([train_rejected_df, new_row], ignore_index=True)

        train_rejected_df.to_csv(os.path.join(inference_dir, "train_rejected_summary.csv"), index=False)

        #################################################################

        val_rejected_df = pd.DataFrame(columns=["PAIR_ID", "rejected"])
        val_inference_dir = os.path.join(inference_dir, "val")
        for file in tqdm(os.listdir(val_inference_dir), desc="Processing val files"):
            pair_id = file.split(".")[0]
            with open(os.path.join(val_inference_dir, file), "r") as f:
                text = f.read()

            rejected_summary = get_rejected_summary(text)
            if rejected_summary != "":
                new_row = pd.DataFrame({'PAIR_ID': [pair_id], 'REJECTED_SUMMARY': [rejected_summary]})
                val_rejected_df = pd.concat([val_rejected_df, new_row], ignore_index=True)

        val_rejected_df.to_csv(os.path.join(inference_dir, "val_rejected_summary.csv"), index=False)

    return train_rejected_df, val_rejected_df