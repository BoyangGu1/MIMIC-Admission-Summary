import argparse
import ast
import os

from typing import Optional

import pandas as pd
from tqdm import tqdm


def parse_args() -> tuple[str, str, str, str, Optional[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=str, help="the dataset path for document-summary pairs, should be a csv file")
    parser.add_argument("--medcat_extraction_dir", required=True, type=str, help="the directory where the medcat extraction files are stored, should be a directory that contains csv files named by pair_ids")
    parser.add_argument("--save_path", required=True, type=str, help="the path where the final dataset will be saved, should be a folder")
    args = parser.parse_args()
    parser.add_argument("--save_name", required=True, type=str, help="the name of the final saved file")
    parser.add_argument("--summary_path", required=False, type=str, default=None, help="the path for summaries, it will overwrite the summary in the dataset file, should be a folder with txts named by pair_ids")
    args = parser.parse_args()
    return args.dataset_path, args.medcat_extraction_dir, args.save_path, args.save_name, args.summary_path


def clean_and_parse_type_ids(type_id_str):
    # Remove extra quotes if present
    if type_id_str.startswith('"') and type_id_str.endswith('"'):
        type_id_str = type_id_str[1:-1]

    # Convert the string representation of list to an actual list
    return ast.literal_eval(type_id_str)


def save_mask_dfs(dataset_path, medcat_extraction_dir, save_path, save_name, summary_path):
    df = pd.read_csv(dataset_path)

    triggered_medcat_types = ["T116", "T195", "T123", "T122", "T200", "T196", "T126", "T131", "T125", "T129", "T130", "T197", "T114", "T109", "T121", "T127", "T020", "T190", "T049", "T019", "T047", "T050", "T033", "T037", "T048", "T191", "T046", "T184", "T060", "T059", "T063", "T061", "T074", "T031"]

    mask_one_df = pd.DataFrame(columns=["MEDCAT_PAIR_ID", "DOCUMENT", "MASKED_SUMMARY", "ANSWER", "DETECTED_NAME", "TYPE_ID"])
    mask_all_df = pd.DataFrame(columns=["PAIR_ID", "DOCUMENT", "MASKED_SUMMARY", "ANSWERS"])

    os.makedirs(save_path, exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {save_name} set"):
        pair_id = row["PAIR_ID"]
        document = row["DOCUMENT"]
        if summary_path:
            with open(os.path.join(summary_path, f"{pair_id}.txt"), "r") as f:
                summary = f.read()
        else:
            summary = row["SUMMARY"]
        mask_all_summary = summary
        masked_answers = []
        medcat_extraction_file = os.path.join(medcat_extraction_dir, f"{pair_id}.csv")
        try:
            medcat_extraction_df = pd.read_csv(medcat_extraction_file)
        except pd.errors.EmptyDataError:
            continue
        medcat_extraction_df['type_ids'] = medcat_extraction_df['type_ids'].apply(clean_and_parse_type_ids)
        valid_medcat_extraction = medcat_extraction_df[
            medcat_extraction_df['type_ids'].apply(lambda x: any(item in triggered_medcat_types for item in x))
        ]
        valid_medcat_extraction_sorted = valid_medcat_extraction.sort_values(by="start", ascending=False)
        for j, row in valid_medcat_extraction_sorted.iterrows():
            detected_name = summary[row["start"]:row["end"]]
            assert detected_name == row["source_value"]
            # change the detected_name in the summary to MASK
            mask_one_summary = summary[:row["start"]] + "MASK" + summary[row["end"]:]
            new_row = pd.DataFrame([
                {"MEDCAT_PAIR_ID": f"{pair_id}_{row['start']}_{row['end']}", "DOCUMENT": document, "MASKED_SUMMARY": mask_one_summary, "ANSWER": detected_name, "DETECTED_NAME": row["detected_name"], "TYPE_ID": row["type_ids"]},
            ])
            mask_one_df = pd.concat([mask_one_df, new_row], ignore_index=True)

            # change all detected_names in the summary to MASK
            mask_all_summary = mask_all_summary[:row["start"]] + "MASK" + mask_all_summary[row["end"]:]
            masked_answers.append(detected_name)
        masked_answers_str = " WORD_SEP ".join(masked_answers[::-1])

        if mask_all_summary != summary:
            new_row = pd.DataFrame([
                {"PAIR_ID": pair_id, "DOCUMENT": document, "MASKED_SUMMARY": mask_all_summary, "ANSWERS": masked_answers_str},
            ])
            mask_all_df = pd.concat([mask_all_df, new_row], ignore_index=True)

        mask_one_df.to_csv(os.path.join(save_path, f"medcat_extraction_{save_name}_mask_one.csv"), index=False)
        mask_all_df.to_csv(os.path.join(save_path, f"medcat_extraction_{save_name}_mask_all.csv"), index=False)


def main():
    dataset_path, medcat_extraction_dir, save_path, save_name, summary_path = parse_args()
    save_mask_dfs(dataset_path, medcat_extraction_dir, save_path, save_name, summary_path)


if __name__ == "__main__":
    main()