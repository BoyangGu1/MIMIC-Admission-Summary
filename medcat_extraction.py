import argparse
import os
from typing import Any, TypedDict

import pandas as pd
from tqdm import tqdm

from medcat.cat import CAT
from medcat.utils.preprocess_umls import UMLS


class EntityDict(TypedDict):
    pretty_name: str
    cui: str
    type_ids: list[str]
    types: list[str]
    source_value: str
    detected_name: str
    acc: float
    context_similarity: float
    start: int
    end: int
    icd10: list[str]
    ontologies: list[str]
    snomed: list[str]
    id: int
    meta_anns: dict[str, Any]


class EntityDataDict(TypedDict):
    entities: dict[int, EntityDict]
    tokens: list[Any]


def parse_args() -> tuple[str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=False, type=str, default="inactivate", help="the dataset path for document-summary pairs, should be a csv file")
    parser.add_argument("--summary_path", required=False, type=str, default="inactivate", help="the path that stores summaries, should be a folder")
    parser.add_argument("--save_path", required=True, type=str, help="the path where the final extracted entities will be saved, should be a folder")
    args = parser.parse_args()
    return args.dataset_path, args.summary_path,args.save_path


def medcat_extraction_by_dataset_path(dataset_path: str, save_path: str) -> None:

    cat = CAT.load_model_pack(os.path.join("medcat_model", "umls_self_train_model_pt2ch_3760d588371755d0.zip"))
    mrconso_file = os.path.join("medcat_model", "MRCONSO.RRF")
    mrsty_file = os.path.join("medcat_model", "MRSTY.RRF")
    umls = UMLS(mrconso_file, mrsty_file)
    umls2icd10 = umls.map_umls2icd10()
    cat.cdb.addl_info["cui2icd10"]= umls2icd10
    df = umls.map_umls2icd10()
    umls2icd10 = dict(zip(df["CUI"], df["CODE"]))
    cat.cdb.addl_info["cui2icd10"]= umls2icd10

    df: pd.DataFrame = pd.read_csv(dataset_path, names=["PAIR_ID", "DOCUMENT", "SUMMARY"])
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting entities in train_val set"):
        summary_text: str = row["SUMMARY"]
        entities_data: EntityDataDict = cat.get_entities(summary_text, only_cui=False, addl_info=["cui2icd10", "cui2ontologies", "cui2snomed"])
        entities: dict[int, EntityDict] = entities_data["entities"]
        os.makedirs(save_path, exist_ok=True)
        entities_df: pd.DataFrame = pd.DataFrame(list(entities.values()))
        entities_df.to_csv(os.path.join(save_path, f"{row['PAIR_ID']}.csv"), index=False)


def medcat_extraction_by_summary_path(summary_path: str, save_path: str) -> None:

    cat = CAT.load_model_pack(os.path.join("medcat_model", "umls_self_train_model_pt2ch_3760d588371755d0.zip"))
    mrconso_file = os.path.join("medcat_model", "MRCONSO.RRF")
    mrsty_file = os.path.join("medcat_model", "MRSTY.RRF")
    umls = UMLS(mrconso_file, mrsty_file)
    umls2icd10 = umls.map_umls2icd10()
    cat.cdb.addl_info["cui2icd10"]= umls2icd10
    df = umls.map_umls2icd10()
    umls2icd10 = dict(zip(df["CUI"], df["CODE"]))
    cat.cdb.addl_info["cui2icd10"]= umls2icd10

    for summary_file in tqdm(os.listdir(summary_path)):
        pair_id = summary_file.split(".")[0]
        with open(os.path.join(summary_path, summary_file), "r") as f:
            summary_text = f.read()
        entities_data: EntityDataDict = cat.get_entities(summary_text, only_cui=False, addl_info=["cui2icd10", "cui2ontologies", "cui2snomed"])
        entities: dict[int, EntityDict] = entities_data["entities"]
        os.makedirs(save_path, exist_ok=True)
        entities_df: pd.DataFrame = pd.DataFrame(list(entities.values()))
        entities_df.to_csv(os.path.join(save_path, f"{pair_id}.csv"), index=False)


def main() -> None:
    dataset_path, summary_path, save_path = parse_args()

    # check exactly one of dataset_path and summary_path is provided
    if (dataset_path == "inactivate" and summary_path == "inactivate") or (dataset_path != "inactivate" and summary_path != "inactivate"):
        raise ValueError("Please provide either --dataset_path or --summary_path")

    os.makedirs(save_path, exist_ok=True)
    if dataset_path != "inactivate":
        medcat_extraction_by_dataset_path(dataset_path, save_path)

    else:
        medcat_extraction_by_summary_path(summary_path, save_path)


if __name__ == "__main__":
    main()