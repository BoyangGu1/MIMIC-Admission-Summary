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


def parse_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=False, type=str, choices=["mimic-iii", "mimic-iv", "mimic-cxr"], default="mimic-iii", help="choose between mimic-iii, iv, and cxr")
    parser.add_argument("--heading_type", required=False, type=str, choices=["hpc", "pmh", "psh", "pe"], default="hpc", help="section you want to extract, choosing between hpc (hospital course), pmh (past medical history), psh (past surgical history), or pe (physical examination)")
    args = parser.parse_args()
    return args.version, args.heading_type


def main() -> None:
    mimic_version: str
    heading_type: str
    mimic_version, heading_type = parse_args()

    cat = CAT.load_model_pack("medcat_model/umls_self_train_model_pt2ch_3760d588371755d0.zip")
    mrconso_file = "medcat_model/MRCONSO.RRF"
    mrsty_file = "medcat_model/MRSTY.RRF"
    umls = UMLS(mrconso_file, mrsty_file)
    umls2icd10 = umls.map_umls2icd10()
    cat.cdb.addl_info["cui2icd10"]= umls2icd10
    df = umls.map_umls2icd10()
    umls2icd10 = dict(zip(df["CUI"], df["CODE"]))
    cat.cdb.addl_info["cui2icd10"]= umls2icd10

    train_val_file: str = os.path.join("dataset", mimic_version, f"by_{heading_type}", "train_val.csv")
    train_val_df: pd.DataFrame = pd.read_csv(train_val_file, names=["PAIR_ID", "DOCUMENT", "SUMMARY"])

    medcat_dir: str = os.path.join("dataset", mimic_version, f"by_{heading_type}", "medcat_extraction_train_val")
    for i, row in tqdm(train_val_df.iterrows(), total=len(train_val_df), desc="Extracting entities in train_val set"):
        summary_text: str = row["SUMMARY"]
        entities_data: EntityDataDict = cat.get_entities(summary_text, only_cui=False, addl_info=["cui2icd10", "cui2ontologies", "cui2snomed"])
        entities: dict[int, EntityDict] = entities_data["entities"]
        os.makedirs(medcat_dir, exist_ok=True)
        entities_df: pd.DataFrame = pd.DataFrame(list(entities.values()))
        entities_df.to_csv(os.path.join(medcat_dir, f"{row['PAIR_ID']}.csv"), index=False)

    test_file: str = os.path.join("MIMIC_project_data/dataset", mimic_version, f"by_{heading_type}", "test.csv")
    test_df: pd.DataFrame = pd.read_csv(test_file, names=["PAIR_ID", "DOCUMENT", "SUMMARY"])

    medcat_dir: str = os.path.join("MIMIC_project_data/dataset", mimic_version, f"by_{heading_type}", "medcat_extraction_test")
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting entities in test set"):
        summary_text: str = row["SUMMARY"]
        entities_data: EntityDataDict = cat.get_entities(summary_text, only_cui=False, addl_info=["cui2icd10", "cui2ontologies", "cui2snomed"])
        entities: dict[int, EntityDict] = entities_data["entities"]
        os.makedirs(medcat_dir, exist_ok=True)
        entities_df: pd.DataFrame = pd.DataFrame(list(entities.values()))
        entities_df.to_csv(os.path.join(medcat_dir, f"{row['PAIR_ID']}.csv"), index=False)


if __name__ == "__main__":
    main()