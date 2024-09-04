import os
import argparse

import pandas as pd
from tqdm import tqdm
from typing import Optional

required_files: list[str] = [
    "ADMISSIONS.csv",
    "CALLOUT.csv",
    "CAREGIVERS.csv",
    "CHARTEVENTS.csv",
    "CPTEVENTS.csv",
    "D_CPT.csv",
    "D_ICD_DIAGNOSES.csv",
    "D_ICD_PROCEDURES.csv",
    "D_ITEMS.csv",
    "D_LABITEMS.csv",
    "DIAGNOSES_ICD.csv",
    "DRGCODES.csv",
    "ICUSTAYS.csv",
    "INPUTEVENTS_CV.csv",
    "INPUTEVENTS_MV.csv",
    "LABEVENTS.csv",
    "MICROBIOLOGYEVENTS.csv",
    "NOTEEVENTS.csv",
    "OUTPUTEVENTS.csv",
    "PATIENTS.csv",
    "PRESCRIPTIONS.csv",
    "PROCEDUREEVENTS_MV.csv",
    "PROCEDURES_ICD.csv",
    "SERVICES.csv",
    "TRANSFERS.csv",
]

processing_files: dict[str, str] = {
    "ADMISSIONS": "ADMISSIONS.csv",
    "NOTEEVENTS": "NOTEEVENTS.csv",
}

def parse_args() -> tuple[str, str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=False, type=str, default=".", help="location for mimic databse, such path should contain a physionet.org folder")
    parser.add_argument("--mimic_version", required=False, type=str, choices=["mimic-iii", "mimic-iv", "mimic-cxr"], default="mimic-iii", help="choose between mimic-iii, iv, and cxr")
    parser.add_argument("--chunksize", required=False, type=int, default=1000000, help="chunksize when reading csv files")
    args = parser.parse_args()
    return args.database_path, args.mimic_version, args.chunksize


def check_mimic_structure(base_directory: str, mimic_version: str) -> bool:
    """Checking whether the MIMIC databse is stored locally and the file strcuture is correct.

    Args:
        base_directory (str): Base directory where the MIMIC database is stored locally.
        mimic_version (str): The version of MIMIC. Should be either mimic-iii, mimic-iv, or mimic-cxr.

    Returns:
        bool: Whether the base directory pass the test.
    """
    if mimic_version == "mimic-iii":
        target_directory = os.path.join(base_directory, "physionet.org", "files", "mimiciii", "1.4")
        # Check if the target directory exists
        if not os.path.isdir(target_directory):
            print(f"Directory '{target_directory}' does not exist.")
            return False
        # Check if all required files are present
        missing_files = []
        for file in required_files:
            if not (os.path.exists(os.path.join(target_directory, file + ".gz")) or 
                    os.path.exists(os.path.join(target_directory, file))):
                missing_files.append(file + ".csv")
        if not missing_files:
            print("All required files are present.")
            return True
        else:
            print("Missing files:\n", "\n".join(missing_files))
            return False
        
    elif mimic_version == "mimic-iv":
        raise NotImplementedError
    
    elif mimic_version == "mimic-cxr":
        raise NotImplementedError
    
    else:
        raise ValueError(f"{mimic_version} is not a valid mimic version, choose between mimic-iii, iv, or cxr.")
    

def create_patients_files(patients_file_path: str, file_store_path: str, mimic_version: str, chunksize: int) -> None:
    """Create a folder for each patient for further operations.

    Args:
        patients_file_path (str): The csv file that contains the basic information for all patients.
        file_store_path (str): The location for creating the folders.
        mimic_version (str): The version of MIMIC. Should be either mimic-iii, mimic-iv, or mimic-cxr.
        chunksize (int): Chunksize when reading csv files.
    """
    compression: Optional[str] = "gzip" if patients_file_path.endswith(".gz") else None
    patients_no: int = 0
    for chunk in pd.read_csv(patients_file_path, chunksize=chunksize, compression=compression):
        assert isinstance(chunk, pd.DataFrame)
        patients_no += len(chunk)

    with tqdm(total=patients_no, desc=f"Processing {patients_file_path}, creating folders for each patient") as pbar:
        for chunk in pd.read_csv(patients_file_path, chunksize=chunksize, compression=compression):
            assert isinstance(chunk, pd.DataFrame)
            for index, row in chunk.iterrows():
                subject_id: str = row['SUBJECT_ID']
                os.makedirs(os.path.join(file_store_path, mimic_version, subject_id))
                pd.DataFrame([row]).to_csv(os.path.join(file_store_path, mimic_version, subject_id, "PATIENTS.csv"), index=False)
                pbar.update(1)


def load_data(file_path: str, data_name: str, file_store_path: str, mimic_version: str, chunksize: int) -> None:
    """Extract related information for each patient, and store them under each patient's folder.

    Args:
        file_path (str): The path for storing the data.
        data_name (str): The name for the type of data that needs extraction.
        file_store_path (str): Storage place, should contain all patients' folders.
        mimic_version (str): The version of MIMIC. Should be either mimic-iii, mimic-iv, or mimic-cxr.
        chunksize (int): Chunksize when reading csv files.
    """
    compression: Optional[str] = "gzip" if file_path.endswith(".gz") else None
    row_no: int = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize, compression=compression):
        assert isinstance(chunk, pd.DataFrame)
        row_no += len(chunk)

    with tqdm(total=row_no, desc=f"Processing {file_path}") as pbar:
        for chunk in pd.read_csv(file_path, chunksize=chunksize, compression=compression):
            assert isinstance(chunk, pd.DataFrame)
            for index, row in chunk.iterrows():
                subject_id: str = row['SUBJECT_ID']
                patient_data_path: str = os.path.join(file_store_path, mimic_version, subject_id, f"{data_name}.csv")
                if os.path.exists(patient_data_path):
                    data_df: pd.DataFrame = pd.read_csv(patient_data_path)
                    data_df = pd.concat([data_df, pd.DataFrame([row])], ignore_index=True)
                else:
                    data_df: pd.DataFrame = pd.DataFrame([row])
                data_df.to_csv(patient_data_path, index=False)
    

def main():
    database_path: str
    mimic_version: str
    chunksize: int
    database_path, mimic_version, chunksize = parse_args()

    if mimic_version == "mimic-iii":
        # check file completion
        if check_mimic_structure(database_path, mimic_version=mimic_version):
            dataset_path: str = os.path.join(database_path, "physionet.org", "files", "mimiciii", "1.4")
            if os.path.exists(os.path.join(dataset_path, "PATIENTS.csv.gz")):
                patients_file_path: str = os.path.join(dataset_path, "PATIENTS.csv.gz")
            else:
                patients_file_path: str = os.path.join(dataset_path, "PATIENTS.csv")
            os.makedirs("dataset", exist_ok=True)
            create_patients_files(patients_file_path, "dataset", mimic_version, chunksize)

            for data_name, file_name in processing_files.items():
                os.path.join(database_path, f"{file_name}.gz")
                if os.path.exists(os.path.join(database_path, f"{file_name}.gz")):
                    file_path: str = os.path.join(database_path, f"{file_name}.gz")
                else:
                    file_path: str = os.path.join(database_path, file_name)
                load_data(file_path, data_name, "dataset", mimic_version, chunksize)

        else:
            raise ValueError("MIMIC database structure incomplete.")
        
    elif mimic_version == "mimic-iv":
        raise NotImplementedError
    
    elif mimic_version == "mimic-cxr":
        raise NotImplementedError
    
    else:
        raise ValueError(f"{mimic_version} is not a valid mimic version, choose between mimic-iii, iv, or cxr.")


if __name__ == "__main__":
    main()