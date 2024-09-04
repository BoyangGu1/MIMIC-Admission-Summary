import argparse
import importlib
import json
import os
import re
import types

from typing import Iterator, Optional

import pandas as pd
from tqdm import tqdm


def parse_args() -> tuple[str, str, str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=False, type=str, default=".", help="location for mimic databse, such path should contain a physionet.org folder")
    parser.add_argument("--version", required=False, type=str, choices=["mimic-iii", "mimic-iv", "mimic-cxr"], default="mimic-iii", help="choose between mimic-iii, iv, and cxr")
    parser.add_argument("--heading_type", required=False, type=str, choices=["hpc", "pmh", "psh", "pe"], default="hpc", help="section you want to extract, choosing between hpc (hospital course), pmh (past medical history), psh (past surgical history), or pe (physical examination)")
    parser.add_argument("--chunksize", required=False, type=int, default=10000, help="chunksize for loading data")
    args = parser.parse_args()
    return args.database_path, args.version, args.heading_type, args.chunksize
    

def get_valid_nondischarge_notes(admission_notes: pd.DataFrame, 
                                 discharge_summary_date: str | float, 
                                 discharge_summary_time: str | float) -> pd.DataFrame:
    """_summary_

    Args:
        admission_notes (pd.DataFrame): _description_
        discharge_summary_date (str | float): _description_
        discharge_summary_time (str | float): _description_

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    if not pd.isna(discharge_summary_time):
        valid_time: pd.Timestamp = pd.to_datetime(discharge_summary_time)
    elif not pd.isna(discharge_summary_date):
        valid_time: pd.Timestamp = pd.to_datetime(discharge_summary_date) + pd.Timedelta(days=1)
    else:
        raise ValueError("Both date and time are nan.")
    
    # Filter valid notes based on time and category
    for i, note in admission_notes.iterrows():
        if pd.isna(note["CHARTTIME"]):
            note["CHARTTIME"] = note["CHARTDATE"]
    valid_notes_indicator: pd.Series[bool] = admission_notes["CHARTTIME"].apply(lambda charttime: pd.to_datetime(charttime) <= valid_time)
    valid_notes: pd.DataFrame = admission_notes[valid_notes_indicator]
    valid_notes = valid_notes[valid_notes["CATEGORY"] != "Discharge summary"]
    
    # Sort DataFrame by CHARTTIME (assuming CHARTTIME is datetime format)
    sorted_notes: pd.DataFrame = valid_notes.sort_values(by="CHARTTIME")
    
    return sorted_notes


def clean_one_text(note: str | list[str]) -> str:
    """_summary_

    Args:
        note (str | list[str]): _description_

    Returns:
        str: _description_
    """
    if isinstance(note, str):
        lines: list[str] = note.split("\n")
    else:
        lines = note.copy()

    # Remove empty lines from the beginning
    while lines and not lines[0].strip():
        lines.pop(0)
    # Remove empty lines from the end
    while lines and not lines[-1].strip():
        lines.pop()
    # replace # and . into empty
    lines = [line if line != "#" and line != "." else "" for line in lines]

    cleaned_text: str = ""
    section_text: str = ""

    for line in lines:
        if line == lines[-1]:
            section_text += (" " + line)

        if line == "" or line == lines[-1]:
            # remove any excess space
            section_text = re.sub(r"\s+", " ", section_text).strip()
            # lowercase the text
            section_text = section_text.lower()

            section_text += "\n"
            cleaned_text += section_text

            # reset for next section
            section_text = ""

        else:
            section_text += (" " + line)

    clean_lines = cleaned_text.split("\n")
    clean_lines = [line for line in clean_lines if line != ""]
    cleaned_text = "\n".join(clean_lines)

    return cleaned_text


def clean_nondischarge_texts(notes: pd.DataFrame) -> list[str]:
    """_summary_

    Args:
        notes (pd.DataFrame): _description_

    Raises:
        ValueError: _description_

    Returns:
        list[str]: _description_
    """

    necessary_columns: list[str] = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "DESCRIPTION", "TEXT"]
    for necessary_column in necessary_columns:
        if not necessary_column in notes.columns:
            raise ValueError(f"{necessary_column} column should be included in the notes.")

    clean_texts_list: list[str] = []
    for index, note in notes.iterrows():
        cleaned_text: str = clean_one_text(note["TEXT"])
        charttime: str = note["CHARTTIME"]
        category: str = note["CATEGORY"]
        description: str = note["DESCRIPTION"]

        # Assert that all variables are strings
        assert isinstance(charttime, str)
        assert isinstance(category, str)
        assert isinstance(description, str)

        if description.strip() == "":
            prompt: str = f"chart time: {charttime.lower()}\nnote category: {category.lower()}\n"
        else:
            prompt: str = f"chart time: {charttime.lower()}\nnote category: {category.lower()}\nnote category description: {description.lower()}\n"

        clean_texts_list.append(prompt + cleaned_text)

    return clean_texts_list


def remove_and_clean_heading(section_lines: list[str], heading: str) -> str:
    """_summary_

    Args:
        section_lines (list[str]): _description_
        heading (str): _description_

    Returns:
        str: _description_
    """
    clean_section: str = clean_one_text(section_lines)
    assert clean_section.startswith(heading.lower())
    # remove heading like (hospital course:)
    clean_section = ":".join(clean_section.split(":")[1:]).strip()

    return clean_section

def extract_section(discharge_summary_text: str, 
                    headings: list[str], 
                    next_headings: list[str], 
                    ignore_headings: list[str],
                    heading_keyword: str,
                    ) -> tuple[int, Optional[str], Optional[str]]:
    """_summary_

    Args:
        discharge_summary_text (str): _description_
        headings (list[str]): _description_
        next_headings (list[str]): _description_
        ignore_headings (list[str]): _description_
        heading_keyword (str): _description_

    Returns:
        tuple[int, Optional[str], Optional[str]]: _description_
    """
    
    # remove last few lines that only explains who dictate the discharge summary
    lines: list[str] = discharge_summary_text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Dictated By"):
            lines = lines[:i]
    if lines[-1].strip() == "":
        lines.pop()

    # match line that looks like [**First Name5 (NamePattern1) **] [**Last Name (NamePattern1) **] MD [**MD Number(2) 617**]
    name_pattern: str = r'\[\*\*(.*)\*\*\](.*)\[\*\*(.*)\*\*\](.*)\[\*\*(.*)\*\*\](.*)'
    match: bool = bool(re.fullmatch(name_pattern, lines[-1].replace(" ", "")))
    if match:
        lines.pop()

    # remove any lines that only contains one . or #
    lines = [line if line != "#" and line != "." else "" for line in lines]

    detected_headings: list[str] = []
    find_next_heading_indicator: bool = False
    heading_line: Optional[int] = None
    next_heading_line: Optional[int] = None

    # extract each section and its heading (if exist)
    section_text: str = ""
    section_start_line: int = 0
    for i, line in enumerate(lines):
        if line == lines[-1]:
            section_text += (" " + line)

        if line == "" or line == lines[-1]:
            section_text = section_text.replace("\n", " ").strip()
            if ":" in section_text:
                section_firstword = section_text.split(":")[0]
                if (heading_keyword.lower() in section_firstword.lower()
                    and not any(char.isdigit() for char in section_firstword)
                    and not section_firstword.startswith((".", "#"))):
                    if detected_headings == []:
                        detected_headings.append(section_firstword)
                        heading_line = section_start_line
                        find_next_heading_indicator = True
                    else:
                        return 0, None, None

                elif find_next_heading_indicator and section_firstword in next_headings:
                    next_heading_line = section_start_line
                    find_next_heading_indicator = False

            section_text = ""
            section_start_line = i
        else:
            section_text += (" " + line)

    if detected_headings == []:
        return 7, None, None
    
    if not detected_headings[0] in headings:
        return 6, None, None

    if next_heading_line:
        return 1, detected_headings[0], remove_and_clean_heading(lines[heading_line:next_heading_line], detected_headings[0])

    section: list[str] = lines[heading_line:]
    detect_unknown_upper_heading: bool = False

    for line in section:
        if line == "" or line == lines[-1]:
            section_text = section_text.replace("\n", " ")
            if ":" in section_text:
                section_firstword = section_text.split(":")[0]
                if (section_firstword.isupper() 
                    and not section_firstword in ignore_headings
                    and not any(char.isdigit() for char in section_firstword)
                    and not section_firstword.startswith((".", "#"))):
                
                    detect_unknown_upper_heading = True

            section_text = ""
        else:
            section_text += line

    if detected_headings[0].isupper() and not detect_unknown_upper_heading:
        return 2, detected_headings[0], remove_and_clean_heading(section, detected_headings[0])

    elif detected_headings[0].isupper() and detect_unknown_upper_heading:
        return 3, None, None

    elif not detected_headings[0].isupper() and not detect_unknown_upper_heading:
        return 4, detected_headings[0], remove_and_clean_heading(section, detected_headings[0])

    else:
        assert not detected_headings[0].isupper() and detect_unknown_upper_heading
        return 5, None, None


def extract_admission_document_summary_pairs(heading_type: str, 
                                             data_by_patient_path: str, 
                                             store_path: str, 
                                             mimic_version: str,
                                             chunksize: int = 10000,
                                             summary_words_lower_bound: int = 30
                                            ) -> None:
    """Extract and store document-summary pairs from mimic noteevents. In the store path, there should be a by_{heading_type} folder that each subfolder corresponding to code 1, 2 and 4. Each code file contains files that are discriminated by the heading name (Note that all headings are of heading_type). Each file contains document-summary pairs. The file has three columns: PAIR-ID, DOCUMENT, and SUMMARY. Further more, PAIR_IDs of code 0, 3, 5, 6, 7 are stored under constants/mimic_version/heading_type.

    code 0: We found more than one headings of heading_type.
    code 1: We found one known heading and one known successive heading.
    code 2: We found one known heading (in non-uppercase) of heading_type and there is no successive heading.
    code 3: We found one known heading (in uppercase) of heading_type, but the successive capital heading (in uppercase) is unknown.
    code 4: We found one known heading (in uppercase) of heading_type and there is no successive heading.
    code 5: We found one known heading (in non-uppercase) of heading_type, but the successive capital heading (in uppercase) is unknown.
    code 6: We found one unknown heading of heading_type.
    code 7: We found no headings of heading_type.
    code 8: The dicharge summary has 'Addendum' as the description.
    code 9: The extracted section contains less than {summary_words_lower_bound} words (Note: code 9 is at higher priority than code 1, 2, and 4).

    Args:
        heading_type (str): The section that serves as the summary in the discharge summary. It can be hpc (hospital course), pmh (past medical history), psh (past surgical history), or pe (physical examination).
        data_by_patient_path (str): Path storing every patient's info.
        store_path (str): Path to save the document-summary pair.
        mimic_version (str): MIMIC version. It can be mimic-iii, iv, or cxr.
        chunksize (int, optional): Chunksize when combining all pairs. Defaults to 10000.
        summary_words_lower_bound (int, optional): Lower bound for code 9. Defaults to 30.
    """

    if heading_type == "hpc": # hospital course
        heading_constants: types.ModuleType = importlib.import_module(f"constants.{mimic_version}.{heading_type}.headings_info")
        headings_dict: dict[str, int] = heading_constants.headings_dict
        next_headings: list[str] = heading_constants.next_headings
        ignore_headings: list[str] = heading_constants.ignore_headings
        heading_keyword: str = heading_constants.heading_keyword

    elif heading_type == "pmh":
        raise NotImplementedError
    
    elif heading_type == "psh":
        raise NotImplementedError
    
    elif heading_type == "pe":
        raise NotImplementedError
    
    else:
        raise ValueError("Heading type not supported. Please use hpc (hospital course), pmh (past medical history), psh (past surgical history), or pe (physical examination).")
    
    store_pair_codes: list[int] = [1, 2, 4]
    only_stats_codes: list[int] = [0, 3, 5, 6, 7, 8, 9]

    # create directory
    empty_df: pd.DataFrame = pd.DataFrame(columns=["PAIR_ID", "DOCUMENT", "SUMMARY"])
    for code in store_pair_codes:
        os.makedirs(os.path.join(store_path, f"by_{heading_type}", f"code{code}"), exist_ok=True)
        for heading, index in headings_dict.items():
            empty_df.to_csv(os.path.join(store_path, f"by_{heading_type}", f"code{code}", f"{index}.csv"), index=False)
    empty_df.to_csv(os.path.join(store_path, f"by_{heading_type}", "all.csv"), index=False)

    # store stats
    invalid_cases_id_dict: dict[int, list[str]] = {}
    for code in only_stats_codes:
        invalid_cases_id_dict[code] = []
    valid_cases_stats_dict: dict[int, list[tuple[str, int, int]]] = {}
    for code in store_pair_codes:
        valid_cases_stats_dict[code] = []

    valid_cases_count: int = 0
    # loop through each patient
    with tqdm(total=len(os.listdir(data_by_patient_path)), desc="Processing each patient's info") as pbar:
        for subject_id in os.listdir(data_by_patient_path):
            patient_path: str = os.path.join(data_by_patient_path, subject_id)

            if (os.path.exists(os.path.join(patient_path, "ADMISSIONS.csv"))
                and os.path.exists(os.path.join(patient_path, "NOTEEVENTS.csv"))):

                admission_df: pd.DataFrame = pd.read_csv(os.path.join(patient_path, "ADMISSIONS.csv"))
                note_df: pd.DataFrame = pd.read_csv(os.path.join(patient_path, "NOTEEVENTS.csv"))

                # loop through each admission
                for index, row in admission_df.iterrows():
                    hadm_id: str = row["HADM_ID"]
                    admission_notes: pd.DataFrame = note_df[note_df["HADM_ID"] == hadm_id]
                    discharge_summaries: pd.DataFrame = admission_notes[admission_notes["CATEGORY"] == "Discharge summary"]

                    # loop through each discharge summary, since there might be more than one discharge summary for one admission
                    for summary_index, discharge_summary in discharge_summaries.iterrows():
                        row_id = discharge_summary["ROW_ID"]
                        pair_id = f"{row_id}_{subject_id}_{hadm_id}"

                        if discharge_summary["DESCRIPTION"] == "Addendum":
                            invalid_cases_id_dict[8].append(pair_id)
                        
                        else:
                            # only extract non-discharge summary notes that is charted before the discharge summary
                            valid_nondischarge_notes: pd.DataFrame = get_valid_nondischarge_notes(admission_notes, 
                                                                                                discharge_summary["CHARTDATE"], 
                                                                                                discharge_summary["CHARTTIME"])
                            # clean text: all lower case, remove excess spacing, and add notes information
                            cleaned_valid_nondischarge_notes: list[str] = clean_nondischarge_texts(valid_nondischarge_notes)
                            valid_nondischarge_notes['CLEANED_TEXT'] = cleaned_valid_nondischarge_notes
                            # join all notes by [DOC_SEP]
                            joined_cleaned_valid_nondischarge_notes: str = " [DOC_SEP] ".join(cleaned_valid_nondischarge_notes)

                            code, section_heading, section_text = extract_section(discharge_summary["TEXT"], 
                                                                                    list(headings_dict.keys()), 
                                                                                    next_headings, 
                                                                                    ignore_headings,
                                                                                    heading_keyword)

                            # store data
                            if code in only_stats_codes:
                                invalid_cases_id_dict[code].append(pair_id)

                            elif code in store_pair_codes:
                                assert section_heading is not None
                                assert section_text is not None

                                summary_words: int = len(section_text.split(" "))
                                if summary_words <= summary_words_lower_bound:
                                    invalid_cases_id_dict[9].append(pair_id)

                                else:
                                    document_words: int = len(joined_cleaned_valid_nondischarge_notes.split(" "))
                                    data = {
                                        "PAIR_ID": [pair_id],
                                        "DOCUMENT": [joined_cleaned_valid_nondischarge_notes],
                                        "SUMMARY": [section_text],
                                    }
                                    df = pd.DataFrame(data)
                                    df.to_csv(os.path.join(store_path, f"by_{heading_type}", f"code{code}", f"{headings_dict[section_heading]}.csv"), 
                                            mode='a', 
                                            index=False, 
                                            header=False)
                                    
                                    valid_cases_stats_dict[code].append((pair_id, document_words, summary_words))

                                    valid_cases_count += 1

            pbar.update(1)

    # store stats
    for code in only_stats_codes:
        with open(os.path.join("constants", mimic_version, heading_type, f"code{code}.json"), 'w') as json_file:
            json.dump(invalid_cases_id_dict[code], json_file, indent=4)
    for code in store_pair_codes:
        with open(os.path.join("constants", mimic_version, heading_type, f"code{code}.json"), 'w') as json_file:
            json.dump(valid_cases_stats_dict[code], json_file, indent=4)

    # combine all valid cases
    write_header = True
    with tqdm(total=valid_cases_count, desc="Combining all valid cases") as pbar:
        for code in store_pair_codes:
            for heading, index in headings_dict.items():
                for chunk in pd.read_csv(os.path.join(store_path, f"by_{heading_type}", f"code{code}", f"{index}.csv"), chunksize=chunksize):
                    assert isinstance(chunk, pd.DataFrame)
                    chunk.to_csv(os.path.join(store_path, f"by_{heading_type}", "all.csv"), mode="a", index=False, header=write_header)
                    pbar.update(len(chunk))
                    write_header = False


if __name__ == "__main__":
    mimic_database_path: str
    mimic_version: str
    heading_type: str
    chunksize: int
    mimic_database_path, mimic_version, heading_type, chunksize = parse_args()

    # make directory for saving
    store_path: str = os.path.join("dataset", mimic_version)
    os.makedirs(store_path, exist_ok=True)

    if mimic_version == "mimic-iii":

        # check if data has been pre-processed by patient, if not, run general_data_preparation.py to construct it
        data_by_patient_path: str = os.path.join("dataset", mimic_version, "by_patient")
        if not os.path.exists(data_by_patient_path):
            raise ValueError(f"{data_by_patient_path} does not exist, run general_data_preparation.py to construct it.")

        # process data
        extract_admission_document_summary_pairs(heading_type, data_by_patient_path, store_path, mimic_version, chunksize=chunksize)
    
    elif mimic_version == "mimic-iv":
        raise NotImplementedError
    
    elif mimic_version == "mimic-cxr":
        raise NotImplementedError
    
    else:
        raise ValueError(f"{mimic_version} is not a valid mimic version, choose between mimic-iii, iv, or cxr.")