import transformers
import matplotlib.pyplot as plt

from bert_score import BERTScorer
from rouge_score import rouge_scorer
from quickumls import QuickUMLS
from unsloth import FastLanguageModel

import pandas as pd
from tqdm import tqdm
import argparse

import os


def parse_args() -> tuple[str, str, str, str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", required=True, type=str, help="location for reference summaries, should be a csv file")
    parser.add_argument("--cand_path", required=True, type=str, help="location for candidate summaries, should be a folder containing txt files named by pair_ids")
    parser.add_argument("--save_name", required=True, type=str, help="the name for saving")
    parser.add_argument("--gpu", required=False, type=str, default="0", help="the gpu index for this script to run on")
    parser.add_argument("--max_tokens", required=False, type=int, help="if the document is larger than this number of tokens, this sample will be ignored for a seperated evaluation")
    args = parser.parse_args()
    return args.ref_path, args.cand_path, args.save_name, args.gpu, args.max_tokens


def get_cuis(matches):
    cuis = []
    for match in matches:
        for m in match:
            cuis.append(m['cui'])
    return set(cuis)


def get_count_tokens_fn(tokenizer, max_tokens):
    def count_tokens(example):
        return len(tokenizer(example['DOCUMENT'], return_tensors='pt')["input_ids"][0]) < max_tokens
    return count_tokens


def compute_metrics(test_df, cand_path, name):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge_results_list = []
    # Loop through each row in the test_df
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Calculating ROUGE scores"):
        ref_result = row["SUMMARY"]
        pair_id = row["PAIR_ID"]
        
        with open(os.path.join(cand_path, f"{pair_id}.txt")) as file:
            generated_text = file.read()
        
        # Calculate ROUGE scores
        scores = scorer.score(ref_result, generated_text)
        
        # Store the results in a dictionary
        rouge_results_list.append({
            'pair-id': pair_id,
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure
        })

    rouge_results_df = pd.DataFrame(rouge_results_list)
    rouge_results_df.to_csv(os.path.join("metrics", "ROUGE", f"{name}.csv"), index=False)

    ###################################################################################################

    bertscore_results_list = []
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Calculating BERTScore'):
        ref_summary = row['SUMMARY']
        pair_id = row['PAIR_ID']
        with open(os.path.join(cand_path, f"{pair_id}.txt"), 'r') as f:
            cand_summary = f.read()

        if cand_summary != '':
            P, R, F1 = scorer.score([cand_summary], [ref_summary])
            bertscore_results_list.append({'PAIR_ID': pair_id, 'P': P, 'R': R, 'F1': F1})

    bertscore_results_df = pd.DataFrame(bertscore_results_list)
    bertscore_results_df.to_csv(os.path.join("metrics", "BERT_score", f"{name}.csv"), index=False)

    ###################################################################################################

    quickumls_fp='quickumls_install'

    UMLS_semantics_types=['T116', 'T195', 'T123', 'T122', 'T200', 'T196', 'T126', 'T131', 'T125', 'T129', 'T130', 'T197', 'T114', 'T109', 'T121', 'T127', 'T020', 'T190', 'T049', 'T019', 'T047', 'T050', 'T033', 'T037', 'T048', 'T191', 'T046', 'T184', 'T060', 'T059', 'T063', 'T061']
    UMLS_semantics_names=['Amino Acid, Peptide, or Protein', 'Antibiotic', 'Biologically Active Substance', 'Biomedical or Dental Material', 'Clinical Drug', 'Element, Ion, or Isotope', 'Enzyme', 'Hazardous or Poisonous Substance', 'Hormone', 'Immunologic Factor', 'Indicator, Reagent, or Diagnostic Aid', 'Inorganic Chemical', 'Nucleic Acid, Nucleoside, or Nucleotide', 'Organic Chemical', 'Pharmacologic Substance', 'Vitamin', 'Acquired Abnormality', 'Anatomical Abnormality', 'Cell or Molecular Dysfunction', 'Congenital Abnormality', 'Disease or Syndrome', 'Experimental Model of Disease', 'Finding', 'Injury or Poisoning', 'Mental or Behavioral Dysfunction', 'Neoplastic Process', 'Pathologic Function', 'Sign or Symptom', 'Diagnostic Procedure', 'Laboratory Procedure', 'Molecular Biology Research Technique', 'Therapeutic or Preventive Procedure']
    UMLS_type_map={
        "Treatment": ["Amino Acid, Peptide, or Protein", "Antibiotic", "Biologically Active Substance",
            "Biomedical or Dental Material", "Chemical", "Chemical Viewed Functionally",
            "Chemical Viewed Structurally", "Clinical Drug", "Element, Ion, or Isotope",
            "Enzyme",  "Hazardous or Poisonous Substance", "Hormone",
            "Immunologic Factor", "Indicator, Reagent, or Diagnostic Aid",
            "Inorganic Chemical", "Nucleic Acid, Nucleoside, or Nucleotide",
            "Organic Chemical", "Pharmacologic Substance",
            "Receptor", "Vitamin", "Therapeutic or Preventive Procedure"],
        "Disease": ["Acquired Abnormality", "Anatomical Abnormality",
            "Cell or Molecular Dysfunction", "Congenital Abnormality",
            "Disease or Syndrome", "Experimental Model of Disease", "Finding", "Injury or Poisoning",
            "Mental or Behavioral Dysfunction", "Neoplastic Process", "Pathologic Function", "Sign or Symptom"],
        "Test": [ "Diagnostic Procedure", "Laboratory Procedure", "Molecular Biology Research Technique"],
    }

    matcher = QuickUMLS(quickumls_fp,window=5,threshold=0.9,accepted_semtypes=UMLS_semantics_types)

    medcon_results_list = []
    # use tqdm to show progress bar
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Calculating MEDCON'):
        ref_summary = row['SUMMARY']
        pair_id = row['PAIR_ID']
        with open(os.path.join(cand_path, f"{pair_id}.txt"), 'r') as f:
            cand_summary = f.read()
        
        ref_matches = matcher.match(ref_summary, ignore_syntax=True)
        cand_matches = matcher.match(cand_summary, ignore_syntax=True)

        ref_cuis = get_cuis(ref_matches)
        cand_cuis = get_cuis(cand_matches)
        intersection_cuis = ref_cuis.intersection(cand_cuis)

        if len(cand_cuis) == 0:
            precision = 1
        else:
            precision = len(intersection_cuis) / len(cand_cuis)
        if len(ref_cuis) == 0:
            recall = 1
        else:
            recall = len(intersection_cuis) / len(ref_cuis)

        medcon_results_list.append({'PAIR_ID': pair_id, 'Precision': precision, 'Recall': recall})

    medcon_results_df = pd.DataFrame(medcon_results_list)
    medcon_results_df.to_csv(os.path.join("metrics", "MEDCON", f"{name}.csv"), index=False)

    ###################################################################################################

    print(f"result for {name}:")

    average_rouge_scores = rouge_results_df.mean(numeric_only=True)
    # Print the average ROUGE scores
    print("Average ROUGE scores:")
    print(average_rouge_scores)

    #get average metrics
    bertscore_avg_P = bertscore_results_df['P'].mean()
    bertscore_avg_R = bertscore_results_df['R'].mean()
    bertscore_avg_F1 = bertscore_results_df['F1'].mean()
    #print average metrics
    print(f"Average bertscore P: {bertscore_avg_P}")
    print(f"Average bertscore R: {bertscore_avg_R}")
    print(f"Average bertscore F1: {bertscore_avg_F1}")

    average_medcon_scores = medcon_results_df.mean(numeric_only=True)
    print(f'Average medcon Precision: {average_medcon_scores["Precision"]}')
    print(f'Average medcon Recall: {average_medcon_scores["Recall"]}')


if __name__ == '__main__':

    ref_path, cand_path, save_name, gpu, max_tokens = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    test_df: pd.DataFrame = pd.read_csv(ref_path)
    test_df = test_df.dropna(subset=["DOCUMENT"])

    if max_tokens:

        if not os.path.exists(f"{save_name}_restricted_{max_tokens}.csv"):
            _, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/Meta-Llama-3.1-8B", # YOUR MODEL YOU USED FOR TRAINING
                max_seq_length = 32768,
                dtype = None,
                load_in_4bit = True,
            )

            count_tokens_fn = get_count_tokens_fn(tokenizer, max_tokens)
            restricted_test_df = test_df[test_df.apply(count_tokens_fn, axis=1)]
            restricted_test_df.to_csv(f"{save_name}_restricted_{max_tokens}.csv", index=False)
        restricted_test_df = pd.read_csv(f"{save_name}_restricted_{max_tokens}.csv")
        compute_metrics(restricted_test_df, cand_path, f"{save_name}_restricted_{max_tokens}")

    compute_metrics(test_df, cand_path, save_name)