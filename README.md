# MIMIC-Admission-Summary

This repository is served for the presentation and reproduction of the master thesis of Boyang Gu. The project developed a comprehensice dataset for brief hospital course summarization and trained several models that achieve SOTA performance.

## Setup

1. Clone this repository: `git clone https://github.com/BoyangGu1/MIMIC-Admission-Summary.git`.

2. Run the following commands:

    ```
    cd MIMIC-Admission-Summary
    mkdir dataset
    mkdir DPO_rejected_summary
    mkdir inference
    mkdir medcat_model
    mkdir metrics
    mkdir metrics/BERT_score
    mkdir metrics/MEDCON
    mkdir metrics/ROUGE
    mkdir outputs
    mkdir quickumls_install
    mkdir umls-2024AA
    mkdir unsloth_DPO_models
    mkdir unsloth_SFT_models
    mkdir unsloth_rewriting_SFT_models
    mkdir vllm_DPO_models
    mkdir vllm_SFT_models
    mkdir vllm_rewriting_SFT_models
    mkdir mask_dfs
    mkdir medcat_extraction
    mkdir rewrite_responses
    mkdir unsloth_rewriting_SFT_models
    mkdir vllm_rewriting_SFT_models
    ```

2. Follow the instructions at https://mimic.mit.edu/docs/gettingstarted/ and download the MIMIC-III database from https://physionet.org/content/mimiciii/1.4/ then put it under the repository directory.

3. Install the virtual environment via Anaconda: 

    ```
    conda env create -f unsloth_env.yaml
    conda env create -f mimic_env.yaml
    conda env create -f medcat_env.yaml
    ```

4. Create a QuickUMLS installation by downloading the 2024AA UMLS `MRCONSO.RRF` and `MRSTY.RRF` files from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html under the `umls-2024AA` folder. Then run the following command:

    ```
    conda activate mimic_env
    python -m quickumls.install umls-2024AA quickumls_install
    ```

    For detailed instuctions, please refer to https://github.com/Georgetown-IR-Lab/QuickUMLS.

5. Create a MedCAT installation by downloading the UMLS Full model by asking for permisstion at https://uts.nlm.nih.gov/uts/login?service=https://medcat.rosalind.kcl.ac.uk/auth-callback. The model should be named as `umls_self_train_model_pt2ch_3760d588371755d0.zip`. Put it under the folder `medcat_model` and unzip it. Then download the 2022AA UMLS `MRCONSO.RRF` and `MRSTY.RRF` files from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html and also put them under the folder `medcat_model`.

    For detailed instuctions, please refer to https://github.com/CogStack/MedCAT.

After the setup, the repository should includes the following:

```
MIMIC-Admission-Summary
├── medcat_model
│   ├── umls_self_train_model_pt2ch_3760d588371755d0
│   │   ├── en_core_web_md
│   │   │   ├── ...
│   │   ├── cdb.dat
│   │   ├── model_card.json
│   │   ├── vocab.dat
│   ├── MRCONSO.RRF
│   ├── MRSTY.RRF
|   ├── umls_self_train_model_pt2ch_3760d588371755d0.zip
├── physionet.org
│   ├── files/mimiciii/1.4
│   │   ├── ...
│   ├── robots.txt
├── quickumls_install
│   ├── ...
├── umls-2024AA
│   ├── MRCONSO.RRF
│   ├── MRSTY.RRF
├── ...
```

## Reproduction of the results

By the restriction of the license of MIMIC database, we cannot provide the processed data and models. However, we provide the code to reproduce the results reported in the report.

1. Preprocess the MIMIC-III dataset:

    ```
    conda activate mimic_env
    python general_data_preparation.py
    python one_admission_data_prep.py
    ```

2. Model training:

    2.1. Supervised Fine-Tuning (SFT):

    ```
    conda activate unsloth_env
    python SFT_train.py SFT_training_paras/sft_para1.json
    ```

    You can change `sft_para1.json` into `sft_para2.json` or `sft_para3.json` to train different models.

    2.2. Direct Preference Optimization (DPO):

    ```
    conda activate mimic_env
    python DPO_rejected_prep.py \
        --gpus 0 \
        --csv_path dataset/mimic-iii/by_hpc/Meta-Llama-3.1-8B_hpc1_32768/train.csv \
        --save_path DPO_rejected_summary/mimic-iii/by_hpc/sft_para3/train
    python DPO_rejected_prep.py \
        --gpus 0 \
        --csv_path dataset/mimic-iii/by_hpc/Meta-Llama-3.1-8B_hpc1_32768/val.csv \
        --save_path DPO_rejected_summary/mimic-iii/by_hpc/sft_para3/val
    conda deactivate
    conda activate unsloth_env
    python DPO_train.py DPO_training_paras/dpo_para1.json
    ```

    You can change `dpo_para1.json` into `dpo_para2.json`, `dpo_para3.json`, `dpo_para4.json`, or `dpo_para5.json` to train different models. `DPO_rejected_prep.py` supports multi-GPU settings so feel free to change the `gpus` argument to `0,1,2,3,4,5,6,7` for example. Sometimes due to different cuda initialization method, you may need to set `CUDA_VISIBLE_DEVICES` first to ensure multi-GPU inference (e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` for example).

3. Inference for the trained models (`sft_para1` for example):

    ```
    conda activate unsloth_env
    python unsloth2vllm.py \
        --model_name unsloth_SFT_models/sft_para1 \
        --vllm_save_path vllm_SFT_models/sft_para1
    conda deactivate
    conda activate mimic_env
    python vllm_inference.py \
        --model_name vllm_SFT_models/sft_para1 \
        --gpus 0 \
        --csv_path dataset/mimic-iii/by_hpc/test.csv \
        --prompt_path dataset/mimic-iii/by_hpc/Meta-Llama-3.1-8B_hpc1_32768/prompt.txt \
        --save_path inference/mimic-iii/by_hpc/sft_para1
    ```

    The inferece supports multi-GPU settings too.

4. Zero-shot inference:

    ```
    conda activate mimic_env
    python vllm_zeroshot.py \
        --gpus 0 \
        --csv_path dataset/mimic-iii/by_hpc/test.csv \
        --save_path inference/mimic-iii/by_hpc/zeroshot
    ```

    The inferece supports multi-GPU settings too.

5. Cloze-form rewriting:

    5.1 MedCAT extraction:

        ```
        conda activate mimic_env
        python medcat_extraction.py \
            --dataset_path dataset/mimic-iii/by_hpc/train_val.csv \
            --save_path dataset/mimic-iii/by_hpc/medcat_extraction_train_val 
        python medcat_extraction.py \
            --dataset_path dataset/mimic-iii/by_hpc/test.csv \
            --save_path dataset/mimic-iii/by_hpc/medcat_extraction_test
        ```

    5.2 Training datasets preparation:

        ```
        conda activate mimic_env
        python rewrite_cloze_data_prep.py \
            --dataset_path dataset/mimic-iii/by_hpc/Meta-Llama-3.1-8B_hpc1_32768/train.csv \
            --medcat_extraction_dir dataset/mimic-iii/by_hpc/medcat_extraction_train_val \
            --save_path dataset/mimic-iii/by_hpc/Meta-Llama-3.1-8B_hpc1_32768 \
            --save_name train
        python rewrite_cloze_data_prep.py \
            --dataset_path dataset/mimic-iii/by_hpc/Meta-Llama-3.1-8B_hpc1_32768/val.csv \
            --medcat_extraction_dir dataset/mimic-iii/by_hpc/medcat_extraction_train_val \
            --save_path dataset/mimic-iii/by_hpc/Meta-Llama-3.1-8B_hpc1_32768 \
            --save_name val
        python rewrite_cloze_data_prep.py \
            --dataset_path dataset/mimic-iii/by_hpc/Meta-Llama-3.1-8B_hpc1_32768/test.csv \
            --medcat_extraction_dir dataset/mimic-iii/by_hpc/medcat_extraction_test \
            --save_path dataset/mimic-iii/by_hpc/Meta-Llama-3.1-8B_hpc1_32768 \
            --save_name test
        ```

    5.3 SFT-based rewriting training:

        ```
        conda activate unsloth_env
        python rewrite_SFT_train.py rewrite_SFT_training_paras/rewrite_sft_para1.json
        ```

    5.4 Rewriting inference:
    
    5.4.1 SFT-based rewriting (use `rewrite_sft_para1` to rewrite `sft_para1' for example):

        ```
        conda activate unsloth_env
        python unsloth2vllm.py \
            --model_name unsloth_rewriting_SFT_models/rewrite_sft_para1 \
            --vllm_save_path vllm_rewriting_SFT_models/rewrite_sft_para1
        conda deactivate
        conda activate medcat_env
        python medcat_extraction.py \
            --summary_path inference/mimic-iii/by_hpc/sft_para1 \
            --save_path medcat_extraction/mimic-iii/by_hpc/sft_para1
        conda deactivate
        conda activate mimic_env
        python rewrite_model_based_inference.py \
            --gpus 0 \
            --ref_pairs_csv dataset/mimic-iii/by_hpc/test.csv \
            --ref_summary_path inference/mimic-iii/by_hpc/sft_para1 \
            --save_path inference/mimic-iii/by_hpc/sft_para1_maskall_with_rewrited_by_rewrite_para1 \
            --save_medcat_extraction_path medcat_extraction/mimic-iii/by_hpc/sft_para1 \
            --save_mask_dfs_save_path mask_dfs/mimic-iii/by_hpc/sft_para1 \
            --rewrite_model vllm_rewriting_SFT_models/rewrite_sft_para1 \
            --save_rewrite_response_name rewrite_responses/mimic-iii/by_hpc/sft_para1_maskall_with_rewrited_by_rewrite_para1
        ```

    5.4.2 Few-shot training-free rewriting (rewriting `sft_para1` for example):

        ```
        conda activate mimic_env
        python rewrite_fewshot_inference.py \
            --gpus 0 \
            --shots 0 \
            --ref_pairs_csv dataset/mimic-iii/by_hpc/test.csv \
            --ref_summary_path inference/mimic-iii/by_hpc/sft_para1 \
            --save_path inference/mimic-iii/by_hpc/sft_para1_maskall_with_rewrited_by_0shots \
            --save_medcat_extraction_path medcat_extraction/mimic-iii/by_hpc/sft_para1 \
            --save_mask_dfs_save_path mask_dfs/mimic-iii/by_hpc/sft_para1 \
            --save_rewrite_response_name rewrite_responses/mimic-iii/by_hpc/sft_para1_maskall_with_rewrited_by_0shot
        ```

    The inferece supports multi-GPU settings too.

6. Compute metrics (`sft_para1` for example):

    ```
    conda activate mimic_env
    python compute_metrics.py \
        --ref_path dataset/mimic-iii/by_hpc/test.csv \
        --cand_path inference/mimic-iii/by_hpc/sft_para1 \
        --save_name sft_para1
    ```

    The metrics computed are BERTScore, MEDCON, and ROUGE.
