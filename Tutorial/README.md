# Comprehensive Tutorial
# FEEL: Quantifying Heterogeneity in Physiological Signals for Generalizable Emotion Recognition
*A step-by-step guide to reproducing results and extending the Framework on other datasets, different modalities, models and binning.*

## Table of Contents
1. [Installation & Environment Setup](#1-installation--environment-setup)  
2. [Dataset Preparation](#2-dataset-preparation)  
3. [Running Experiments](#3-running-experiments)  
4. [Extending the Framework](#4-extending-the-framework)  
5. [Troubleshooting](#5-troubleshooting)

##  1. Installation & Environment Setup

### 1.1 Prerequisites
Ensure your system meets the following requirements:

- **Python** ≥ 3.8  
- **CUDA-capable GPU** (Recommended: NVIDIA A100 / H100)  
- **Minimum RAM:** 16 GB  

### 1.2 Clone the Repository
```bash
git clone git@github.com:alchemy18/FEEL.git
cd FEEL
```

### 1.3 Create Virtual Environment
- Using conda:
```bash
conda create -n feel python=3.8
conda activate feel
```
- using venv:
  ```bash
  python -m venv feel_env
  source feel_env/bin/activate  # On Windows: feel_env\Scripts\activate
  ```
  
### 1.4 Install Dependencies
```bash
 # Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install -r requirements.txt
```

Key Project Dependencies
This paper relies on the following core libraries for physiological signal processing, machine learning, and data handling.
| **Dependency** | **Purpose** |
|-----------------|-------------|
| **NeuroKit2** | Advanced processing and analysis of physiological signals (e.g., ECG, PPG, EDA, respiration). |
| **scikit-learn** | Implementation of traditional machine learning models and utility functions (e.g., classification, regression, clustering, model selection). |
| **transformers** | Utilized for the Contrastive Language Signal Pre-training (CLSP) text encoder and fine-tuning of the CLSP using two Meta-Net families (CNN, MLP). |
| **pandas** | High-performance, easy-to-use data structures and data analysis tools for structured data manipulation. |
| **numpy** | Fundamental package for scientific computing, providing support for large, multi-dimensional arrays and matrices. |
| **matplotlib** | Comprehensive library for creating static, animated, and interactive visualizations in Python. |
| **seaborn** | Statistical data visualization library based on matplotlib. |

### 1.5 Expected Directory Structure
```bash
FFEL/
├── Datasets/
│   ├── [Individual Dataset]/
│         ├── Raw_EDA.csv
│         ├── Raw_PPG.csv 
│         ├── Features_EDA.csv
│         ├── Features_PPG.csv 
│         ├── Features_Combined.csv 
│         ├── Features_PPG.csv
│         ├── Features_FourClass_PPG.csv
│         ├── Features_FourClass_EDA.csv 
│         ├── Features_FourClass_Combined.csv
│         ├── Features_Combined_Demographics.csv (if applicable)
│         ├── Features_PPG_Demographics.csv (if applicable)
│         ├── Features_EDA_Demographics.csv (if applicable)
├── Benchmarking/
│   ├── Model_running.py # for running scripts for base model training
│   ├── Group_running.py # for running scripts for base model training
│   ├── Model_running_fourclass.py  # for running scripts for base model training for four-class binning
├── Fine_tuning/
│   ├── finetuning_CLSP_two_class_CNN.py # for fine-tuning clsp pre-trained model on 2-class with CNN Meta-net
│   ├── finetuning_CLSP_two_class_MLP.py # for fine-tuning clsp pre-trained model on 2-class with MLP Meta-net
│   ├── finetuning_CLSP_four_class_MLP.py # for fine-tuning clsp pre-trained model on 4-class with MLP Meta-net
│   ├── finetuning_CLSP_four_class_MLP.py # for fine-tuning clsp pre-trained model on 4-class with CNN Meta-net
├── Scripts/
│   ├── LDA.py
│   ├── MLP.py
│   ├── RandomForest.py
│   ├── clsp_com.py
│   ├── clsp_eda.py
│   ├── clsp_ppg.py
│   ├── lstm_features.py
│   ├── lstm_signals.py
│   ├── resnet_features.py
│   ├── resnet_signals.py
│   ├── transformer_features.py
│   ├── transformer_signals.py
│   ├── visualize.py # for visualizating TSNE, UMAP, and data distribution
│   ├── model.py # file to load base_model for analyzing PPG artifacts
│   ├── ppg_artifact.py
│   ├── eda_artifact.py
│   ├── SA_Detection.json # resource file for eda-artifacts
│   ├── TinyPPG_model_best_params.pth # resource file for ppg-artifacts
├── Group-Benchmarking/
│   ├── LDA_group.py
│   ├── MLP_group.py
│   ├── RandomForest_group.py
│   ├── accross_group_mlp_lda_rf.py
│   ├── clsp_group_setting.py
│   ├── finetuning_CLSP_group_CNN_representative.py
│   ├── finetuning_CLSP_group_representative.py
│   ├── mlp_rf_clsp_zero_shot_within_group.ipynb
├── Analysis/
│   ├── [Individual Dataset]/
│         ├── Benchmarking_analysis.ipynb
│         ├── vizualize_UMAP.ipynb
│         ├── Benchmarking_results.csv
│   ├── [Individual Group]/
│         ├── Benchmarking_analysis.ipynb
│         ├── vizualize_UMAP.ipynb
```

##  2. Dataset Preparation
### 2.1 Downloading Datasets

The framework uses 19 publicly available datasets. The details are given in the README.MD, you can follow each dataset's usage policy to download it. 
You can extend the framework to other datasets by following their usage policy. 
In order to submit the results, please fill out this form or reach out to us at pragyas@iiitd.ac.in. 

### 2.2 Preprocessing Steps

Each dataset has its own labelling and frequency. Please refer to the Data_Preparation.ipynb that has steps and example of preparing Exercise dataset. 

### 2.3 Cross Dataset Grouping

**Grouping Datasets by Cohorts and Preparing for Cross-Dataset Evaluation**

To conduct consistent experiments, we first grouped the datasets into common cohorts based on three key factors:
- Device used for data collection
- Experimental setup and conditions
- Labelling methodology

All this metadata is documented in the respective READMEs, dataset documentation, and associated research papers.

Once the grouping was complete, we merged the normalized feature CSV files of datasets belonging to the same cohort. This step created a single unified CSV file per cohort, which was then used for our experiments.

---

**Extending Cross-Dataset Evaluation to Demographic Parameters**

Beyond device- or experiment-based grouping, we also extended cross-dataset evaluation to demographic attributes such as age and gender.
- **For gender-based transferability**, we used nine datasets that include gender metadata: WESAD, ScientISST MOVE, UBFC_PHYS, Exercise, PhyMER, EmoWear, CASE, CEAP-360VR, and NURSE (female subjects only).
- **For age-based transferability**, we considered seven datasets: WESAD, ScientISST MOVE, Exercise, PhyMER, EmoWear, CASE, and CEAP-360VR.

---

**Handling Demographic Metadata Across Datasets**

Participant-level demographic information is available in only a subset of datasets, and the format varies widely.
- Some datasets, like UBFC_PHYS, provide individual description files for each participant alongside the raw data.
- Others, such as PhyMER, include demographic details in a single common metadata file or within their README documentation.

To perform demographic-based cross-dataset experiments, it’s recommended that you:
- Inspect each dataset closely to identify where demographic information is stored.
- Collaborate with dataset authors, if necessary, to clarify participant-level metadata.
- Group participants into demographic cohorts (e.g., male/female or age ranges).
- Merge the corresponding normalized feature CSVs to create cohort-level CSVs for your cross-dataset evaluations.


## 3. Running Experiments
### 3.1 Benchmarking Individual Datasets (2-class)
#### 3.1.1 Running Base experiments 

Larger datasets require 600-720 GPU hours each, and on average, other datasets require 24-30 hours each to run Model_running.py. 
We recommend creating a tmux instance for each dataset and running the script individually, following the given steps. 

- Activate your environment
```bash
source path/to/feel_env/bin/activate // if using venv environment
conda activate feel // if using conda 
```

- Start tmux
```bash
tmux new -s feel_dataset
```

- Running Script
```bash
python3 Benchmarking/Model_running.py \
  --eda_fet_path 
  --ppg_fet_path 
  --com_fet_path
  --eda_raw_path 
  --ppg_raw_path
  --out_path_benchmark Analysis/Dataset_Name/Benchmarking_results.csv
```

- Detach from tmux while it runs
```bash
Ctrl + B, then D
```

- Reattach later to check progress
```bash
tmux attach -t feel_datset
```

#### 3.1.2 Running CLSP Fine-Tuning experiments 
CLSP fine-tuning takes on average 20 minutes of the GPU hours. We recommend creating a tmux instance. 
To run the clsp script, follow these commands:

```bash
python3 Fine_tuning/finetuning_CLSP_two_class_CNN.py --dataset_name 
python3 Fine_tuning/finetuning_CLSP_two_class_MLP.py --dataset_name
```

### 3.2 Benchmarking Individual Datasets (four-class)
#### 3.2.1 Running Base experiments 

We recommend creating a tmux instance for each dataset and running the script individually, following this command. 
```bash
python3 Benchmarking/Model_running_fourclass.py \
  --eda_fet_path 
  --ppg_fet_path 
  --com_fet_path
  --out_path_benchmark Analysis/Dataset_Name/Benchmarking_results.csv
```

#### 3.2.2 Running CLSP FineTuning experiments 
We recommend creating ta mux instance  for each dataset and run the script individually following this command. 

```bash
python3 Fine_tuning/finetuning_CLSP_four_class_CNN.py --dataset_name 
python3 Fine_tuning/finetuning_CLSP_four_class_MLP.py --dataset_name
```
### 3.3 Cross-Dataset Evaluation
You can create multiple groupings based on your requirements. We have made three sets of grouping (Experimental Setting,  Device Type,  Labelling Method)
#### 3.3.1 Curating CSV files for each group 
Please refer to the Data_Preparation.ipynb that has steps for curating csv files for each group. 

#### 3.2.2 Running Experiments
Group_running.py file runs both base experiments and CLSP fine-tuning experiments.
We recommend creating a tmux instance  for each group and run the script individually following this command. 
```bash
python3 Benchmarking/Group_running.py \
  --eda_fet_path 
  --ppg_fet_path 
  --com_fet_path
  --out_path_benchmark Analysis/Dataset_Name/Benchmarking_results.csv
```
## 4. Extending the Framework
#### 4.1 Adding a new Model 
- To add a new model, write the new_model.py file and add it to the script folder.
- Add a code block in the Model_running.py, with the proper import statements.

#### 4.2 Adding a new dataset
- Complete the data curation step first to create the CSV files.
- Modify the grouping CSVs.
- Run the experiments.

## 5. Troubleshooting

### Common Issues
#### Issue 1: CUDA Out of Memory

Fixes:
- Reduce batch size
- Use gradient accumulation
- Use mixed precision training

#### Issue 2: Feature Extraction Fails

Fixes:
- Adjust window size

#### Issue 3: EDA Artifact Issues

Fixes:
- Most of the time out of memory issue comes for the large datasets; for that, run the eda_artifact.py file for that individual dataset on an individual GPU core.
