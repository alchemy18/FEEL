# FEEL: Quantifying Heterogeneity in Physiological Signals for Generalizable Emotion Recognition

> Accepted at NeurIPS 2025 [Citation Coming Soon]

This is the code repository for the paper **"FEEL: Quantifying Heterogeneity in Physiological Signals for Generalizable Emotion Recognition"**, which introduces **FEEL** — the first large-scale benchmarking framework for emotion recognition using physiological signals across heterogeneous datasets.

## 🔍 Overview

Emotion recognition from physiological signals like **EDA (Electrodermal Activity)** and **PPG (Photoplethysmography)** is gaining traction due to its potential in health and affective computing applications. However, comparing model performance across diverse real-world datasets remains a major challenge.

**FEEL** addresses this gap by:

- Curating and harmonizing **19 public emotion datasets** from lab, real-life, and constrained settings
- Benchmarking **16 models** across four modeling paradigms:
  - Traditional Machine Learning (Random Forest, LDA)
  - Deep Learning with handcrafted features (MLP, RESNET, LSTM+MLP, Attention Layer + MLP)
  - Deep Learning on raw signals (Resnet, LSTM+MLP, CNN+ Transformer Encoder Block)
  - Pretrained models using **Contrastive Language-Signal Pretraining (CLSP)** https://proceedings.neurips.cc/paper_files/paper/2024/hash/1cba8502063fab9df252a63968691768-Abstract-Datasets_and_Benchmarks_Track.html
  - Finetuning using 2 different ablation (MLP and 1D-CNN) of Meta Net
    
- Performing **cross-dataset generalization analysis** across:
  - Experimental setting (Lab, Constraint, Real)
  - Device type (Wearable, Lab Based Device, Custom Wearable)
  - Labeling strategy (Stimulus-Label, Self-report, Expert-Annotated)

---

## Key Features

- Unified preprocessing and feature extraction pipeline for EDA & PPG
- Comprehensive cross-domain evaluation using Leave-One-Subject-Out CV
- Performance analysis across 3 input types: EDA-only, PPG-only, EDA+PPG
- Few-shot and zero-shot adaptation with CLSP models
- Results on arousal, valence and four-quadrant classification.

---

## Key Findings

- CLSP-based models achieved **73/114** best results, demonstrating strong cross-dataset transfer.
- Models using **handcrafted features** consistently outperformed raw-signal DL models in noisy or low-resource settings.
- Models trained in **real-world settings** transferred well to lab and constraint domains.
- **Labeling method and device heterogeneity** were key factors influencing generalization.

---

## Datasets Overview

FEEL benchmarks the following 19 datasets (Appendix A.1 in the paper for details): 

- WESAD, NURSE, EMOGNITION, UBFC_PHYS, VERBIO, PhyMER, EmoWear, MAUS, CLAS, CASE, CEAP-360VR, Unobtrusive, ForDigitStress, Dapper, LAUREATE, ADARP, Exercise, MOCAS, ScientISST MOVE

For each dataset, we generated preprocessed EDA and PPG signals, extracted features for EDA, PPG, and EDA + PPG, defined task descriptions, and standardized arousal/valence labels. Due to the nature of these datasets, most of which are available only upon request, we are unable to share the processed or raw data files publicly. However, the list below includes links to the respective papers for each dataset, through which you can contact the authors to request access to the raw data. Additionally, we have shared the binning details and preprocessing code for these raw files in this GitHub repository and paper.

Below is the list of the 19 publicly available emotion recognition datasets used in the FEEL benchmark, along with their access links:

- [**WESAD**](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) 
- [**NURSE**](https://datadryad.org/dataset/doi:10.5061/dryad.5hqbzkh6f#citations) 
- [**EMOGNITION**](https://dataverse.harvard.edu/citation?persistentId=doi:10.7910/DVN/R9WAF4) 
- [**UBFC_PHYS**](https://sites.google.com/view/ybenezeth/ubfc-phys)
- [**PhyMER**](https://sites.google.com/view/phymer-dataset) 
- [**EmoWear**](https://zenodo.org/records/10407279)
- [**MAUS**](https://ieee-dataport.org/open-access/maus-dataset-mental-workload-assessment-n-back-task-using-wearable-sensor) 
- [**CLAS**](https://ieee-dataport.org/open-access/database-cognitive-load-affect-and-stress-recognition) 
- [**CASE**](https://gitlab.com/karan-shr/case_dataset) 
- [**Unobtrusive**](https://zenodo.org/records/10371068)
- [**CEAP-360VR**](https://github.com/cwi-dis/CEAP-360VR-Dataset) 
- [**ScientISST MOVE**](https://www.scientisst.com/projects/run-like-a-scientisst)
- [**LAUREATE**](https://pc.inf.usi.ch/studentproject/affect-and-learning-in-the-laureate-dataset/) 
- [**ForDigitStress**](https://hcai.eu/fordigitstress/) 
- [**Dapper**](https://synapse.org/Synapse:syn22418021) 
- [**ADARP**](https://zenodo.org/records/6640290)
- [**MOCAS**](https://zenodo.org/records/7023242)  
- [**VERBIO**](https://hubbs.engr.tamu.edu/resources/verbio-dataset/) 
- [**Exercise**](https://physionet.org/content/wearable-device-dataset/1.0.0/) 

## Set-up

To get started with the FEEL benchmark framework, follow these steps to set up your environment.

1. Clone the Repository
2. pip install -r requirements.txt

## 📦 Repository Structure

```bash
FFEL/
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
│   ├── Benchmarking_analysis.ipynb
│   ├── vizualize_UMAP.ipynb
├── Tutorial/
│   ├── Data_Preparation.ipynb #tutorial for cleaning, bining, pre-processing datasets
│   ├── README.md # tutorial for running and extending FEEL Benchmark
```

## 🔗 Project Continuation

➡️ This repository is a continuation of the previous phase of **[this project](https://github.com/alchemy18/EEVR/)**

➡️ For more details and updates, visit the **[project webpage](https://alchemy18.github.io/FEEL_Benchmark/)**

🤝 If you want to contribute to the FEEL with new dataset or new models, submit your results **[here](https://docs.google.com/forms/d/e/1FAIpQLSchhaTlXFliCb1fS2zCK7-66zWAExXn6RavqeLaH2nE8vKs8A/viewform)**


# License

This code is released under the MIT license. Please see the license file for details.
