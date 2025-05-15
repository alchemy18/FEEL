# FEEL: Quantifying Heterogeneity in Physiological Signals for Generalizable Emotion Recognition

> Under peer review at NeurIPS 2025

This is the code repository for the paper **"FEEL: Quantifying Heterogeneity in Physiological Signals for Generalizable Emotion Recognition"**, which introduces **FEEL** — the first large-scale benchmarking framework for emotion recognition using physiological signals across heterogeneous datasets.

## 🔍 Overview

Emotion recognition from physiological signals like **EDA (Electrodermal Activity)** and **PPG (Photoplethysmography)** is gaining traction due to its potential in health and affective computing applications. However, generalizing models across diverse real-world datasets remains a major challenge.

**FEEL** addresses this gap by:

- Curating and harmonizing **19 public emotion datasets** from lab, real-life, and constrained settings
- Benchmarking **16 models** across four modeling paradigms:
  - Traditional Machine Learning (Random Forest, LDA)
  - Deep Learning with handcrafted features (MLP, RESNET, LSTM+MLP, Attention Layer + MLP)
  - Deep Learning on raw signals (Resnet, LSTM+MLP, CNN+ Transformer Encoder Block)
  - Pretrained models using **Contrastive Language-Signal Pretraining (CLSP)** and finetuning using 2 different ablation (MLP and 1D-CNN) of Meta Net 
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

- [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
- [NURSE](https://www.nature.com/articles/s41597-022-01361-y#:~:text=The%20dataset%20provides%20more%20than,validated%20stressful%20events%20by%20nurses.)
- [EMOGNITION](https://www.nature.com/articles/s41597-022-01262-0)
- [UBFC_PHYS](https://sites.google.com/view/ybenezeth/ubfc-phys)
- [VERBIO](https://hubbs.engr.tamu.edu/resources/verbio-dataset/)
- [PhyMER](https://ieeexplore.ieee.org/document/10265252)
- [EmoWear](https://www.nature.com/articles/s41597-024-03429-3)
- [MAUS](https://ieee-dataport.org/open-access/maus-dataset-mental-workload-assessment-n-back-task-using-wearable-sensor)
- [CLAS](https://ieee-dataport.org/open-access/database-cognitive-load-affect-and-stress-recognition)
- [CASE](https://www.nature.com/articles/s41597-019-0209-0)
- [Unobtrusive](https://www.nature.com/articles/s41597-024-03738-7)
- [CEAP-360VR](https://ieeexplore.ieee.org/document/9599346)
- [ScientISST MOVE](https://physionet.org/content/scientisst-move-biosignals/1.0.1/)
- [LAUREATE](https://dl.acm.org/doi/10.1145/3610892)
- [ForDigitStress](https://ieeexplore.ieee.org/document/10756706)
- [Dapper](https://www.nature.com/articles/s41597-021-00945-4)
- [ADARP](https://arxiv.org/abs/2206.14568)
- [MOCAS](https://polytechnic.purdue.edu/ahmrs/mocas-dataset)
- [Exercise](https://www.nature.com/articles/s41597-025-04845-9)

## Set-up

To get started with the FEEL benchmark framework, follow these steps to set up your environment.

1. Clone the Repository
2. pip install -r requirements.txt

## 📦 Repository Structure

├── Benchmarking/             # Model running file and compiling IPYNB

├── Fine_tuning/              # CLSP FineTuning Scripts

├── Scripts/                  # Benchmark, artificats, visualization scripts

├── Vizualize/                # IPYNB to visualize Cross Dataset represntations when group against one metric

├── group_finetuning/         # Cross Dataset grouping Benchmarking and adaptation scripts

├── README.md

└── requirements.txt

# License

This code is released under the MIT license. Please see the license file for details.
