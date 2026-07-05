# SLN
The prediction of sentinel lymph node metastasis
Overview
This repository contains the core code for the manuscript:
Multimodal AI Enhanced by Tumor-node Spatial Chen-Dai Distance for De-escalation of Axillary Surgery in Breast Cancer

Environment Setup
See environment.yml
bash
# Create conda environment
conda env create -f environment.yml


Data Structure
../data_incohort or data_excohort{No.}/
├── MRMG
│   ├── Positive/case/nii_data
│   ├── Negative/case/nii_data
├── US/case/nii_data
└── output_data_{split_type}.pkl or output_excohort{No.}_data_{split_type}.pkl
└── CP.xlsx

Usage
Basic Commands
bash
# Train MAT full multi-modal model (5 images + 2 features)
python train_test.py --mode train --model MAT --dataset_type allimgclinical --gpu 0 --bs 4
# Internal test
python train_test.py --mode test --model MAT --gpu 0
# External test (excohort: 1-7)
python train_test.py --mode test_external --model MAT --excohort 1 --gpu 0

