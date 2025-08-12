# ClinicalPET_SEG
This repository provides a complete pipeline built on nnU-Net for automated segmentation of tumor lesions from whole-body 2-[18F]-fluorodeoxyglucose (FDG) PET/CT scans, followed by extraction of key radiomics and metabolic parameters. It is designed for clinical research, multi-center studies, and large-scale imaging analysis.


## Automatic Lesion Segmentation
End-to-end lesion segmentation using nnU-Net without manual network architecture or hyperparameter tuning.

## Multi-Modal Input
Processes PET and CT two channels jointly to improve accuracy in tumor boundary delineation and metabolic activity detection.

## Radio-bio Marker Extraction
Automatically computes metabolic tumor volume (MTV), total lesion glycolysis (TLG), maximum standardized uptake value (SUVmax), and dissemination metrics (Dmax, SDmax), among others.

## Installation
1. Clone the repository


  git clone https://github.com/IMIT-BlackTech/ClinicalPET_SEG.git
  cd ClinicalPET_SEG
  

2. Install dependencies

  pip install -r requirements.txt


## Acknowledgements
We would like to acknowledge the contributions of nnU-Net.
