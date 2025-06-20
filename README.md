# UNL-QA

This project is the source code of a paper published at the IEEE ISBI 2025 conference, titled: UNSUPERVISED NORMATIVE LEARNING FOR  QUALITY ASSESSMENT ON 
DIFFUSION MRI.

## Introduction

UNL-QA is an unsupervised quality assessment tool for **diffusion MRI (dMRI)** that automatically detects artifacts like ghosting, spikes, noise, and swaps **without** requiring labeled training data. Leveraging a Vector Quantized Variational Autoencoder (**VQ-VAE**), it learns normative patterns from artifact-free dMRI scans. By quantifying Structural Similarity Index Metric (**SSIM**) differences between original and reconstructed images, UNL-QA identifies abnormalities with high precision. Validated across multiple datasets with varying acquisition parameters, this approach significantly reduces manual inspection burdens while maintaining clinical-grade accuracy. 


## Key Features
> - **Unsupervised learning**: Requires only artifact-free dMRI data - no manual artifact labeling needed
> - **Multi-artifact detection**: Identifies ghosting, spikes, noise, and swap artifacts
> - **Cross-protocol robustness**: Validated on SUDMEX, BTC, dHCP datasets with different acquisition parameters
> - **Interpretable results**: Quantifies artifact probability using SSIM divergence metrics

## Project Structure
```
|—— dataset.py                 # Dataloader for training, validation, and testing, including log normalization
|—— model.py                   # Model architecture for VQ-VAE
|—— train.py                   # Training VQ-VAE code to learn normative patterns
|—— functions.py               # Incluing the external components for model of VQ-VAE that are imported from model.py
|—— utils1.py                  # Visualization of the classification results
|—— utils3.py                  # Visualization and calculating the possibility of classification
|—— test.py                    # Use the finishing trained model to execute the quality assessment
|—— requirements.txt           # Requirements that are needed to run the project
```

## Usage
To use this project, follow these steps:
1. Install the required packages by running `pip install -r requirements.txt`.
2. Then you can run `python train.py` to train VQ-VAE the model only using artifact-free dMRI data, after finishing, you need to choose the best model by validation.
3. Finally, you can run `python test.py` to classify the quality of the volume exclude b0 from DWI images


## Data Requirements
```Input Structure
dataset/
├── subject_01/
│   ├── dwi.nii.gz       # Diffusion-weighted images
│   ├── bval             # b-values
│   └── bvec             # b-vectors
|   └── dwi_mask.nii.gz  # Mask image
├── subject_02/
└── ...
```

## Citation
if you find this project useful, please consider citing this paper:
```
@inproceedings{yu2025unsupervised,
  title={Unsupervised Normative Learning for Quality Assessment on Diffusion MRI},
  author={Yu, Jiahao and Wang, Tenglong and He, Yifei and Pan, Yiang and He, Jianzhong and Wu, Ye},
  booktitle={2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  pages={1--4},
  year={2025},
  organization={IEEE}
}
```
