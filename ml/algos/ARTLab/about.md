# LAB ART 

## Datasets :

## ADNI_IMAGES for Alzheimer Detection
ADNI MRI Brain Scans for Alzheimer's Detection (PNG Slices)
Overview
This dataset provides a collection of preprocessed MRI brain scan images from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) project, specifically formatted as PNG slices. It is designed to facilitate the development and evaluation of deep learning and machine learning models for Alzheimer's disease detection and analysis.

Data Collection and Preprocessing
Source: Alzheimer’s Disease Neuroimaging Initiative (ADNI) (https://adni.loni.usc.edu/)
Data Type: MRI Brain Scans
MRI Sequences: Primarily T1-weighted images, including Accelerated Sagittal MPRAGE and Sagittal 3D FLAIR.
Acquisition Planes: Primarily Axial slices.
Preprocessing Pipeline:
Original DICOM files were converted to NIfTI format using [mention library, if used].
NIfTI files were then sliced along the axial plane and converted to PNG images.
Only slices within the range of 50 to 170 were retained, focusing on the brain's central region. This selection aims to concentrate on areas most relevant to Alzheimer's-related changes.
Note: The preprocessing steps were performed using the dcm_to_png_axial.py script.
Dataset Structure
The dataset is organized into folders corresponding to five distinct classes:

AD (Alzheimer’s Disease): 4075 images
CN (Cognitively Normal): 4077 images
EMCI (Early Mild Cognitive Impairment): 3958 images
LMCI (Late Mild Cognitive Impairment): 4074 images
MCI (Mild Cognitive Impairment): 4073 images
Each folder contains PNG images representing axial slices of the MRI scans. Additionally, a train.csv file provides metadata associated with each image, including patient demographics and clinical information.

Usage
This dataset is particularly useful for:

Developing and training deep learning models (e.g., Convolutional Neural Networks) for Alzheimer's disease detection and classification.
Investigating MRI-based biomarkers for cognitive decline analysis.
Experimenting with various CNN architectures and transfer learning techniques for medical image analysis.
Image segmentation of brain regions from the PNG slices.
Acknowledgements
This dataset is derived from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (http://adni.loni.usc.edu/). As such, the investigators within the ADNI have contributed to the collection of this data and made it available.

Citation
When utilizing this dataset in your research or projects, please acknowledge the ADNI database as the original source and cite this Kaggle dataset.

## Attacks 

### Fast Gradient Sign Attack Implementation
Adding small perturbations to an image can cause a big change in model performance. To test security vulnerabilities of ML models an important tool is adversarial machine learning. The Fast Gradient Sign Attack (FGSM) is a type of adversarial attack.
It's a white box attack (i.e. the attacker knows the model, the parameters and the architecture).
It is designed to attack neural networks using gradients. The attack uses the gradient of the loss w.r.t the input data, then adjusts (perturb) the input data to maximize the loss.

### Projected Gradient Descent Attack Implementation
PGD is a variant of FGSM applied iteratively with projection. It is a commonly used attack for adversarial training: training the network to be robust.

https://owasp.org/www-project-machine-learning-security-top-10/
https://ieeexplore.ieee.org/document/9064510