# CS339N Final Project - Head Motion as a Disease Biomarker

By Karan Singh and Kevin Liu

## Overview

This project explores the use of head motion, specifically framewise displacement, as a potential biomarker for Parkinson's disease (PD). We focus on analyzing the framewise displacement derived from rsfMRI data to extract meaningful insights about patient movement during scans.

A more comprehensive description of this project (including results) can be found in the paper (`Paper.pdf`).

## fMRI Preprocessing

The raw fMRI BOLD signals undergo several preprocessing steps to ensure the data is clean and suitable for analysis. The major preprocessing steps include:

* Motion Correction: Aligns the fMRI images to correct for head movement during the scan.
* Slice Timing Correction: Adjusts for differences in acquisition time between slices.
* Spatial Normalization: Transforms the images to a standard space to allow for group analysis.
* Smoothing: Applies a Gaussian filter to reduce noise and improve signal-to-noise ratio.
* Confound Regression: Removes unwanted variability from the data, such as physiological noise and motion artifacts.
 
One of the key outputs of this preprocessing is the generation of timeseries "confounds," which include various metrics that describe the quality and characteristics of the fMRI data. Among these confounds is framewise displacement, a measure of head motion between consecutive frames.

## Data

In `src/data_processing.py`, we extract the framewise displacement data along with the metadata of each patient. This data is stored in the `framewise_displacement_data.npz` file. The extracted data is first loaded and converted into a dictionary format. This dictionary, `framewise_displacement_data`, is structured as follows:

* Keys: Each key is a combination of the subject ID and run ID, formatted as sub-<subject_id>_run-<run_id>.
* Values: Each value is a dictionary containing:
  * framewise_displacement: A list of framewise displacement values for the corresponding run. I believe the first element of each one is NaN--will fix this later and resave the file, though dropping the first element is a simple temporary fix.
  * dvars: spatial framewise standard deviation
  * rmsd: root mean square displacement between successive images
  * participant_info: A dictionary with metadata about the participant, including group, sex, and age.

We also introduced a second dataset of controls (HCP) and added it to address class imbalance in the original dataset. We also employed quantile matching to ensure consistency among controls (see `src/data_processing.py`). We ultimately used only the `rmsd` time series for our modeling.

## Modeling

We explore two general classes of models: Classical (i.e., HMM-based) and deep learning.

In `notebooks/modeling_stats.ipynb`, we train an unsupervised Gaussian HMM to learn hidden states associated with the RMSD series for control and PD subjects. We also train a supervised version where the proportions of time spent in each hidden state are used as features in a logistic regression model.

In `notebooks/modeling_nn.ipynb`, we train deep learning models (GRU, LSTM, and Transformer) for predicting disease status (PD or control) based on the RMSD series.
 
## Relevant Prior Work

1. https://link.springer.com/article/10.1007/s10548-014-0358-6. Closest but older and small sample size with no biomarkers and averaged parameters
2. https://www.sciencedirect.com/science/article/pii/S2666956022000095#bib4
3. https://www.sciencedirect.com/science/article/pii/S1053811919310249
4. https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.70143. Only large-scale attempt, but again the analyses are quite limited
