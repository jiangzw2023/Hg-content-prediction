# Hg Content Prediction Using Machine Learning

This repository contains code and data for a machine learning regression model used to predict mercury (Hg) content based on geochemical features. The approach supports geochemical proxy analysis and paleoenvironmental reconstruction in deep-time studies.

## This model was applied in the study:
Jiang ZW, Cai CF, Dou LR, et al. (2025). Volcanism-Driven Shift in the Mesoproterozoic Carbon Cycle and Oxygen Dynamics, under review.

## Contents：
### •	Hg_prediction_code.py：

This is the main script. It sets up input parameters, performs data preprocessing, runs the regression model (with hyperparameter tuning), and plots the results. Run this file to execute the entire workflow.

### •	Compiled_dataset.xlsx：
 
The dataset used in model training and evaluation. It includes Hg content and related elemental concentrations from both this study and published datasets.

## Features：

•	Implements multiple machine learning regression algorithms (e.g., XGBoost, Random Forest, SVR, Stacking).

•	Supports parameter tuning and cross-validation.

•	Outputs evaluation metrics and result visualizations (e.g., predicted vs. observed plots).

•	Designed to facilitate reproducibility and extension to other geochemical targets.

## Requirements：

•	Python 3.8+

•	scikit-learn

•	xgboost

•	pandas

•	numpy

•	matplotlib

•	seaborn
