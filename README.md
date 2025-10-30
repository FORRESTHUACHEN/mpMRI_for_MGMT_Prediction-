# mpMRI-based MGMT Methylation Status Prediction for Glioblastoma through Off-the-shelf Deep Features: A Multi‐Dataset Feasibility Study
Here is the repository of 'mpMRI-based MGMT Methylation Status Prediction for Glioblastoma through Off-the-shelf Deep Features: A Multi‐Dataset Feasibility Study'

Pipeline of study is shown as follow:
![Pipeline of study](https://github.com/FORRESTHUACHEN/mpMRI_for_MGMT_Prediction-/blob/main/Figure1.png)

Extracted deep features summaried in [DeepFeaturesAndTFRecords/GBM_kaggle_deep_feature_transfor_detal_summary.xlsx](https://github.com/FORRESTHUACHEN/mpMRI_for_MGMT_Prediction-/blob/main/DeepFeaturesAndTFRecords/GBM_kaggle_deep_feature_transfor_detal_summary.xlsx)

You can use script [codes/tfmaker](https://github.com/FORRESTHUACHEN/mpMRI_for_MGMT_Prediction-/blob/main/codes/tfmaker%20-%20v3.py) to create tfrecords for model training and validation and example tfrecord are shown as [DeepFeaturesAndTFRecords/Traintransdetalrunning1_0.tfrecords and DeepFeaturesAndTFRecords/Testtransdetalrunning1_0.tfrecords](https://github.com/FORRESTHUACHEN/mpMRI_for_MGMT_Prediction-/tree/main/DeepFeaturesAndTFRecords).
