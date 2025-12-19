# TrousersROICalibrator

This repository presents a ROI calibration/alignment workflow for industrial inspection, involving deep learning-based keypoint detection.

The use case presented is for fashion industry, in particular for trousers.

## Keypoint Detection

A keypoint detection model is trained with a subset of [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset, which contains images and annotations with "trousers" category.

The definition of keypoints is as below:

<img width="164" height="242" alt="Trousers Landmarks (DeepFashion2)" src="https://github.com/user-attachments/assets/1ab19605-aed3-4f1e-973c-29ba2a260bd9" /> 
<img width="195" height="242" alt="Keypoint Detection on Jeans" src="https://github.com/user-attachments/assets/3600dea3-4db3-4c6a-8488-e820e390fb7a" />

__Model weight:__

<a href="https://huggingface.co/kengboon/keypointrcnn-trousers" target="_blank"><img src="https://img.shields.io/badge/Download%20Model%20Weight-EA9F00?style=for-the-badge&logoColor=white&logo=huggingface" height="40" alt="Download Model Weight"/></a>

## ROI Calibration

By computing the [barycentric coordinates](https://mathworld.wolfram.com/BarycentricCoordinates.html) of reference ROIs (bounding boxes) corresponding to the triangles drawn from the keypoints detected, ROIs can be auto aligned and adjusted on target images.

__Robust:__ The position, size, and ratio of ROIs are adjusted accordingly based on keypoints.

<img width="814" height="482" alt="Sample ROIs Prediction" src="https://github.com/user-attachments/assets/0150a189-befa-483e-9191-fea78140680a" />

# Licenses

The open-source / permissive license applied to the __code only__, for demonstrating the keypoint detection results can be used for ROI calibration.

Refer [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) for its license (by this time, strictly for acedemic research only, not for commercial use).

You should use licensed or self-annotated dataset for any commercial usage.
