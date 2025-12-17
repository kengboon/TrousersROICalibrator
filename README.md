# TrousersROICalibrator

This repository presents a ROI calibration/alignment workflow for industrial inspection, involving deep learning-based keypoint detection.

The use case presented is for fashion industry, in particular for trousers.

## Keypoint Detection

A keypoint detection model is trained with a subset of [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset, which contains images and annotations with "trousers" category.

The definition of keypoints is as below:

<img width="164" height="242" alt="Trousers Landmarks (DeepFashion2)" src="https://github.com/user-attachments/assets/1ab19605-aed3-4f1e-973c-29ba2a260bd9" /> 
<img width="195" height="242" alt="Keypoint Detection on Jeans" src="https://github.com/user-attachments/assets/3600dea3-4db3-4c6a-8488-e820e390fb7a" />

__Model weight:__

<a href="[https://huggingface.co/kengboon](https://huggingface.co/kengboon/keypointrcnn-trousers)" target="_blank"><img src="https://img.shields.io/badge/Download%20Model%20Weight-EA9F00?style=for-the-badge&logoColor=white&logo=huggingface" height="40" alt="Download Model Weight"/></a>

## ROI Calibration

By computing the [barycentric coordinates](https://mathworld.wolfram.com/BarycentricCoordinates.html) of reference ROIs (bounding boxes) corresponding to the triangles drawn from the keypoints detected, ROIs can be auto aligned on target images.

<img width="890" height="447" alt="2025-12-17 10 24 56" src="https://github.com/user-attachments/assets/d5d70897-d7b3-4ae7-a53b-2c237950ea21" />


# Licenses

The open-source / permissive license applied to the __code only__, for demonstrating the keypoint detection results can be used for ROI calibration.

Refer [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) for its license (by this time, strictly for acedemic research only, not for commercial use).

You should use licensed or self-annotated dataset for any commercial usage.
