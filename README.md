# TrousersROICalibrator

This repository presents a ROI calibration workflow for industrial inspection, involving deep learning-based keypoint detection.

The use case presented is for fashion industry, in particular for trousers.

## Keypoint Detection

A keypoint detection model is trained with a subset of [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset, filtering the images without trousers category.

The definition of keypoints is as below:

<img width="164" height="242" alt="Trousers Landmarks (DeepFashion2)" src="https://github.com/user-attachments/assets/1ab19605-aed3-4f1e-973c-29ba2a260bd9" />

## ROI Calibration



# Licenses

The open-source / permissive license applied to the __code only__, for demonstrating the keypoint detection results can be used for ROI calibration.

Refer [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) for its license (by this time, strictly for acedemic research only, not for commercial use).

You should use licensed or self-annotated dataset for any commercial usage.