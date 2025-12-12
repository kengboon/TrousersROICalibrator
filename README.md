# TrousersROICalibrator

This repository presents a ROI calibration workflow for industrial inspection, involving deep learning-based keypoint detection.

The use case presented is for fashion industry, in particular for trousers.

## Keypoint Detection

A keypoint detection model is trained with a subset of [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset, filtering the images without trousers category.

The definition of keypoints is as below:

<img width="164" height="242" alt="Trousers Landmarks (DeepFashion2)" src="https://github.com/user-attachments/assets/1ab19605-aed3-4f1e-973c-29ba2a260bd9" />
