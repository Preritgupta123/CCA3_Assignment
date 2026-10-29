# CCA 3 - Computer Vision Assignment

## 📌 Overview

This repository contains the complete implementation of the CCA3 practical assignment covering **Units 3, 4, and 5** of the Computer Vision course — Image Restoration, Image Segmentation, Feature Extraction, Object Recognition, and Motion Estimation.

**Author:** Prerit Gupta
**Roll No:** 1272250521
**Course:** MCA
**Programming Language:** Python

---

## 🎯 Objective

To implement and compare important computer vision techniques using real, self-captured data — including noise restoration, morphological segmentation, feature matching, object detection (classical vs deep learning), and motion estimation in video.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| OpenCV | Image processing, filtering, morphology, SIFT, Haar Cascade, optical flow |
| NumPy | Array operations |
| Matplotlib | Visualization and plotting |
| scikit-image | HOG features, PSNR/SSIM metrics |
| Ultralytics YOLO | Deep learning-based object detection |

---

## 📂 Folder Structure
CCA3_ComputerVision/
│
├── data/
│ ├── cube_photo1.jpg # Primary photo (front angle)
│ ├── cube_photo2.jpg # Second photo (tilted angle, for SIFT)
│ ├── haarcascade_frontalface_default.xml # Haar cascade model file
│ └── original_video.mp4 # Self-recorded video (see Drive link below)
│
├── outputs/
│ ├── task1_restoration/ # Noisy, restored, binary, morphology, canny, watershed results
│ ├── task2_features/ # HOG visualization, SIFT keypoints & matches
│ ├── task3_detection/ # Haar Cascade & YOLO detection results
│ └── task4_motion/ # Frame differencing, GMM mask, optical flow results
│
├── src/
│ ├── task1_restoration_segmentation.py
│ ├── task2_feature_matching.py
│ ├── task3_object_recognition.py
│ └── task4_motion_estimation.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── CCA3_Technical_Report.pdf


---

## 🧪 Tasks Implemented

### Task 1: Image Restoration & Segmentation (Unit 3)
- Added Gaussian noise and Salt-and-Pepper noise to the cube photograph
- Restored using Gaussian filtering and Median filtering
- Evaluated restoration quality using **PSNR** and **SSIM**
- Applied Otsu's thresholding for binarization
- Performed morphological operations: erosion, dilation, opening, closing
- Applied Canny edge detection
- Performed Watershed segmentation

### Task 2: Feature Extraction & Image Matching (Unit 4)
- Extracted and visualized **HOG** (Histogram of Oriented Gradients) features
- Detected **SIFT** keypoints and descriptors on two cube photographs
- Matched SIFT features using BFMatcher with Lowe's ratio test
- Repeated matching after rotating (45°) and resizing (50%) one image
- Compared match quality across all conditions

### Task 3: Object Recognition (Unit 5)
- Applied a pretrained **Haar Cascade** classifier for face detection on a video frame
- Applied a pretrained **YOLOv8** model for general object detection
- Tested YOLO on both the cube photo (out-of-distribution object) and the video frame
- Identified and documented real failure cases:
  - Haar Cascade false positive (detected sunlit leaves as a "face")
  - YOLO misclassification (cube detected as "bus", "person", "stop sign")

### Task 4: Video Motion Estimation (Unit 5)
- Extracted consecutive frames from the original video
- Applied **Frame Differencing** to detect moving regions
- Applied **GMM (Gaussian Mixture Model)** background subtraction
- Applied **Lucas-Kanade Optical Flow** to estimate motion vectors
- Compared results across all three motion estimation techniques

---
