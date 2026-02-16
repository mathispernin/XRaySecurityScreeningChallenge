# 🔍 X-Ray Security Screening Challenge

Classical Computer Vision pipeline for detecting prohibited objects in X-ray baggage images using handcrafted features and XGBoost classification.  
This project was developed for the Kaggle competition **"Object Detection in Baggage using Classic Computer Vision"**, where deep learning is not allowed.

## 📌 Overview

This project implements a complete object detection system using traditional computer vision techniques.

### 🎯 Objectives

Detect 6 classes of dangerous objects in X-ray images:

- 🔨 Hammer  
- 🔪 Knife  
- 🔫 Gun  
- 🔧 Wrench  
- ⛓ Handcuffs  
- 💣 Bullet  

The system uses:

- Image preprocessing and enhancement
- Selective Search for region proposals
- Handcrafted feature extraction
- XGBoost multi-class classification
- Hard Negative Mining for improved accuracy

## 🧠 Pipeline Architecture

### 1. Preprocessing
- Grayscale conversion
- CLAHE contrast enhancement
- Bilateral filtering
- HSV conversion (for color features)

### 2. Region Proposal
- Selective Search
- Non-Maximum Suppression (NMS)

### 3. Feature Extraction

Combination of complementary handcrafted features:

- **HOG** – shape features  
- **LBP** – texture features  
- **GLCM** – texture statistics  
- **HSV Color Histograms** – material/color information  
- **SIFT Bag of Visual Words** - texture/shape representation

### 4. Classification

- **XGBoost multi-class classifier**
- PCA dimensionality reduction (HOG)
- Hard negative mining for iterative improvement

### 5. Post-processing

- Global Non-Maximum Suppression

## 📂 Dataset

The dataset consists of X-ray .png images of baggage with annotated bounding boxes for 6 classes of dangerous objects.
It can be downloaded from the Kaggle competition page (see below).

xray_data/
│── data.yaml
│── images/
│ ├── train/
│ ├── val/
│── labels/
│ ├── train/
│ ├── val/
│── test/

## 📊 Dataset Statistics

| Split | Images | Objects |
|------|--------|---------|
| Train | 4200 | 6215 |
| Validation | 900 | 1320 |
| Test | 900 | 1321 |

### Label format (YOLO format)

class_id x_center y_center width height

Example: 2 0.903688 0.588403 0.033612 0.134967

Coordinates are normalized between 0 and 1.

## 📏 Evaluation Metric

The competition uses **mean IoU with greedy matching and prediction penalty**.

For each image:

Frame Score = sum(IoU matches) / max(num_GT, num_predictions)

Final score = average over all frames.

Key properties:

- Perfect prediction → score = 1.0  
- Missing predictions → penalty  
- Extra predictions → penalty  
- Wrong class → IoU = 0  

## 📤 Submission Format

CSV format:

frame_id,bbs
PID_xray_00637,"[[0,0.5,0.5,0.2,0.2],[1,0.3,0.3,0.1,0.1]]"
PID_xray_00638,"[[2,0.4,0.6,0.15,0.15]]"
PID_xray_00663,"[]"

Bounding box format: [class_id, center_x, center_y, width, height]

## 🏆 Competition

Kaggle Competition: VIC: Object Detection in Baggage using Classic Computer Vision

Stelios Perrakis. VIC: Object Detection in Baggage using Classic. https://kaggle.com/competitions/vic-vision-par-ordinateur-object-detection-in-baggage-using-classic, 2026. Kaggle.
