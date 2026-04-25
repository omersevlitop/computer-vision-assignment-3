# Computer Vision Assignment 3

## Camera Calibration, Stereo Vision, Optical Flow and Classical Segmentation

This repository contains the implementation of classical computer vision techniques developed for Assignment 3. The project focuses on camera modeling, stereo geometry, depth estimation, motion analysis, and image segmentation using traditional image processing and geometric methods.

---

## Objectives

- Perform camera calibration using checkerboard images
- Estimate intrinsic matrix and lens distortion parameters
- Analyze epipolar geometry between stereo image pairs
- Apply stereo rectification
- Generate disparity maps for depth estimation
- Estimate dense motion using optical flow
- Segment images using classical non-deep-learning methods
- Compare methods quantitatively and qualitatively

---

## Project Structure

computer-vision-assignment-3/

├── Dataset/  
│   ├── calibration/  
│   ├── stereo/  
│   └── optical_flow/  

├── src/  
│   ├── Task1_CameraCalibration.py  
│   ├── Task2_EpipolarGeometry.py  
│   ├── Task3_DisparityDepth.py  
│   ├── Task4_OpticalFlow.py  
│   └── Task5_ClassicalSegmentation.py  

└── README.md

---

## Tasks

### Task 1 — Camera Calibration

- Checkerboard corner detection
- Intrinsic parameter estimation
- Distortion coefficient estimation
- Reprojection error analysis
- Undistortion result generation

### Task 2 — Epipolar Geometry and Stereo Rectification

- SIFT feature detection
- Descriptor matching
- Fundamental matrix estimation using RANSAC
- Epipolar line visualization
- Stereo rectification

### Task 3 — Disparity and Depth Estimation

- StereoBM disparity estimation
- StereoSGBM disparity estimation
- Parameter comparison
- Depth interpretation from disparity maps

### Task 4 — Optical Flow

- Dense motion estimation using Farneback method
- Motion vector visualization
- Color-coded flow analysis
- Pedestrian movement interpretation

### Task 5 — Classical Segmentation

- Otsu thresholding
- K-means color segmentation
- Multi-image qualitative comparison

---

## Methods Used

- OpenCV
- NumPy
- Matplotlib
- SIFT Feature Detection
- RANSAC
- StereoBM
- StereoSGBM
- Farneback Optical Flow
- Otsu Thresholding
- K-means Clustering

---

## Results

- Accurate camera calibration with low reprojection error
- Robust stereo correspondence estimation
- Successful disparity-based depth recovery
- Reliable pedestrian motion estimation
- Effective classical segmentation on multiple scenes

---

## Requirements

Install dependencies:

pip install opencv-python numpy matplotlib

---

## How to Run

Example:

python src/Task1_CameraCalibration.py  
python src/Task2_EpipolarGeometry.py  
python src/Task3_DisparityDepth.py  
python src/Task4_OpticalFlow.py  
python src/Task5_ClassicalSegmentation.py

---

## Notes

- Output files can be reproduced by running the scripts.
- Public benchmark datasets and custom video samples were used.
- Folder paths may be adjusted depending on local environment.

---

## AI Usage Disclosure

AI tools were used for:

- code structuring support
- debugging assistance
- explanation of concepts
- report writing support

All final implementations were reviewed, tested, and fully understood by the author.

---

## Author

Ömer Sevli

Computer Vision Course – Assignment 3
