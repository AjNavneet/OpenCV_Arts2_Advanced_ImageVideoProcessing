# OpenCV Arts - Advanced Techniques

## Business Objective

OpenCV (Open-Source Computer Vision Library: [http://opencv.org](http://opencv.org)) is an open-source library that includes several hundreds of computer vision algorithms. OpenCV has a modular structure, which means that the package includes several shared or static libraries.

OpenCV is a huge open-source library for computer vision, machine learning, and image processing. It can process images and videos to identify objects, faces, or even the handwriting of a human. When integrated with various libraries, such as "NumPy," a highly optimized library for numerical operations, the number of weapons increases in your Arsenal, i.e., whatever operations one can do in NumPy can be combined with OpenCV.

---

## Data Description

In this project, we will be using sample images and videos as our input data and perform various algorithms on top of it.

---

## Aim

The project aims at performing some complex techniques and algorithms using the OpenCV library.

---

## Tech Stack

- Language: `Python`
- Libraries: `numpy`, `matplotlib`, `cv2 (OpenCV)`

---

## Approach

1. Importing the required libraries.
2. Implement Background subtraction.
3. Perform the Meanshift algorithm.
4. Perform the Camshift algorithm.
5. Implement the Lucas Kanade Optical Flow algorithm.
6. Implement the Franeback Dense Flow algorithm.
7. Perform High Dynamic Range (HDR) imaging.
8. Implement Epipolar Geometry using SIFT and Stereo images.
9. Implement Depth Map on Stereo images.
10. Perform Color Quantization using Clustering.
11. Perform Image De-noising.

---

## Modular Code Overview

### 1. Input folder

It contains all the data that we have for analysis. Here we have a few sample images and a video for performing different operations using OpenCV.

### 2. Source folder

This is the most important folder of the project. This folder contains all the modularized code for all the above steps in a modularized manner. This folder consists of:

- `Engine.py`
- `ML_Pipeline`

The `ML_Pipeline` is a folder that contains all the functions put into different Python files, which are appropriately named. These Python functions are then called inside the `engine.py` file.

### 3. Output folder

The output folder contains output images generated after running all the functions created.

---

---

## Concepts Explored

1. Understand OpenCV
2. Background Subtraction.
3. Meanshift Algorithm
4. Camshift Algorithm
5. Understand Optical Flow
6. Lucas Kanade Optical Flow Algorithm.
7. Franeback Dense Optical Flow Algorithm.
8. High Dynamic imaging process (HDR)
9. Basics of Epipolar Geometry
10. Epipolar Geometry using SIFT.
11. Depth Map on Stereo images
12. Image de-noising.
13. Understand Clustering
14. Understand Kmeans Algorithm
15. Color Quantization.

---
