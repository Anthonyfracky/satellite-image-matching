# Satellite Image Matching with

This project implements an efficient pipeline for matching large satellite images. By leveraging keypoint detection techniques such as **SIFT** or **ORB**, the system identifies and compares key features in overlapping image regions, enabling accurate matching of similar areas.

## Description

1.  **Images folder**:
    
    - The `images` folder contains image pairs for testing the matching algorithm
2. **Image Processing**:
    
    - The script `image_processing.py` splits large satellite images into smaller tiles, detects keypoints using **SIFT** or **ORB**, and matches keypoints between two images for efficient image analysis and feature matching
3. **Demo Script**:
    
    - The script `image_matching_demo.py` demonstrates the algorithm using example images. It processes the images, detects and matches keypoints, and visualizes the results.
5. **Interactive Notebook**:
    
    - `ner_demo.ipynb` provides an interactive demonstration of code usage, showcasing how the system works with satellite images
6. **Environment Setup**:
    
    - `requirements.txt` lists all necessary Python dependencies to set up the project environment.

## Installation

### Clone the Repository 
```bash 
git clone https://github.com/Anthonyfracky/satellite-image-matching.git
cd satellite-image-matching
```

### Setting up the Environment

Install all necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the demo and test the image matching:

```bash
image_matching_demo.py
```

This script will generate a matched image called `Notre_Dame_matched_image.jpg`. If you'd like to test with your own images, change the image paths in the script:
```python
image1_path = 'images/NDdP_1.png'
image2_path = 'images/NDdP_2.png'
```