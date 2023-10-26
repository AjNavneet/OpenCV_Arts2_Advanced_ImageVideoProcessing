from ML_Pipeline.BackGround_Substraction import BackGroundSubtraction
from ML_Pipeline.meanshift import MeanShift
from ML_Pipeline.camshift import CamShift
from ML_Pipeline.Lucal_Kanade_OpticalFlow import LucasKanadeOpticalFlow
from ML_Pipeline.DenseOpticalFlow import DenseOpticalFlow
from ML_Pipeline.HDRImages import HDRImaging
from ML_Pipeline.EpipolarGeometry import EpipolarGeometry
from ML_Pipeline.DepthMapforStereo import DepthMapStereo
from ML_Pipeline.ColorQuantization import ColorQuantization
from ML_Pipeline.ImageDenoising import ImageDenoising
from ML_Pipeline.admin import input_folder
import os

### 1. BackGround Subtraction ###
# Initialize the input video file path
input_video = os.path.join(input_folder,'videoplayback.mp4')
# Create an instance of the BackGroundSubtraction class
background_obj = BackGroundSubtraction(input_video)
# Perform background subtraction
background_obj.background_substract()

### 2. MeanShift ###
# Create an instance of the MeanShift class using the same input video
meanshift_obj = MeanShift(input_video)
# Apply MeanShift algorithm for object detection
meanshift_obj.detect()

### 3. CamShift ###
# Create an instance of the CamShift class using the same input video
camshift_obj = CamShift(input_video)
# Apply CamShift algorithm for object detection
camshift_obj.detect()

### 4. Lucas Kanade Optical Flow ###
# Create an instance of the LucasKanadeOpticalFlow class using the same input video
lucal_optical_flow_obj = LucasKanadeOpticalFlow(input_video)
# Detect optical flow using the Lucas-Kanade method
lucal_optical_flow_obj.detect()

### 5. Farneback Dense Optical Flow ###
# Create an instance of the DenseOpticalFlow class using the same input video
dense_optical_flow_obj = DenseOpticalFlow(input_video)
# Detect dense optical flow using the Farneback method
dense_optical_flow_obj.detect()

### 6. High Dynamic Range(HDR) Imaging ###
# Create an instance of the HDRImaging class
hdr_imaging = HDRImaging()
# Convert images to High Dynamic Range (HDR)
hdr_imaging.convert()

### 7. Epipolar Geometry ###
# Define paths for left and right stereo images
left_image = os.path.join(input_folder, "view0.png")
right_image = os.path.join(input_folder, "view2.png")
# Create an instance of the EpipolarGeometry class with stereo images
epipolar_geo_obj = EpipolarGeometry(left_image, right_image)
# Detect epipolar geometry and corresponding points
epipolar_geo_obj.detect()

### 8. Depth Map on Stereo Images ###
# Create an instance of the DepthMapStereo class with stereo images
depth_map_obj = DepthMapStereo(left_image, right_image)
# Generate depth map from stereo images
depth_map_obj.depth()

### 9. Color Quantization ###
# Define the path of the input image for color quantization
image_path = os.path.join(input_folder, "download.jpeg")
# Create an instance of the ColorQuantization class with the image
color_quantize_obj = ColorQuantization(image_path)
# Perform color quantization on the image
color_quantize_obj.quantize()

### 10. Image Denoising ###
# Create an instance of the ImageDenoising class with the left stereo image and the input video
image_denoising_object = ImageDenoising(left_image, input_video)
# Perform colored image denoising
image_denoising_object.denoisingcolored()
# Perform grayscale image denoising
image_denoising_object.denoising_grayscale()
# Perform multi-image denoising
image_denoising_object.denoising_multi()
