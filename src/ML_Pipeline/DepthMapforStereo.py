import cv2 as cv
from matplotlib import pyplot as plt

class DepthMapStereo:
    def __init__(self, left_image, right_image):
        # Initialize the object with paths to the left and right stereo images
        self.left_image = left_image
        self.right_image = right_image

    def depth(self):
        # Read both stereo images in grayscale format
        imgL = cv.imread(self.left_image, 0)
        imgR = cv.imread(self.right_image, 0)

        # Create a StereoBM object with specified parameters
        stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)

        # Compute the depth (disparity map) using StereoBM
        disparity = stereo.compute(imgL, imgR)

        # Display the disparity map using matplotlib
        plt.imshow(disparity, 'gray')
        plt.show()
