import numpy as np
import cv2 as cv

class ColorQuantization:
    def __init__(self, image_path):
        # Initialize the object with the path to the input image
        self.image_path = image_path

    def quantize(self):
        # Read the input image
        img = cv.imread(self.image_path)

        # Reshape the image to have all pixels in a single row
        Z = img.reshape((-1, 3))

        # Convert the data type to np.float32
        Z = np.float32(Z)

        # Define criteria for the k-means clustering algorithm
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Define the number of clusters (K) for k-means
        K = 2

        # Apply the k-means clustering algorithm
        ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        # Convert the cluster centers back to uint8 data type
        center = np.uint8(center)

        # Reshape the clustered data to match the original image shape
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        # Display the color-quantized image
        cv.imshow('res2', res2)
        cv.waitKey(0)
        cv.destroyAllWindows()
