import cv2 as cv
import numpy as np
import os
from .admin import output_folder, input_folder

class HDRImaging:
    def __init__(self):
        # Initialize the object with the path to the input folder
        self.path = os.path.join(input_folder, "HDRImagesInput")

    def loadExposureSeq(self, path):
        # Load input images and their exposure times from a user-defined folder
        images = []
        times = []
        with open(os.path.join(path, 'list.txt')) as f:
            content = f.readlines()
        for line in content:
            tokens = line.split()
            images.append(cv.imread(os.path.join(path, tokens[0]))
            times.append(1 / float(tokens[1]))
        return images, np.asarray(times, dtype=np.float32)

    def convert(self):
        # Load input images and exposure times
        images, times = self.loadExposureSeq(self.path)
        
        # Estimate the camera response function (CRF) for HDR construction
        calibrate = cv.createCalibrateDebevec()
        response = calibrate.process(images, times)
        
        # Construct the HDR image using the estimated response
        merge_debevec = cv.createMergeDebevec()
        hdr = merge_debevec.process(images, times, response)

        # Apply tonemapping to map the HDR image to an 8-bit range for display
        tonemap = cv.createTonemap(2.2)
        ldr = tonemap.process(hdr)

        # Merge the images using a different method for comparison
        merge_mertens = cv.createMergeMertens()
        fusion = merge_mertens.process(images)

        # Save the results as images
        cv.imwrite(os.path.join(output_folder, 'fusion.png'), fusion * 255)
        cv.imwrite(os.path.join(output_folder, 'ldr.png'), ldr * 255)
        cv.imwrite(os.path.join(output_folder, 'hdr.hdr'), hdr)
