import numpy as np
import cv2 as cv

class DenseOpticalFlow:
    """http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf"""

    def __init__(self, input_video_path):
        # Initialize the object with the input video path
        self.input_video = input_video_path

    def detect(self):
        # Open the video capture object for the input video
        cap = cv.VideoCapture(self.input_video)
        ret, frame1 = cap.read()
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        
        # Create an HSV image with the same dimensions as the frame
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        
        while True:
            ret, frame2 = cap.read()
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            
            # Calculate the dense optical flow using Farneback method
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Convert the flow field to polar coordinates (magnitude and angle)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Set the hue channel of the HSV image to the flow angle in degrees
            hsv[..., 0] = ang * 180 / np.pi / 2
            
            # Set the value channel of the HSV image to the normalized flow magnitude
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            
            # Convert the HSV image back to BGR color space
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            
            # Display the optical flow visualization
            cv.imshow('frame2', bgr)
            k = cv.waitKey(30) & 0xff
            
            if k == 27:  # Press 'Esc' to break the loop
                break
            elif k == ord('s'):
                # Save the frame and the optical flow visualization as images
                cv.imwrite('opticalfb.png', frame2)
                cv.imwrite('opticalhsv.png', bgr)
            
            # Update the previous frame for the next iteration
            prvs = next
