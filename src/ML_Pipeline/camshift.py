import numpy as np
import cv2 as cv

class CamShift:

    def __init__(self, input_video_path):
        # Initialize the object with the input video path
        self.input_video = input_video_path

    def detect(self):
        # Open the video capture object for the input video
        cap = cv.VideoCapture(self.input_video)
        
        # Take the first frame of the video
        ret, frame = cap.read()
        
        # Setup the initial location of the tracking window
        x, y, w, h = 300, 200, 100, 50  # Hardcoded initial values
        track_window = (x, y, w, h)
        
        # Set up the region of interest (ROI) for tracking
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        
        # Create a mask for the ROI in HSV color space
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        
        # Calculate the histogram of the ROI
        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        
        # Setup the termination criteria for tracking
        term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        
        while True:
            ret, frame = cap.read()
            if ret:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                
                # Apply CamShift algorithm to get the new location of the tracked object
                ret, track_window = cv.CamShift(dst, track_window, term_crit)
                
                # Draw the tracked object on the image
                pts = cv.boxPoints(ret)
                pts = np.int0(pts)
                img2 = cv.polylines(frame, [pts], True, 255, 2)
                cv.imshow('img2', img2)
                
                k = cv.waitKey(30) & 0xff
                if k == 27:  # Press 'Esc' to break the loop
                    break
            else:
                break
