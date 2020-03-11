import cv2
import numpy as np
import math
from networktables import NetworkTables
import logging
import time
import subprocess

width = 0
height = 0
lower_hsv =[40, 58, 94]
upper_hsv = [255,255,255]


def init_table(server="10.3.34.2"):
    # Initialize network tables
    logging.basicConfig(level=logging.DEBUG)
    NetworkTables.initialize(server)
    VISION_TABLE = NetworkTables.getTable('vision')
    return VISION_TABLE

def nothing(x):
    pass

def main(trackbar = True, showWindows = True, usingWindows = True, usingNetworkTables = False):
    # captures the first camera
    cap = cv2.VideoCapture(0)
    ret, cap_for_exposure = cap.read()
    if usingWindows:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE , -1)
    else:
        subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_absolute=10",shell=True)
    if usingNetworkTables:
        rio_address = '10.3.34.2'
        VISION_TABLE = init_table(rio_address)
    if trackbar:
        # creates a new window with a trackbar
        cv2.namedWindow('Trackbar')

        # creates "variables" (low_h,low_s,...) linked to the 'Trackbar' window, with a range of 0, 255
        cv2.createTrackbar('low_h', 'Trackbar', 0, 255, nothing)
        cv2.createTrackbar('low_s', 'Trackbar', 0, 255, nothing)
        cv2.createTrackbar('low_v', 'Trackbar', 0, 255, nothing)

        cv2.createTrackbar('up_h', 'Trackbar', 255, 255, nothing)
        cv2.createTrackbar('up_s', 'Trackbar', 255, 255, nothing)
        cv2.createTrackbar('up_v', 'Trackbar', 255, 255, nothing)

    # continuously run this code as long as we are capturing video
    while cap.isOpened():
        # captures the frame
        ret, frame = cap.read()
        width = frame.shape[1]
        height = frame.shape[0]
        # converts the frame to hsv color spectrum from a rgb
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if trackbar:
            # gets the trackbar position from the 'Trackbar' window, then sets that to those values to the variables
            low_h = cv2.getTrackbarPos('low_h', 'Trackbar')
            low_s = cv2.getTrackbarPos('low_s', 'Trackbar')
            low_v = cv2.getTrackbarPos('low_v', 'Trackbar')

            up_h = cv2.getTrackbarPos('up_h', 'Trackbar')
            up_s = cv2.getTrackbarPos('up_s', 'Trackbar')
            up_v = cv2.getTrackbarPos('up_v', 'Trackbar')

            lower_hsvRange = np.array([low_h, low_s, low_v])
            upper_hsvRange = np.array([up_h, up_s, up_v])
        else:
            lower_hsvRange = np.array(lower_hsv)
            upper_hsvRange = np.array(upper_hsv)

        # this asks: are the hsv values in the lower and upper ranges?
        # if the pixels in the frame are in the range, they show up to be white, and black if they are not
        # ie, white = in range, black = not in range
        hsv = cv2.inRange(hsv, lower_hsvRange, upper_hsvRange)

        # kernel for the morphological transformations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # morphological transformations are used to reduce noise
        hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)

        # blur the frame to reduce noise
        hsv = cv2.medianBlur(hsv, 5)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # gets the contours based on the blurred variable(the blurred + morphed frame image)
        contours, hierarchy = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        try:
            # get the maximum contour, or the MAIN(the big rectangle) contour of the object, because there are a lot of
            # contours that from noise this is in a try block because if the frame has no contours, the max() function
            # will return an error
            c = max(contours, key=cv2.contourArea)
        except Exception as e:
            print(e)
            pass
        else:
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroid = (cx,cy)
            center_of_window = (int(width/2),int(height/2))
            offset = cx - (width / 2)
            print(offset)
            if usingNetworkTables:
                VISION_TABLE.putNumber('pid_offset',offset)
                time.sleep(0.00000001)
            if showWindows:
                frame = cv2.line(frame, center_of_window,centroid,(255,255,0),2)
                frame = cv2.circle(frame,center_of_window,2,(255,0,0),-1)
                frame = cv2.circle(frame,centroid,2,(0,255,0),-1)
                frame = cv2.drawContours(frame,[c],0,(0,0,255),2)
        if showWindows:
            cv2.imshow('frame', frame)
            cv2.imshow('hsv', hsv)
            cv2.imshow('blurred', blurred)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    
if __name__ == "__main__":
    main(False, True, True, False)
