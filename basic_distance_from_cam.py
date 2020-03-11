import cv2
import numpy as np
import math
import subprocess

lower_bounds = [40, 58, 94]
upper_bounds = [255, 255, 255]

def nothing(x):
    pass

def main(trackbar = True, showWindows = True, usingWindows = True):
    if trackbar:
        # creates the trackbar
        cv2.namedWindow('Trackbar')
        # creates "variables" (low_h,low_s,...) linked to the 'Trackbar' window, with a range of 0, 255
        cv2.createTrackbar('low_h', 'Trackbar', 0, 255, nothing)
        cv2.createTrackbar('low_s', 'Trackbar', 0, 255, nothing)
        cv2.createTrackbar('low_v', 'Trackbar', 0, 255, nothing)
        cv2.createTrackbar('up_h', 'Trackbar', 255, 255, nothing)
        cv2.createTrackbar('up_s', 'Trackbar', 255, 255, nothing)
        cv2.createTrackbar('up_v', 'Trackbar', 255, 255, nothing)

    cap = cv2.VideoCapture(0)
    # we capture the first frame for the camera to adjust itself to the exposure
    ret_val , cap_for_exposure = cap.read()
    if showWindows:
        cv2.namedWindow('hsv',cv2.WINDOW_NORMAL)
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    print(cap_for_exposure.shape[1])
    print(cap_for_exposure.shape[0])
    if showWindows:
        pass
        #cv2.imshow('testing_frame', cap_for_exposure)
    if usingWindows:
        # set exposure so that we brightness does not affect color sensing
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE , -1)
    else:
        subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_auto=1 -c exposure_absolute=10",shell=True)

    # continuously run this code as long as we are capturing video
    while cap.isOpened():
        # captures the frame
        ret, frame = cap.read()
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
            # sets the lower and upper hsv range based on the variables from the trackbar
            lower_hsvRange = np.array([low_h, low_s, low_v])
            upper_hsvRange = np.array([up_h, up_s, up_v])
        else:
            lower_hsvRange = np.array(lower_bounds)
            upper_hsvRange = np.array(upper_bounds)

        # this asks: are the hsv values in the lower and upper ranges?
        # if the pixels in the frame are in the range, they show up to be white, and black if they are not
        # ie, white = in range, black = not in range
        hsv = cv2.inRange(hsv, lower_hsvRange, upper_hsvRange)

        # ret, hsv = cv2.threshold(hsv, 155, 255, cv2.THRESH_BINARY)

        # kernel for the morphological transformations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # morphological transformations are used to reduce noise
        hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)

        # blur the frame to reduce noise
        # hsv = cv2.medianBlur(hsv, 5)
        # blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # gets the contours based on the blurred variable(the blurred + morphed frame image)
        contours, hierarchy = cv2.findContours(hsv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        try:
            # get the maximum contour, or the MAIN(the big rectangle) contour of the object, because there are a lot of
            # contours that from noise this is in a try block because if the frame has no contours, the max() function
            # will return an error
            c = max(contours, key=cv2.contourArea)
        except Exception as e:
            # print(e)
            continue
        else:

            # creates the smallest rectangle from the max, or MAIN contours
            rect = cv2.minAreaRect(c)

            # gets the box points of that rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # draws the box on the frame
            if showWindows:
                frame = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            # this is mainly following the formula

            # focal length is a constant for the width you are measuring
            # focallength = (pixelwidth x knowndistance) / knownwidth
            #F = (P x  D) / W

            KNOWN_DISTANCE = 15
            KNOWN_WIDTH = 3.25
            width_and_height = rect[1]
            
            # uncomment this to use basic distance detection
            # f = 803 # for the longer side, with minarearect
            f = 1431
            # width_and_height = np.array([height, width])
            r = np.amax(width_and_height)
            # f = (r * KNOWN_DISTANCE) / KNOWN_WIDTH
            d = (f * KNOWN_WIDTH) / r
            print(str(d) + ': feet')
            # print(f)
        if showWindows:
            cv2.imshow('frame', frame)
            cv2.imshow('hsv', hsv)
            cv2.resizeWindow('frame',640,480)
            cv2.resizeWindow('hsv',640,480)
        # cv2.imshow('blurred', blurred)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main(trackbar = False, showWindows = False, usingWindows = False)
