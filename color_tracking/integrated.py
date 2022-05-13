
from collections import deque
from imutils.video import VideoStream
import cap as cap
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# Capturing video through webcam or video path
if not args.get("video", False):
    webcam = VideoStream(src=0).start()
    # webcam = cv2.VideoCapture(0)
else:
    webcam = cv2.VideoCapture(args["video"])


# Start a while loop
while True:

    # Reading the video from the
    # webcam in image frames
    frame = webcam.read()
    imageFrame = frame[1] if args.get("video", False) else frame

    if imageFrame is None:
        break

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=blue_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            imageFrame = cv2.circle(imageFrame, center, max(w, h)//2, (0, 0, 255), 2)
            imageFrame = cv2.circle(imageFrame, center, max(w, h)//10, (0, 0, 255), -1)

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            imageFrame = cv2.circle(imageFrame, center, max(w, h)//2, (0, 255, 0), 2)
            imageFrame = cv2.circle(imageFrame, center, max(w, h)//10, (0, 255, 0), -1)

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            imageFrame = cv2.circle(imageFrame, center, max(w, h)//2, (255, 0, 0), 2)
            imageFrame = cv2.circle(imageFrame, center, max(w, h)//10, (255, 0, 0), -1)

    # Program Termination
    # show the frame to our screen
    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    webcam.stop()

# otherwise, release the camera
else:
    webcam.release()

# close all windows
cv2.destroyAllWindows()
