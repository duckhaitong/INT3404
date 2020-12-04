# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-11-11 15:22:41
# @Last modified by:   khai
# @Last Modified time: 2020-12-04 08:58:57

# Import the modules
import cv2
import joblib
from skimage.feature import hog
from skimage import exposure
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

# Load the classifier
clf = joblib.load("digits_cls.pkl")
# Read the input image 
im = cv2.imread(args["image"])

# Convert to grayscale and apply Gaussian filtering to filter noisy pixels
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
cv2.namedWindow("Gaussian Blur", cv2.WINDOW_NORMAL)
cv2.imshow("Gaussian Blur", im_gray)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 85, 255, cv2.THRESH_BINARY_INV)
cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
cv2.imshow("Threshold", im_th)

# Find contours in the image
contours, hierarchy = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0,255,0), 3)
cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
cv2.imshow("Contours", im)

# Get rectangles contains each contour
rects = [] # rect return format x,y,w,h
for ctr in contours:
    rects.append(cv2.boundingRect(ctr))


# For each rectangular region, calculate HOG features and predict the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    # Regions of interested
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # cv2.imshow("ROI", roi)
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imshow("normalized ROI", roi)
  
    # Calculate the HOG features
    roi_hog_fd, hog_image = hog(roi, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), visualize=True)
    # cv2.imshow("ROI HOG FD", roi_hog_fd)
    cv2.imshow("HOG IMAGE", hog_image)

    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    
cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
cv2.destroyAllWindows() 
