# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-11-11 15:22:41
# @Last modified by:   khai
# @Last Modified time: 2020-12-05 08:28:36

# Import the modules
import cv2
import joblib
from skimage.feature import hog
from skimage import exposure
import numpy as np
import argparse
import matplotlib.pyplot as plt


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
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
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

    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
  
    # Calculate the HOG features
    roi_hog_fd, hog_image = hog(roi, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), block_norm="L2", visualize=True)

    # Get the predicted digits
    predicted_digits = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(predicted_digits[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)


cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
cv2.destroyAllWindows() 

# Visualize the HOG image
fd, hog_image = hog(im_gray, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), block_norm="L1", visualize=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(im, cmap=plt.cm.gray)
ax1.set_title('Input image')

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
