# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-11-11 15:15:40
# @Last modified by:   khai
# @Last Modified time: 2020-12-04 07:06:18


# Import the modules
from sklearn import datasets 									
from skimage.feature import hog									
from sklearn.svm import LinearSVC								
import numpy as np 												
import cv2
import joblib

# Download the dataset
dataset = datasets.fetch_openml('mnist_784', version=1) 

# Extract the features and labels
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

# Extract the HOG features
list_hog_fd = []
for feature in features:
    fd, hog_image = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=True)
    # cv2.imshow("Hog Image", hog_image)
    list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)