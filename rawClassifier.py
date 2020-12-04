# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-11-17 15:25:50
# @Last modified by:   khai
# @Last Modified time: 2020-12-04 07:06:17
				
from sklearn import datasets 		
from sklearn.svm import LinearSVC								
import numpy as np 											
from collections import Counter
import cv2
import joblib


# Download the dataset
dataset = datasets.fetch_openml('mnist_784', version=1) 

# Extract the features and labels
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')


# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(features, labels)

# Save the classifier
joblib.dump(clf, "raw_digits_cls.pkl", compress=3)