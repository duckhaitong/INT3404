# -*- coding: utf-8 -*-
# @Author: Duc Khai Tong
# @Date:   2020-11-11 15:15:40
# @Last modified by:   khai
# @Last Modified time: 2020-12-04 10:32:20


# Import the modules
from sklearn import datasets 									
from skimage.feature import hog							
from sklearn.svm import LinearSVC								
import numpy as np 												
import cv2
import joblib
from skimage import exposure
import matplotlib.pyplot as plt


# Download the dataset
dataset = datasets.fetch_openml('mnist_784', version=1) 

# Extract the features and labels
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

print(np.shape(features))

# Extract the HOG features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), visualize=False)
    list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')

print(np.shape(fd))

# # Create an linear SVM object
# clf = LinearSVC()

# # Perform the training
# clf.fit(hog_features, labels)
# # Save the classifier
# joblib.dump(clf, "digits_cls.pkl", compress=3)

# Visualize the HOG image of MNIST data
image = features[0].reshape((28, 28))
fd, hog_image = hog(image, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(1, 1), visualize=True)
print(np.shape(hog_image))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
# # Rescale histogram for better display
# hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
