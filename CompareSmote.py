from collections import Counter
import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
import cv2

# Define the path to the directory containing the image files
dir_path = 'data\\xray\\train'

# Define the size of the images
new_size = (100, 100)

# Load the image data using Keras
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    dir_path,
    target_size=new_size,
    batch_size=32,
    class_mode='categorical')

# Print the class names and the total number of images in each class
print("Class names: ", train_generator.class_indices)
print("Total images in each class: ", np.bincount(train_generator.classes))

# Convert the images to numpy arrays
images, labels = train_generator.next()

######################################
# Get the class labels for each image
class_labels = np.argmax(labels, axis=1)
# Get the indices of the minority classes
minority_class = [c for c, count in enumerate(np.bincount(
    class_labels)) if count < np.max(np.bincount(class_labels))]
print("Minority class: ", minority_class)
###########################################

# Reshape the images to 2D arrays
height, width, channels = images[0].shape
images_2d = images.reshape(images.shape[0], height, width, channels)

# Resize the images
new_images = np.array([cv2.resize(img, new_size) for img in images_2d])

# Flatten the images to 1D arrays
images_flat = new_images.reshape(new_images.shape[0], -1)

# Apply SMOTE to generate new images
smote = SMOTE(sampling_strategy='auto', k_neighbors=3)
x_train_res, y_train_res = smote.fit_resample(images_flat, labels)

print(y_train_res)

comparison_result = np.setdiff1d(x_train_res, images_flat)
intesectresult = np.intersect1d(x_train_res, images_flat)
#print(comparison_result)
print(len(comparison_result))
#print(intesectresult)
print(len(intesectresult))
