import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
import cv2

# Define the path to the directory containing the image files
dir_path = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\train'

# Define the size of the images
new_size = (4000, 4000)

# Define the path to the directory where the new images will be saved
new_dir_path = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\generated'

# Load the image data using Keras
datagen = ImageDataGenerator(rescale=1. / 255)
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

# Get the class labels for each image
class_labels = np.argmax(labels, axis=1)

# Reshape the images to 2D arrays
height, width, channels = images[0].shape
images_2d = images.reshape(images.shape[0], height, width, channels)

# Resize the images
new_images = np.array([cv2.resize(img, new_size) for img in images_2d])

# Get the indices of the minority classes
minority_classes = [c for c, count in enumerate(np.bincount(class_labels)) if count < np.max(np.bincount(class_labels))]

# Loop through each minority class
for minority_class in minority_classes:
    print("Minority class: ", minority_class)

    # Get the indices of the minority class
    minority_class_indices = np.where(class_labels == minority_class)[0]

    # Get the images and labels of the minority class
    minority_images = new_images[minority_class_indices]
    minority_labels = class_labels[minority_class_indices]

    # Flatten the images to 1D arrays
    minority_images_flat = minority_images.reshape(minority_images.shape[0], -1)

    # Apply SMOTE to generate new images only if there is more than one class in the minority_labels
    if len(np.unique(minority_labels)) > 1:
        smote = SMOTE()
        minority_images_resampled, minority_labels_resampled = smote.fit_resample(minority_images_flat, minority_labels)
    else:
        minority_images_resampled, minority_labels_resampled = minority_images_flat, minority_labels

    # Reshape the new images to 2D arrays
    new_minority_images = minority_images_resampled.reshape(minority_images_resampled.shape[0], height, width, 3)

    # Clip the pixel values to the range [0, 1]
    new_minority_images = np.clip(new_minority_images, 0, 1)

    # Convert the labels to categorical format
    new_minority_labels = np.zeros((len(minority_labels_resampled), len(train_generator.class_indices)))
    for i, label in enumerate(minority_labels_resampled):
        new_minority_labels[i][label] = 1

    # Create the directory to save the new images
    if not os.path.exists(os.path.join(new_dir_path, str(minority_class))):
        os.makedirs(os.path.join(new_dir_path, str(minority_class)))

    # Save the new images to disk
    for i in range(len(new_minority_images)):
        img = new_minority_images[i]
        label = np.argmax(new_minority_labels[i])
        filename = os.path.join(new_dir_path, str(minority_class), 'new_image_%d_label_%d.jpg' % (i, label))
        cv2.imwrite(filename, img * 255)

