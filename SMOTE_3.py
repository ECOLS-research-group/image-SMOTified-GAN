import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
import cv2

# Define the path to the directory containing the image files
dir_path = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\flowers\\train'

# Define the size of the images
new_size = (1000, 1000)

# Define the path to the directory where the new images will be saved
new_dir_path = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\flowers\\generated'

# Load the image data using Keras
datagen = ImageDataGenerator(rescale=1./255,
                             brightness_range=[0.9, 1.1],
                             contrast_range=[0.9, 1.1])
train_generator = datagen.flow_from_directory(
        dir_path,
        target_size=new_size,
        batch_size=32,
        class_mode='categorical')

# Convert the images to numpy arrays
images, labels = train_generator.next()

# Reshape the images to 2D arrays
height, width, channels = images[0].shape
images_2d = images.reshape(images.shape[0], height, width, channels)

# Resize the images
new_images = np.array([cv2.resize(img, new_size) for img in images_2d])

# Flatten the images to 1D arrays
images_flat = new_images.reshape(new_images.shape[0], -1)

# Apply SMOTE to generate new images
smote = SMOTE()
new_images_flat, new_labels = smote.fit_resample(images_flat, labels)

# Reshape the new images to 2D arrays
new_images = new_images_flat.reshape(new_images_flat.shape[0], height, width, 3)

# Clip the pixel values to the range [0, 1]
new_images = np.clip(new_images, 0, 1)

# Create the directory to save the new images
if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)

# Save the new images to disk
for i in range(len(new_images)):
    img = new_images[i]
    label = new_labels[i]
    filename = os.path.join(new_dir_path, 'new_image_%d_label_%d.jpg' % (label,i))
    cv2.imwrite(filename, img * 255)
