import os
import cv2
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

def load_images_from_folder(folder, target_size=(100, 100)):
    images = []
    labels = []
    class_mapping = {}
    class_index = 0

    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            class_mapping[class_name] = class_index
            class_index += 1
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = cv2.resize(img, target_size)  # Resize images to a consistent size
                images.append(img)
                labels.append(class_mapping[class_name])

    return np.array(images), np.array(labels), class_mapping

def apply_smote(images, labels, smote_ratio=1.0):
    # Flatten the images
    flattened_images = images.reshape(images.shape[0], -1)

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=smote_ratio, k_neighbors=3)
    new_images, new_labels = smote.fit_resample(flattened_images, labels)

    # Reshape the new images to their original shape
    new_images = new_images.reshape(new_images.shape[0], *images.shape[1:])

    return new_images, new_labels

# Set your image folder path
image_folder_path = 'data/anime_class'

# Load images and labels from the folder
images, labels, class_mapping = load_images_from_folder(image_folder_path)

# Shuffle the data
X_train, y_train = shuffle(images, labels, random_state=42)

# Count the samples in the minority class before SMOTE
minority_class_label = min(class_mapping.values())
minority_class_count = np.sum(y_train == minority_class_label)


# Apply SMOTE to the training set
X_train_smote, y_train_smote = apply_smote(X_train, y_train, smote_ratio=1.0)

# Find indices of samples from the minority class after skipping the first 5 samples
minority_class_indices = np.where(y_train_smote == minority_class_label)[0][5:]

# Extract only the data from the minority class after generating 5 images
x_train_generated_minority_class = X_train_smote[minority_class_indices]

# (ensure the output path exists before running this)
output_path = 'data/generated_images_smote'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save only the generated images from the minority class in the output folder
for i, image in enumerate(x_train_generated_minority_class):
    class_name = [k for k, v in class_mapping.items() if v == y_train_smote[minority_class_indices][i]][0]
    filename = f"{class_name}_smote_{i}.png"
    cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))