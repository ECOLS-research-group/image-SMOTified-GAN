import os
import numpy as np
from PIL import Image

folder_path = ["/path/to/folder1", "/path/to/folder2"]
img_width = 1708  # Placeholder for the desired width of your images
img_height = 1944  # Placeholder for the desired height of your images

images = []
labels = []

for class_folder in os.listdir(folder_path):
    class_path = os.path.join(folder_path, class_folder)
    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((img_width, img_height))
        image = np.array(image)
        images.append(image)
        labels.append(class_folder)

images = np.array(images)
labels = np.array(labels)
