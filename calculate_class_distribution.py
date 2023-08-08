import os

# Directory containing the image dataset
dataset_dir = 'dataset/covid19/test'

# Dictionary to hold the class counts
class_counts = {}

# Iterate over the class directories
for class_dir in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_dir)
    if os.path.isdir(class_path):
        # Count the number of images in this class directory
        num_images = len(os.listdir(class_path))

        # Add the count to the class counts dictionary
        class_counts[class_dir] = num_images

# Print the class distribution
print(class_counts)