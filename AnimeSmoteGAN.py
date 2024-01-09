import os
import cv2
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

minority_class_count = 30
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

# Count the samples in the minority class before SMOTE
minority_class_label = min(class_mapping.values())
minority_class_count = np.sum(y_train == minority_class_label)


# Apply SMOTE to the training set
X_train_smote, y_train_smote = apply_smote(X_train, y_train, smote_ratio=1.0)

# Find indices of samples from the minority class after skipping the first 5 samples
minority_class_indices = np.where(y_train_smote == minority_class_label)[0][minority_class_count:]

# Extract only the data from the minority class after generating 5 images
x_train_generated_minority_class = X_train_smote[minority_class_indices]
x_train_generated_flat = x_train_generated_minority_class.reshape(x_train_generated_minority_class.shape[0], -1)
#GAN###############

# Define the paths and constants
data_path = 'data/anime_class/blue'
output_path = 'data/generated_images/'
batch_size = 20
epochs = 1500
latent_dim = 30000
interval = 1500

# Function to load and preprocess images
def load_images(image_path, img_size):
    images = []
    for filename in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, filename))
        img = img.resize((img_size, img_size))
        img = np.array(img)
        img = (img.astype(np.float32) - 127.5) / 127.5  # Normalize to range [-1, 1]
        images.append(img)
    return np.array(images)

# Generator model
def build_generator(latent_dim, img_size):
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod((img_size, img_size, 3)), activation='tanh'))
    model.add(layers.Reshape((img_size, img_size, 3)))
    return model

# Discriminator model
def build_discriminator(img_size):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(img_size, img_size, 3)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Build and compile the discriminator
img_size = 64  # Adjust this based on your image size
discriminator = build_discriminator(img_size)
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim, img_size)

# Create the GAN
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
generated_img = generator(gan_input)
gan_output = discriminator(generated_img)
gan = models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Load and preprocess images
real_images = load_images(data_path, img_size)

# Training loop
for epoch in range(epochs + 1):
    # Generate random noise as input to the generator
    noise = x_train_generated_flat
    # Generate fake images using the generator
    generated_images = generator.predict(noise)

    # Combine real and fake images into a batch
    batch_real_images = real_images[np.random.randint(0, real_images.shape[0], batch_size)]
    batch_labels_real = np.ones((batch_size, 1))
    batch_labels_fake = np.zeros((batch_size, 1))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(batch_real_images, batch_labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_images, batch_labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator (discriminator weights are frozen)
    noise = x_train_generated_flat
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print progress and save generated images at certain intervals
    if epoch % interval == 0:
        print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # Save generated images
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        generated_images = generated_images * 0.5 + 0.5  # Rescale images from [-1, 1] to [0, 1]
        for i in range(generated_images.shape[0]):
            plt.imsave(os.path.join(output_path, f"smotifiedGAN_{epoch}_{i}.png"), generated_images[i])

# plt.figure(figsize=(5, 5))
# for i in range(min(20, generated_images.shape[0])):
#     plt.subplot(5, 4, i + 1)  # Use 5 rows and 4 columns for a batch size of 20
#     plt.imshow(generated_images[i])
#     plt.axis('off')
# plt.show()
