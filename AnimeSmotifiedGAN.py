import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import cv2
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, target_size=(64, 64)):
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
images, labels = shuffle(images, labels, random_state=42)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
X_train_smote, y_train_smote = apply_smote(X_train, y_train, smote_ratio=1.0)

# Save the oversampled images if needed
# (ensure the output path exists before running this)
output_path = 'data/generated_images_smote'
if not os.path.exists(output_path):
    os.makedirs(output_path)

for i, image in enumerate(X_train_smote):
    class_name = [k for k, v in class_mapping.items() if v == y_train_smote[i]][0]
    filename = f"{class_name}_smote_{i}.png"
    cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Now, X_train_smote and y_train_smote contain the oversampled training set


# Define the paths and constants
data_path = 'data/anime'
output_path = 'data/generated_smotified_gan'
batch_size = 20
epochs = 1000
latent_dim = 100

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
    noise = X_train_smote

    # # Reshape the oversampled data to match the input shape of the generator
    # noise = noise.reshape((noise.shape[0], latent_dim))

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
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print progress and save generated images at certain intervals
    if epoch % 200 == 0:
        print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # Save generated images
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        generated_images = generated_images * 0.5 + 0.5  # Rescale images from [-1, 1] to [0, 1]
        for i in range(generated_images.shape[0]):
            plt.imsave(os.path.join(output_path, f"generated_image_{epoch}_{i}.png"), generated_images[i])

# Display generated images
# plt.figure(figsize=(10, 10))
# for i in range(20):
#     plt.subplot(5, 5, i + 1)
#     plt.imshow(generated_images[i])
#     plt.axis('off')
# plt.show()
plt.figure(figsize=(10, 10))
for i in range(min(20, generated_images.shape[0])):
    plt.subplot(5, 4, i + 1)  # Use 5 rows and 4 columns for a batch size of 20
    plt.imshow(generated_images[i])
    plt.axis('off')
plt.show()
