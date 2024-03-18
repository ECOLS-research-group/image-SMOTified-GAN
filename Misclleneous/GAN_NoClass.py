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
x_train_res = x_train_res[8:]
x_train_res = x_train_res[:, :100]

#x_train_res = x_train_res.reshape(x_train_res.shape[0], height, width)  # Reshape to match image dimensions
#x_train_res = x_train_res / 255.0

# Define the Generator model


def build_generator(latent_dim, channels):
    model = models.Sequential()
    model.add(layers.Dense(100 * 100 * channels, input_dim=latent_dim))
    model.add(layers.Reshape((100, 100, channels)))
    model.add(layers.Conv2DTranspose(
        100, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.Conv2DTranspose(
        64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.Conv2DTranspose(channels, (4, 4),
              activation='sigmoid', padding='same'))
    return model


# Define the Discriminator model
def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2),
              padding='same', input_shape=img_shape))
    model.add(layers.Conv2D(100, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# Define GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()

    # Add generator layers
    model.add(generator)

    # Add discriminator layers (without the fully connected layer)
    # Exclude the last layer (fully connected)
    for layer in discriminator.layers[:-1]:
        model.add(layer)

    return model


# Parameters
latent_dim = 100
img_shape = (100, 100, 3)  # Update this to match your dataset's image size
channels = img_shape[-1]
num_categories = 2

# Build and compile the models
generator = build_generator(latent_dim, channels)
discriminator = build_discriminator((100, 100, 3))
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(
    lr=0.0002, beta_1=0.5))  # Compile discriminator
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

# Load and preprocess the dataset using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255)

data_dir = "data\\xray\\train"
# Adjust the batch size to be a multiple of the number of categories (e.g., 32/2 = 16)
batch_size = 32
image_size = (100, 100)

# Load and preprocess real data
real_data_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


# Create a directory to save generated images
output_dir = "data\\xray\\generated\\smotifiedGAN"
os.makedirs(output_dir, exist_ok=True)

# Training loop
epochs = 50
steps_per_epoch = len(real_data_generator)

for epoch in range(epochs):
    print(f"Epoch {epoch}")
    for step in range(steps_per_epoch):
        real_images, real_labels = real_data_generator.next()

        noise = x_train_res
        generated_images = generator.predict(noise)

        resized_generated_images = tf.image.resize(
            generated_images, (100, 100))

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(
            resized_generated_images, fake_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = x_train_res
        valid_labels = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid_labels)

    print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

    if epoch == 0:
        samples = 10
        for i in range(samples):
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
            plt.axis('off')
            plt.savefig(f'{output_dir}/generated_epoch_{epoch}_sample_{i}.png')
            plt.close()

    if epoch % 10 == 0:
        samples = 10
        for i in range(samples):
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
            plt.axis('off')
            plt.savefig(f'{output_dir}/generated_epoch_{epoch}_sample_{i}.png')
            plt.close()
