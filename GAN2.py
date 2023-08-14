import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the Generator model
def build_generator(latent_dim, channels):
    model = models.Sequential()
    model.add(layers.Dense(128 * 128 * channels, input_dim=latent_dim))
    model.add(layers.Reshape((128, 128, channels)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.Conv2DTranspose(channels, (4, 4), activation='sigmoid', padding='same'))  # Updated here
    return model

# Define the Discriminator model
def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
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
    for layer in discriminator.layers[:-1]:  # Exclude the last layer (fully connected)
        model.add(layer)

    return model

# Parameters
latent_dim = 100
img_shape = (128, 128, 3)  # Update this to match your dataset's image size
channels = img_shape[-1]
num_categories = 2

# Build and compile the models
generator = build_generator(latent_dim, channels)
discriminator = build_discriminator((128, 128, 3))
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))  # Compile discriminator
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

# Load and preprocess the dataset using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255)
data_dir = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\xray\\train'
batch_size = 32  # Adjust the batch size to be a multiple of the number of categories (e.g., 32/2 = 16)
image_size = (128, 128)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


# Create a directory to save generated images
output_dir = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\xray\\generated'
os.makedirs(output_dir, exist_ok=True)

# Training loop
epochs = 5000
steps_per_epoch = len(train_generator)

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        real_images, real_labels = train_generator.next()  # Get both images and labels from the generator

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)

        real_labels = np.ones(batch_size)
        fake_labels = np.zeros(batch_size)

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones(batch_size)

        g_loss = gan.train_on_batch(noise, valid_labels)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

        if epoch % 1000 == 0:
            samples = 10
            noise = np.random.normal(0, 1, (samples, latent_dim))
            generated_images = generator.predict(noise)
            for i in range(samples):
                plt.imshow(generated_images[i, :, :, 0], cmap='gray')
                plt.axis('off')
                plt.savefig(f'{output_dir}/generated_{epoch}_{i}.png')
                plt.close()
