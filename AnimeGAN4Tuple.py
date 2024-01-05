from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

height = 64
width = 64
channel = 3
latent_dim = 100
img_size = 64

data_path = 'data/anime'
output_path = 'data/generated_images/'

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

def build_generator(height, width, channel):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(height, width, channel)))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    # Adjust the number of units in the last dense layer
    model.add(layers.Dense(3, activation='tanh'))

    model.add(layers.Reshape((height, width, channel)))
    return model



def build_discriminator(height, width, channel):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(height, width, channel)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


generator = build_generator(height, width, channel)
discriminator = build_discriminator(height, width, channel)


# GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Set discriminator weights as non-trainable
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


# Build and compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build the generator
generator = build_generator(height, width, channel)

# Build and compile the GAN
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Load and preprocess images (replace this with your data loading code)
real_images = load_images(data_path, img_size)

# Training loop
batch_size = 5  # Adjust as needed
epochs = 400  # Adjust as needed

for epoch in range(epochs + 1):
    # Generate random noise as input to the generator
    noise = np.random.normal(0, 1, (batch_size, height, width, channel))

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
    noise = np.random.normal(0, 1, (batch_size, height, width, channel))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print progress and save generated images at certain intervals
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # Save generated images
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        generated_images = generated_images * 0.5 + 0.5  # Rescale images from [-1, 1] to [0, 1]
        for i in range(generated_images.shape[0]):
            plt.imsave(os.path.join(output_path, f"generated_image_{epoch}_{i}.png"), generated_images[i])
