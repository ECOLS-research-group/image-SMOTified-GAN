import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to the directory containing the image files
dir_path = 'data\\flowers\\train'

# Define the size of the images
new_size = (1000, 1000)

# Define the path to the directory where the new images will be saved
smote_dir_path = 'data\\flowers\\generated'

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
minority_class = [c for c, count in enumerate(np.bincount(class_labels)) if count < np.max(np.bincount(class_labels))]
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
smote = SMOTE()
new_images_flat, new_labels = smote.fit_resample(images_flat, labels)

# Reshape the new images to 2D arrays
new_images = new_images_flat.reshape(new_images_flat.shape[0], height, width, 3)

# Clip the pixel values to the range [0, 1]
new_images = np.clip(new_images, 0, 1)

# Create the directory to save the new images
if not os.path.exists(smote_dir_path):
    os.makedirs(smote_dir_path)

smote_batch = 0

# Save the new images to disk
for i in range(len(new_images)):
    img = new_images[i]
    label = new_labels[i]
    if label in minority_class:
        filename = os.path.join(smote_dir_path, 'label_%d_new_image_%d.jpg' % (label,i))
        cv2.imwrite(filename, img * 255)
        smote_batch+=1

##  GAN #######

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

# Get the indices of real images from the minority class
minority_class_indices = np.where(np.isin(class_labels, minority_class))[0]
minority_class_images = new_images[minority_class_indices]

# Parameters
latent_dim = 100
img_shape = (128, 128, 3)  # Update this to match your dataset's image size
channels = img_shape[-1]

# Build and compile the models
generator = build_generator(latent_dim, channels)
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))  # Compile discriminator
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

# Load and preprocess the dataset using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255)

batch_size = smote_batch
image_size = (128, 128)

print("train generator start")
train_generator = datagen.flow_from_directory(
    smote_dir_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True
)
print("train generator end")

# Create a directory to save generated images
output_dir = 'data\\flowers\\generated'
os.makedirs(output_dir, exist_ok=True)

# Training loop
epochs = 50
steps_per_epoch = len(minority_class_images) // batch_size  # Use the number of real minority class images for training

print("Starting training....\n")
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        # Get real images from the minority class
        start_idx = step * batch_size
        end_idx = (step + 1) * batch_size
        real_images = minority_class_images[start_idx:end_idx]
        real_labels = np.ones((batch_size, 1))

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)

        # Resize generated images to match the discriminator's input shape
        resized_generated_images = tf.image.resize(generated_images, (128, 128))

        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(resized_generated_images, fake_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid_labels)

        print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

        samples = 10
        noise = np.random.normal(0, 1, (samples, latent_dim))
        generated_images = generator.predict(noise)
        for i in range(samples):
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
            plt.axis('off')
            plt.savefig(f'{output_dir}/generated_{epoch}_{i}.png')
            plt.close()