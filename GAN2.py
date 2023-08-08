import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, LeakyReLU, BatchNormalization, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define paths
dir_path = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\flowers\\train'
new_dir_path = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\flowers\\generatedByGAN'

# Step 1: Load and preprocess the image data
image_size = (128, 128)  # Adjust image size as needed
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    dir_path,
    target_size=image_size,
    batch_size=32,
    class_mode='input',  # Use 'input' for loading images without changing their format
    shuffle=True
)

# Convert the images to numpy arrays
images = []
for _ in range(len(train_generator)):
    batch_images, _ = train_generator.next()
    images.append(batch_images)

images = np.concatenate(images)

# Step 2: Define the generator network
latent_dim = 100
generator_input = Input(shape=(latent_dim,))
generator_model = Dense(128 * 16 * 16, activation='relu')(generator_input)
generator_model = BatchNormalization()(generator_model)
generator_model = Reshape((16, 16, 128))(generator_model)
generator_model = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(generator_model)
generator_model = BatchNormalization()(generator_model)
generator_output = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid')(generator_model)
generator = Model(inputs=generator_input, outputs=generator_output)

# Step 3: Compile the generator
generator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Step 4: Generate synthetic images using the GAN
num_samples = len(images)
num_synthetic_samples = num_samples  # Adjust the number of synthetic samples as needed

z = np.random.normal(size=(num_synthetic_samples, latent_dim))
synthetic_images = generator.predict(z)

# Step 5: Save synthetic images
os.makedirs(new_dir_path, exist_ok=True)
for i, synthetic_image in enumerate(synthetic_images):
    image_path = os.path.join(new_dir_path, f"synthetic_image_index_{i}.png")
    cv2.imwrite(image_path, (synthetic_image * 255).astype(np.uint8))

print("Synthetic image generation finished.")
