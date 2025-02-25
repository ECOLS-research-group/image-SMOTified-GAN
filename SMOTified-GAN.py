import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

image_folder_path = 'flowers'  # Set your image folder path
output_path = 'output'  # Define the paths and constants
img_size = 64  # Adjust this based on your image size
epochs = 2000

interval = epochs
latent_dim = img_size * img_size * 3
#################    SMOTified-GAN  ##########################
def load_images_from_folder(folder, target_size=(img_size, img_size)):
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


def apply_smote(images, labels):
    # Flatten the images
    flattened_images = images.reshape(images.shape[0], -1)

    # Apply SMOTE to balance the classes
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3)  # 'auto' will balance all classes
    new_images, new_labels = smote.fit_resample(flattened_images, labels)

    # Reshape the new images to their original shape
    new_images = new_images.reshape(new_images.shape[0], *images.shape[1:])

    return new_images, new_labels


# Load images and labels from the folder
images, labels, class_mapping = load_images_from_folder(image_folder_path)

# Shuffle the data
X_train, y_train = shuffle(images, labels, random_state=42)

# Get the original class distribution
original_class_counts = {class_label: np.sum(y_train == class_label) for class_label in class_mapping.values()}

# Apply SMOTE to the entire dataset
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

# Create a dictionary to track the oversampled data for each class
oversampled_data = {}

######################### Generator and Discriminator ########################

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

# For each class, check if its count increased, indicating it was a minority class
for class_label in class_mapping.values():
    class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_label)]

    # Get the original and new count for this class
    original_count = original_class_counts[class_label]
    new_count = np.sum(y_train_smote == class_label)

    # Check if SMOTE increased the count (i.e., the class was oversampled)
    if new_count > original_count:
        # Extract only the newly generated samples for this class
        X_smote_class = X_train_smote[y_train_smote == class_label]
        #original class data
        X_original_class = X_train[y_train == class_label]
        # The number of new synthetic samples created
        synthetic_samples_count = new_count - original_count
        # Get only the newly generated samples (skip the original ones)
        X_smote_new_samples = X_smote_class[original_count:]  # Only the synthetic samples
        # Store the oversampled data for this class
        oversampled_data[class_label] = X_smote_new_samples

        # Load and preprocess original images for discriminator
        real_images = X_original_class
        #  oversampled images as input to the generator
        X_smote_new_samples_flat = X_smote_new_samples.reshape(X_smote_new_samples.shape[0], -1)

        ########### GAN Initialise #######################
        # Build and compile the discriminator
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

        noise = X_smote_new_samples_flat
        # Training loop
        for epoch in range(epochs + 1):
            # Generate fake images using the generator
            generated_images = generator.predict(noise)

            # Combine real and fake images into a batch
            batch_real_images = real_images[np.random.randint(0, real_images.shape[0], synthetic_samples_count)]
            batch_labels_real = np.ones((synthetic_samples_count, 1))
            batch_labels_fake = np.zeros((synthetic_samples_count, 1))

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(batch_real_images, batch_labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, batch_labels_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            valid_labels = np.ones((synthetic_samples_count, 1))
            g_loss = gan.train_on_batch(noise, valid_labels)

            # Train the generator (discriminator weights are frozen)
            noise = np.random.normal(0, 1, (synthetic_samples_count, latent_dim))

            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            # Print progress and save generated images at certain intervals
            if epoch == interval:
                # Save generated images
                class_output_path = os.path.join(output_path, class_name)
                if not os.path.exists(class_output_path):
                    os.makedirs(class_output_path)
                generated_images = generated_images * 0.5 + 0.5  # Rescale images from [-1, 1] to [0, 1]
                for i in range(generated_images.shape[0]):
                    plt.imsave(os.path.join(class_output_path, f"smotifiedGAN_{epoch}_{i}.png"), generated_images[i])