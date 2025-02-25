import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import Counter

# Configurations
image_folder_path = 'flowers'  # Set your image folder path
output_path = 'output'  # Define output path
img_size = 100  # Image dimensions
latent_dim = 100  # Latent space size
epochs = 1000
interval = epochs
batch_size = 32

####################### Load and Preprocess Images #######################
def load_images_from_folder(folder, target_size=(img_size, img_size)):
    images, labels = [], []
    class_mapping = {}
    class_index = 0
    
    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            class_mapping[class_name] = class_index
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                images.append(img)
                labels.append(class_index)
            class_index += 1
    
    return np.array(images), np.array(labels), class_mapping

# Load images and labels
images, labels, class_mapping = load_images_from_folder(image_folder_path)
images = (images.astype(np.float32) - 127.5) / 127.5  # Normalize [-1, 1]
X_train, y_train = shuffle(images, labels, random_state=42)
class_counts = Counter(y_train)

# Identify minority classes (below average class count)
avg_count = np.mean(list(class_counts.values()))
minority_classes = {cls for cls, count in class_counts.items() if count < avg_count}

####################### Build GAN Model #######################
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

####################### Train GAN for Minority Classes #######################
for class_label in minority_classes:
    class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_label)]
    print(f"Training GAN for class {class_name}...")
    class_images = X_train[y_train == class_label]
    
    generator = build_generator(latent_dim, img_size)
    discriminator = build_discriminator(img_size)
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(latent_dim,))
    generated_img = generator(gan_input)
    gan_output = discriminator(generated_img)
    gan = models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    
    # Train GAN
    for epoch in range(epochs + 1):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)
        
        batch_real_images = class_images[np.random.randint(0, class_images.shape[0], batch_size)]
        batch_labels_real = np.ones((batch_size, 1))
        batch_labels_fake = np.zeros((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(batch_real_images, batch_labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_images, batch_labels_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss[0]} - G Loss: {g_loss}")
        
        if epoch == interval:
            # Save generated images
            class_output_path = os.path.join(output_path, class_name)
            if not os.path.exists(class_output_path):
                os.makedirs(class_output_path)
            generated_images = generated_images * 0.5 + 0.5  # Rescale images from [-1, 1] to [0, 1]
            for i in range(generated_images.shape[0]):
                plt.imsave(os.path.join(class_output_path, f"GAN_{epoch}_{i}.png"), generated_images[i])