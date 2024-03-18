from random import shuffle
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle


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

def apply_smote(images, labels):
    # Flatten the images
    flattened_images = images.reshape(images.shape[0], -1)

    # Apply SMOTE
    smote = SMOTE(sampling_strategy="auto", k_neighbors=3)
    new_images, new_labels = smote.fit_resample(flattened_images, labels)

    # Reshape the new images to their original shape
    new_images = new_images.reshape(new_images.shape[0], *images.shape[1:])

    return new_images, new_labels

# Set your image folder path
image_folder_path = 'data/chestxray_SMOTE'

# Load images and labels from the folder
images, labels, class_mapping = load_images_from_folder(image_folder_path)

# Shuffle the data
X_train, y_train = shuffle(images, labels, random_state=42)

# Count the samples in the minority class before SMOTE
minority_class_label = min(class_mapping.values())
minority_class_count = np.sum(y_train == minority_class_label)


# Apply SMOTE to the training set
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

# Find indices of samples from the minority class after skipping the first 5 samples
minority_class_indices = np.where(y_train_smote == minority_class_label)[0][20:]

# Extract only the data from the minority class after generating 5 images
x_train_generated_minority_class = X_train_smote[minority_class_indices]

# (ensure the output path exists before running this)
output_path = 'data/chestxray_SMOTE/0TB'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save only the generated images from the minority class in the output folder
for i, image in enumerate(x_train_generated_minority_class):
    class_name = [k for k, v in class_mapping.items() if v == y_train_smote[minority_class_indices][i]][0]
    filename = f"{class_name}_smote_{i}.png"
    cv2.imwrite(os.path.join(output_path, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


#################    GAN      #########################
    
# Define the paths and constants
data_path = 'data/chestxray_GAN/0TB'
output_path = 'data/chestxray_GAN/0TB'
batch_size = 60
epochs = 1000
latent_dim = 30000
interval = 1000

# Function to load and preprocess images
def load_images(image_path, img_size):
    images = []
    for filename in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, filename))
        img = img.resize((img_size, img_size))
        img = img.convert("RGB")
        img = np.array(img)
        img = (img.astype(np.float32) - 127.5) / 127.5  # Normalize to range [-1, 1]
        images.append(img)
    return np.array(images)

# Generator model
def build_generator(latent_dim, img_size):
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    #model.add(layer_norm.LeakyReLU(alpha=0.2))
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
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
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

    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

    # Print progress and save generated images at certain intervals
    if epoch == interval:
        print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # Save generated images
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        generated_images = generated_images * 0.5 + 0.5  # Rescale images from [-1, 1] to [0, 1]
        for i in range(generated_images.shape[0]):
            plt.imsave(os.path.join(output_path, f"generated_image_{epoch}_{i}.png"), generated_images[i])

#################    SMOTified-GAN  ##########################
            
minority_class_count = 20
# Set your image folder path
image_folder_path = 'data/chestxray_SMOTifiedGAN'
# Define the paths and constants
data_path = 'data/chestxray_SMOTifiedGAN/0TB'
output_path = 'data/chestxray_SMOTifiedGAN/0TB'
batch_size = 60
epochs = 1000
latent_dim = 30000
interval = 1000


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

def apply_smote(images, labels):
    # Flatten the images
    flattened_images = images.reshape(images.shape[0], -1)

    # Apply SMOTE
    smote = SMOTE(sampling_strategy="auto", k_neighbors=3)
    new_images, new_labels = smote.fit_resample(flattened_images, labels)

    # Reshape the new images to their original shape
    new_images = new_images.reshape(new_images.shape[0], *images.shape[1:])

    return new_images, new_labels

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
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

# Find indices of samples from the minority class after skipping the first 5 samples
minority_class_indices = np.where(y_train_smote == minority_class_label)[0][minority_class_count:]

# Extract only the data from the minority class after generating 5 images
x_train_generated_minority_class = X_train_smote[minority_class_indices]
x_train_generated_flat = x_train_generated_minority_class.reshape(x_train_generated_minority_class.shape[0], -1)

# Function to load and preprocess images
def load_images(image_path, img_size):
    images = []
    for filename in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, filename))
        img = img.resize((img_size, img_size))
        img = img.convert("RGB")
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

    print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
    # Print progress and save generated images at certain intervals
    if epoch == interval:
        #print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # Save generated images
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        generated_images = generated_images * 0.5 + 0.5  # Rescale images from [-1, 1] to [0, 1]
        for i in range(generated_images.shape[0]):
            plt.imsave(os.path.join(output_path, f"smotifiedGAN_{epoch}_{i}.png"), generated_images[i])

################  GoogleNET ######################
nonOversampledData = 'data/chestxray_nonOversampled'
SMOTEData = 'data/chestxray_SMOTE'
GANData = 'data/chestxray_GAN'
SMOTifiedGANData = 'data/chestxray_SMOTifiedGAN'

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

# Load images and labels from the folder
images_nonOversampled, labels_nonOversampled, class_mapping_nonOversampled = load_images_from_folder(nonOversampledData)
images_SMOTE, labels_SMOTE, class_mapping_SMOTE = load_images_from_folder(SMOTEData)
images_GAN, labels_GAN, class_mapping_GAN = load_images_from_folder(GANData)
images_SMOTifiedGAN, labels_SMOTifiedGAN, class_mapping_SMOTifiedGAN = load_images_from_folder(SMOTifiedGANData)

# Split data into train and test sets
X_train_nonOversampled, X_test_nonOversampled, y_train_nonOversampled, y_test_nonOversampled = train_test_split(images_nonOversampled, labels_nonOversampled, test_size=0.2, random_state=42)
X_train_SMOTE, X_test_SMOTE, y_train_SMOTE, y_test_SMOTE = train_test_split(images_SMOTE, labels_SMOTE, test_size=0.2, random_state=42)
X_train_GAN, X_test_GAN, y_train_GAN, y_test_GAN = train_test_split(images_GAN, labels_GAN, test_size=0.2, random_state=42)
X_train_SMOTifiedGAN, X_test_SMOTifiedGAN, y_train_SMOTifiedGAN, y_test_SMOTifiedGAN = train_test_split(images_SMOTifiedGAN, labels_SMOTifiedGAN, test_size=0.2, random_state=42)


# Define the GoogLeNet (Inception) model
epochs = 10

def create_googlenet_model(input_shape=(100, 100, 3), num_classes=2):
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

############### non oversampled  #############
# Create and compile the model
model = create_googlenet_model(input_shape=X_train_nonOversampled.shape[1:], num_classes=len(class_mapping_nonOversampled))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_nonOversampled, y_train_nonOversampled, epochs=epochs)

# Evaluate the model on the training set
train_loss_nonOversampled, train_accuracy_nonOversampled = model.evaluate(X_train_nonOversampled, y_train_nonOversampled)
test_loss_nonOversampled, test_accuracy_nonOversampled = model.evaluate(X_test_nonOversampled, y_test_nonOversampled)

############### SMOTE  #############
# Create and compile the model
model = create_googlenet_model(input_shape=X_train_SMOTE.shape[1:], num_classes=len(class_mapping_SMOTE))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_SMOTE, y_train_SMOTE, epochs=epochs)

# Evaluate the model on the training set
train_loss_SMOTE, train_accuracy_SMOTE = model.evaluate(X_train_SMOTE, y_train_SMOTE)
test_loss_SMOTE, test_accuracy_SMOTE = model.evaluate(X_test_SMOTE, y_test_SMOTE)

############### GAN  #############
# Create and compile the model
model = create_googlenet_model(input_shape=X_train_GAN.shape[1:], num_classes=len(class_mapping_GAN))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_GAN, y_train_GAN, epochs=epochs)

# Evaluate the model on the training set
train_loss_GAN, train_accuracy_GAN = model.evaluate(X_train_GAN, y_train_GAN)
test_loss_GAN, test_accuracy_GAN = model.evaluate(X_test_GAN, y_test_GAN)

############### SMOTIfied-GAN  #############
# Create and compile the model
model = create_googlenet_model(input_shape=X_train_SMOTifiedGAN.shape[1:], num_classes=len(class_mapping_SMOTifiedGAN))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_SMOTifiedGAN, y_train_SMOTifiedGAN, epochs=epochs)

# Evaluate the model on the training set
train_loss_SMOTifiedGAN, train_accuracy_SMOTifiedGAN = model.evaluate(X_train_SMOTifiedGAN, y_train_SMOTifiedGAN)
test_loss_SMOTifiedGAN, test_accuracy_SMOTifiedGAN = model.evaluate(X_test_SMOTifiedGAN, y_test_SMOTifiedGAN)


print(f"\nTrain Accuracy Non Oversampled: {train_accuracy_nonOversampled * 100:.2f}%")
print(f"\nTest Accuracy Non Oversampled: {test_accuracy_nonOversampled * 100:.2f}%")

print(f"\nTrain Accuracy SMOTE: {train_accuracy_SMOTE * 100:.2f}%")
print(f"\nTest Accuracy SMOTE: {test_accuracy_SMOTE * 100:.2f}%")

print(f"\nTrain Accuracy GAN: {train_accuracy_GAN * 100:.2f}%")
print(f"\nTest Accuracy GAN: {test_accuracy_GAN * 100:.2f}%")

print(f"\nTrain Accuracy SMOTified-GAN: {train_accuracy_SMOTifiedGAN * 100:.2f}%")
print(f"\nTest Accuracy SMOTified-GAN: {test_accuracy_SMOTifiedGAN * 100:.2f}%")