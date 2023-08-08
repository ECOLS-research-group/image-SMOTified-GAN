import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Concatenate
from keras.layers import LeakyReLU, BatchNormalization, Conv2DTranspose, Conv2D
from keras.optimizers import Adam

print("Task Started")
# Step 1: Load imbalanced image dataset
dir_path = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\xray\\train'
new_dir_path = 'C:\\Users\\singh\\OneDrive\\Documents\\Projects\\SMOTified-GAN-Image\\data\\xray\\generated'  # New directory for saving generated images

# Define the size of the images
new_size = (1000, 1000)

# Load the image data using Keras
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    dir_path,
    target_size=new_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Convert the images to numpy arrays
images_list = []
labels_list = []
for _ in range(len(train_generator)):
    images, labels = train_generator.next()
    images_list.append(images)
    labels_list.append(labels)

images = np.concatenate(images_list)
labels = np.concatenate(labels_list)

# Step 2 and 3: Calculate class distribution and determine underrepresented classes
class_counts = np.sum(labels, axis=0)
underrepresented_classes = np.where(class_counts < np.max(class_counts))[0]

# Step 4: Oversample underrepresented classes using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
images_flat = images.reshape(images.shape[0], -1)
images_resampled_flat, labels_resampled = smote.fit_resample(images_flat, labels)
images_resampled = images_resampled_flat.reshape(images_resampled_flat.shape[0], *new_size, 3)

# Step 5: Load the oversampled image dataset
images = images_resampled
labels = labels_resampled

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 7: Define the generator network with an additional input layer for class labels
def build_generator(latent_dim, num_classes):
    z = Input(shape=(latent_dim,))
    label = Input(shape=(num_classes,))
    combined = Concatenate()([z, label])
    generator = Dense(256, activation='relu')(combined)
    generator = BatchNormalization()(generator)
    generator = Dense(125 * 125 * 128, activation='relu')(generator)
    generator = BatchNormalization()(generator)
    generator = Reshape((125, 125, 128))(generator)
    generator = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(generator)
    generator = BatchNormalization()(generator)
    generator = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid')(generator)
    generator = Conv2DTranspose(3, kernel_size=8, strides=1, padding='same', activation='sigmoid')(generator)  # Add this line
    model = Model([z, label], generator)
    return model


def build_discriminator(input_shape, num_classes):
    img_input = Input(shape=input_shape)
    label = Input(shape=(num_classes,))

    # Add a Dense layer to match the spatial dimensions of the image input
    label_embedding = Dense(np.prod(input_shape[1:3]))(label)
    label_embedding = Reshape((1, 1, -1))(label_embedding)

    # Upsample the label embedding to match the spatial dimensions of the image input
    label_embedding = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(label_embedding)
    label_embedding = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(label_embedding)

    combined = Concatenate()([img_input, label_embedding])
    discriminator = Conv2D(64, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(combined)
    discriminator = Conv2D(128, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2))(
        discriminator)
    discriminator = Flatten()(discriminator)
    discriminator = Dense(1, activation='sigmoid')(discriminator)
    model = Model([img_input, label], discriminator)
    return model


# Step 9: Compile the generator and discriminator networks
latent_dim = 100
num_classes = labels.shape[1]  # Update num_classes based on the number of unique labels (one-hot encoded)
generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator(new_size + (3,), num_classes)

discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

z = Input(shape=(latent_dim,))
label = Input(shape=(num_classes,))
img = generator([z, label])
discriminator.trainable = False
validity = discriminator([img, label])
combined = Model([z, label], validity)
combined.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Step 10: Define the SMOTE-GAN model by connecting the generator and discriminator networks
def train_smote_gan(generator, discriminator, combined, X_train, y_train, latent_dim, num_classes):
    epochs = 1000  # You can adjust the number of epochs as needed
    batch_size = 32
    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        for batch_idx in range(num_batches):
            # Select a random batch of images and labels
            batch_images = X_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_labels = y_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            # Generate fake images using the generator
            z = np.random.normal(size=(batch_size, latent_dim))
            generated_images = generator.predict([z, batch_labels])

            # Train the discriminator with real and fake images
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch([batch_images, batch_labels], real_labels)
            d_loss_fake = discriminator.train_on_batch([generated_images, batch_labels], fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator to fool the discriminator
            valid_labels = np.ones((batch_size, 1))
            g_loss = combined.train_on_batch([z, batch_labels], valid_labels)  # Corrected line

        # Save generated images after every 10 epochs
        if epoch % 10 == 0:
            save_generated_images(generator, epoch)

        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")


def save_generated_images(generator, epoch):
    # Create the new directory if it doesn't exist
    os.makedirs(new_dir_path, exist_ok=True)

    # Generate and save 10 images with random labels
    num_generated_images = 10
    for i in range(num_generated_images):
        label = np.random.randint(num_classes, size=(1,))
        z = np.random.normal(size=(1, latent_dim))
        generated_image = generator.predict([z, label])[0]

        # Rescale the generated image to 0-255 range
        generated_image = (generated_image * 255).astype(np.uint8)

        # Save the image
        cv2.imwrite(
            os.path.join(new_dir_path, f"generated_image_epoch_{epoch}_label_{label}.png"),
            cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)
        )


# Step 12: Save the trained SMOTE-GAN model and the oversampled image dataset
# Save the models using model.save() and np.savez() to save the oversampled dataset
generator.save('smote_gan_generator.h5')
discriminator.save('smote_gan_discriminator.h5')

# Save the oversampled dataset as a NumPy array
np.savez('oversampled_dataset.npz', X_train=X_train, y_train=y_train)

print("Task Completed")