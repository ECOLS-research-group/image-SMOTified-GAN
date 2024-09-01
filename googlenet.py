from random import shuffle
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

################  GoogleNET ######################
nonOversampledData = 'data/chestxray_nonOversampled'
SMOTEData = 'data/chestxray_SMOTE'
GANData = 'data/chestxray_GAN'
MCMCData = 'data/chestxray_MCMC'
SmotifiedGANData = 'data/chestxray_SmotifiedGAN'

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
images_MCMC, labels_MCMC, class_mapping_MCMC = load_images_from_folder(MCMCData)
images_SmotifiedGAN, labels_SmotifiedGAN, class_mapping_SmotifiedGAN = load_images_from_folder(SmotifiedGANData)

# Split data into train and test sets
X_train_nonOversampled, X_test_nonOversampled, y_train_nonOversampled, y_test_nonOversampled = train_test_split(images_nonOversampled, labels_nonOversampled, test_size=0.2, random_state=42)
X_train_SMOTE, X_test_SMOTE, y_train_SMOTE, y_test_SMOTE = train_test_split(images_SMOTE, labels_SMOTE, test_size=0.2, random_state=42)
X_train_GAN, X_test_GAN, y_train_GAN, y_test_GAN = train_test_split(images_GAN, labels_GAN, test_size=0.2, random_state=42)
X_train_MCMC, X_test_MCMC, y_train_MCMC, y_test_MCMC = train_test_split(images_MCMC, labels_MCMC, test_size=0.2, random_state=42)
X_train_SmotifiedGAN, X_test_SmotifiedGAN, y_train_SmotifiedGAN, y_test_SmotifiedGAN = train_test_split(images_SmotifiedGAN, labels_SmotifiedGAN, test_size=0.2, random_state=42)


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
model_NONOversampled = create_googlenet_model(input_shape=X_train_nonOversampled.shape[1:], num_classes=len(class_mapping_nonOversampled))
model_NONOversampled.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_NONOversampled.fit(X_train_nonOversampled, y_train_nonOversampled, epochs=epochs)

# Evaluate the model on the training set
train_loss_nonOversampled, train_accuracy_nonOversampled = model_NONOversampled.evaluate(X_train_nonOversampled, y_train_nonOversampled)
test_loss_nonOversampled, test_accuracy_nonOversampled = model_NONOversampled.evaluate(X_test_nonOversampled, y_test_nonOversampled)

############### SMOTE  #############
# Create and compile the model
model_SMOTE = create_googlenet_model(input_shape=X_train_SMOTE.shape[1:], num_classes=len(class_mapping_SMOTE))
model_SMOTE.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_SMOTE.fit(X_train_SMOTE, y_train_SMOTE, epochs=epochs)

# Evaluate the model on the training set
train_loss_SMOTE, train_accuracy_SMOTE = model_SMOTE.evaluate(X_train_SMOTE, y_train_SMOTE)
test_loss_SMOTE, test_accuracy_SMOTE = model_SMOTE.evaluate(X_test_SMOTE, y_test_SMOTE)

############### GAN  #############
# Create and compile the model
model_GAN = create_googlenet_model(input_shape=X_train_GAN.shape[1:], num_classes=len(class_mapping_GAN))
model_GAN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_GAN.fit(X_train_GAN, y_train_GAN, epochs=epochs)

# Evaluate the model on the training set
train_loss_GAN, train_accuracy_GAN = model_GAN.evaluate(X_train_GAN, y_train_GAN)
test_loss_GAN, test_accuracy_GAN = model_GAN.evaluate(X_test_GAN, y_test_GAN)

############### MCMC  #############
# Create and compile the model
model_MCMC = create_googlenet_model(input_shape=X_train_MCMC.shape[1:], num_classes=len(class_mapping_MCMC))
model_MCMC.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_MCMC.fit(X_train_MCMC, y_train_MCMC, epochs=epochs)

# Evaluate the model on the training set
train_loss_MCMC, train_accuracy_MCMC = model_MCMC.evaluate(X_train_MCMC, y_train_MCMC)
test_loss_MCMC, test_accuracy_MCMC = model_MCMC.evaluate(X_test_MCMC, y_test_MCMC)

############### SmotifiedGAN  #############
# Create and compile the model
model_SmotifiedGAN = create_googlenet_model(input_shape=X_train_SmotifiedGAN.shape[1:], num_classes=len(class_mapping_SmotifiedGAN))
model_SmotifiedGAN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_SmotifiedGAN.fit(X_train_SmotifiedGAN, y_train_SmotifiedGAN, epochs=epochs)

# Evaluate the model on the training set
train_loss_SmotifiedGAN, train_accuracy_SmotifiedGAN = model_SmotifiedGAN.evaluate(X_train_SmotifiedGAN, y_train_SmotifiedGAN)
test_loss_SmotifiedGAN, test_accuracy_SmotifiedGAN = model_SmotifiedGAN.evaluate(X_test_SmotifiedGAN, y_test_SmotifiedGAN)

# Define a function to print precision, recall, and F1 score
def print_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, digits=4)
    print(f"\nClassification Report for {model_name}:")
    print(report)

# Call the function for each model
# Non Oversampled
y_pred_nonOversampled = model_NONOversampled.predict(X_test_nonOversampled).argmax(axis=1)
print_classification_report(y_test_nonOversampled, y_pred_nonOversampled, "Non Oversampled")
print(f"Train Accuracy Non Oversampled: {train_accuracy_nonOversampled * 100:.4f}%")
print(f"\nTest Accuracy Non Oversampled: {test_accuracy_nonOversampled * 100:.4f}%")

# SMOTE
y_pred_SMOTE = model_SMOTE.predict(X_test_SMOTE).argmax(axis=1)
print_classification_report(y_test_SMOTE, y_pred_SMOTE, "SMOTE")
print(f"Train Accuracy SMOTE: {train_accuracy_SMOTE * 100:.4f}%")
print(f"\nTest Accuracy SMOTE: {test_accuracy_SMOTE * 100:.4f}%")

# GAN
y_pred_GAN = model_GAN.predict(X_test_GAN).argmax(axis=1)
print_classification_report(y_test_GAN, y_pred_GAN, "GAN")
print(f"Train Accuracy GAN: {train_accuracy_GAN * 100:.4f}%")
print(f"\nTest Accuracy GAN: {test_accuracy_GAN * 100:.4f}%")

# MCMC
y_pred_MCMC = model_MCMC.predict(X_test_MCMC).argmax(axis=1)
print_classification_report(y_test_MCMC, y_pred_MCMC, "MCMC")
print(f"Train Accuracy MCMC: {train_accuracy_MCMC * 100:.4f}%")
print(f"\nTest Accuracy MCMC: {test_accuracy_MCMC * 100:.4f}%")

# SmotifiedGAN
y_pred_SmotifiedGAN = model_SmotifiedGAN.predict(X_test_SmotifiedGAN).argmax(axis=1)
print_classification_report(y_test_SmotifiedGAN, y_pred_SmotifiedGAN, "SmotifiedGAN")
print(f"Train Accuracy SmotifiedGAN: {train_accuracy_SmotifiedGAN * 100:.4f}%")
print(f"\nTest Accuracy SmotifiedGAN: {test_accuracy_SmotifiedGAN * 100:.4f}%")

#Precision N Recall Curve

def plot_precision_recall(y_true, y_pred_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_score = auc(recall, precision)
    plt.plot(recall, precision, label=f"{model_name} (AUC = {auc_score:.4f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

# Call the function for each model
# Non Oversampled
y_pred_proba_nonOversampled = model_NONOversampled.predict(X_test_nonOversampled)[:, 1]
plot_precision_recall(y_test_nonOversampled, y_pred_proba_nonOversampled, "Non Oversampled")

# SMOTE
y_pred_proba_SMOTE = model_SMOTE.predict(X_test_SMOTE)[:, 1]
plot_precision_recall(y_test_SMOTE, y_pred_proba_SMOTE, "SMOTE")

# GAN
y_pred_proba_GAN = model_GAN.predict(X_test_GAN)[:, 1]
plot_precision_recall(y_test_GAN, y_pred_proba_GAN, "GAN")

# MCMC
y_pred_proba_MCMC = model_MCMC.predict(X_test_MCMC)[:, 1]
plot_precision_recall(y_test_MCMC, y_pred_proba_MCMC, "MCMC")

# SmotifiedGAN
y_pred_proba_SmotifiedGAN = model_SmotifiedGAN.predict(X_test_SmotifiedGAN)[:, 1]
plot_precision_recall(y_test_SmotifiedGAN, y_pred_proba_SmotifiedGAN, "SmotifiedGAN")

plt.show()

plt.show()