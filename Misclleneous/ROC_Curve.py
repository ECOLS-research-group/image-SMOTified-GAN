import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix


# Define the GoogLeNet (Inception) model
def create_googlenet_model(input_shape=(224, 224, 3), num_classes=2):
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

# Load your dataset from a folder (replace 'your_dataset_folder' with the actual path)
dataset_folder = 'data/chestxray'
batch_size = 80

# Use ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into training and validation sets
)

train_generator = train_datagen.flow_from_directory(
    dataset_folder,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_folder,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create and compile the model
model = create_googlenet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

############ ROC Curve ###############
# Make predictions on the validation set
y_pred_prob = model.predict(validation_generator)

# Convert one-hot encoded true labels to binary
y_true_binary = validation_generator.classes

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
