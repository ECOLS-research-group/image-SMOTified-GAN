from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix


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
epochs = 10
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

############ Precision-Recall Curve ###############
# Make predictions on the validation set
y_pred_prob = model.predict(validation_generator)

y_true_binary = validation_generator.classes

# Calculate precision-recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_true_binary, y_pred_prob[:, 1])
average_precision = average_precision_score(y_true_binary, y_pred_prob[:, 1])

# Print average precision and recall
print(f'Average Precision: {average_precision:.4f}')
print(f'Average Recall: {np.mean(recall):.4f}')

# Plot precision-recall curve
# Plot precision-recall curve as a line graph
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve (AUC = {:.2f})'.format(average_precision))
plt.show()