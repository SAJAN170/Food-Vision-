Food Vision: Advanced Image Classification with TensorFlow
Project Overview
This project showcases the application of Convolutional Neural Networks (CNNs) and Transfer Learning for food image classification using TensorFlow. The workflow includes binary and multi-class classification tasks on datasets such as Food-101, utilizing techniques like fine-tuning, data augmentation, and experiment tracking.

Key Features
1. Convolutional Neural Networks and Computer Vision
Preprocessing datasets (e.g., pizza, steak categories).
Building CNN architectures for binary and multi-class classification.
Training, evaluation, and prediction with models.
2. Transfer Learning and Fine-Tuning
Adapting pre-trained models like EfficientNet.
Fine-tuning on small datasets with data augmentation.
Using ModelCheckpoint callback to save training progress.
Experiment comparison with TensorBoard.
3. Dataset Scaling
Scaling from small subsets to the full Food-101 dataset.
Preprocessing images (resizing, normalization) with TensorFlow Datasets.
Visualization of samples and predictions with Matplotlib.
Installation
pip install tensorflow tensorflow_datasets matplotlib
Usage
Binary Classification Example
Download and preprocess the dataset:

!wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip
unzip_data("pizza_steak.zip")
Train a simple CNN:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=5, validation_data=test_data)
Fine-Tuning Example
Fine-tune a pre-trained EfficientNet model:

!wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip
unzip_data("10_food_classes_10_percent.zip")
import tensorflow as tf
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=5, validation_data=test_data)
Results
Binary Classification Accuracy: Achieved 95% accuracy (pizza vs. steak).
Multi-Class Classification: Competitively performed on Food-101 using fine-tuned models.
References
TensorFlow Documentation
Food-101 Dataset
Author
Developed as part of an advanced TensorFlow deep learning course.
