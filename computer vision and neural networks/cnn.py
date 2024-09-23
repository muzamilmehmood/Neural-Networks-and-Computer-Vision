import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


mnist = tf.keras.datasets.mnist

# loading the data and splitting them into training and testing datasets

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the values between 0-1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


# basic neural network model
model = tf.keras.models.Sequential()

#flattening the 28x28 image into 1d array
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1))) # The one in "(28, 28, 1)" is required or else it'll error out
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(48, (3,3), activation='sigmoid'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5)

model.save('handwritten_cnn_model.keras')