import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# http://yann.lecun.com/exdb/mnist/

mnist = tf.keras.datasets.mnist

# loading the data and splitting them into training and testing datasets

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# normalize the values between 0-1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


# basic neural network model
model = tf.keras.models.Sequential()

#flattening the 28x28 image into 1d array
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))

model.add(tf.keras.layers.Dense(10, activation = 'softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 3, batch_size=32)

model.save('handwritten_model.keras')