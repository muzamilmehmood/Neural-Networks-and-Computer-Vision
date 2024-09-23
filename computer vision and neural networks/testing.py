import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the values between 0-1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


model = tf.keras.models.load_model('handwritten_model.keras')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

#256 neurons in dense layers give loss: 0.0913
#256 neurons in dense layers give Accuracy 0.9728