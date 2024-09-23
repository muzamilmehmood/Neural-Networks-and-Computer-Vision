import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model = tf.keras.models.load_model('handwritten_model.keras')

image_number = 0

'''
[:,:,0]:

This part of the code is used to slice the array returned by cv2.imread.
[:, :]: This means "all rows and all columns" of the image.
0: This indicates that only the first channel of the image should be selected.
'''

while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img = cv2.imread(f'digits/digit{image_number}.png')[:,:,0]
        img = np.invert(np.array([img]))

        prediction = model.predict(img)
        print(f'This digit is probably a: {np.argmax(prediction)}')

        #we can also show the image in graph just to make sure
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as ex:
        print(f'Error: {ex}')
    finally:
        image_number += 1