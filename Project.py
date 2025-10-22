# project code
import os

import numpy as np

import tensorflow as tf


#building the model using CNN

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1) 
x_test = x_test.reshape(-1, 28, 28, 1)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

model.save('handwritten_cnn.keras')


#testing
model = tf.keras.models.load_model('handwritten.keras')
loss,accuracy = model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

#using data and showing results
from genericpath import isfile
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
model = tf.keras.models.load_model('handwritten.keras')
ImgNum = 1

while os.path.isfile(f"Images/Image{ImgNum}.png"):
  try:
    inputIMG = cv2.imread(f"Images/Image{ImgNum}.png")[:,:,0]
    inputIMG = np.invert(np.array([inputIMG]))
    result = model.predict(inputIMG)
    print(f"Digit is most likely a {np.argmax(result)}")
    plt.imshow(inputIMG[0],cmap=plt.cm.binary)
    plt.show()
  except:
    print("ERROR!")
  finally:
    ImgNum += 1








