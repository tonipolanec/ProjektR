import os
import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, InputLayer
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.models import Sequential
from keras.utils import load_img, img_to_array
from skimage.color import lab2rgb, rgb2lab
from skimage import color
from matplotlib.pyplot import imshow

gray_folder = "dataset_bw"
images1 = []
for img in os.listdir(gray_folder):
    img = gray_folder + "\\" + img
    img = load_img(img, target_size=(100, 100))
    img = img_to_array(img)/255

    X = color.rgb2gray(img)
    images1.append(X)

folder_train_rgb = "dataset"
images2 = []
for img in os.listdir(folder_train_rgb):
    img = folder_train_rgb + "\\" + img
    img = load_img(img, target_size=(100, 100))
    img = img_to_array(img)/255
    lab_img = rgb2lab(img)
    lab_img_norm = (lab_img + [0, 128, 128]) / [100, 255, 255]
    # Input je black and white layer
    Y = lab_img_norm[:, :, 1:]
    images2.append(Y)

X = np.array(images1)
Y = np.array(images2)

# Samo lightness channel (gray) input, a na output je u boji
x1 = keras.Input(shape=(None, None, 1))
x2 = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(x1)
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x4 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x3)
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
x6 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x5)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(32, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(16, (3, 3), activation='relu', padding='same')(x9)
x11 = UpSampling2D((2, 2))(x10)
x12 = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x11)

x12 = tf.reshape(x12, (1, 104, 104, 2))
x12 = tf.image.resize(x12, [100, 100])
x12 = tf.reshape(x12, (1, 100, 100, 2))

# Finish model
model = keras.Model(x1, x12)

model.compile(optimizer='Adam', loss='mse')
model.fit(X, Y, batch_size=1, epochs=100, verbose=1)
model.save('model.h5')

model.evaluate(X, Y, batch_size=1)

# Test image path
test_folder = "testing/"
name_img = "bw_6.jpg"
img = test_folder + name_img

img2 = load_img(img)
img2 = img_to_array(img2)/255
ss = img2.shape

img = load_img(img, target_size=(100, 100))
arr_img = img_to_array(img)/255

X = np.array(arr_img)
X = np.expand_dims(X, axis=2)
X = np.resize(X, [1, 100, 100, 1])
#X = np.reshape(X, (1, 100, 100, 1))

# Predicting test image and producing output
output = model.predict(X)
output = np.reshape(output, (100, 100, 2))
output = cv2.resize(output, (ss[1], ss[0]))

# Combining output with input test image
AB_img = output
outputLAB = np.zeros(ss)
outputLAB[:, :] = img2
outputLAB[:, :, 1:] = AB_img
outputLAB = (outputLAB * [100, 255, 255]) - [0, 128, 128]
rgb_image = lab2rgb(outputLAB)

imshow(rgb_image)
plt.show()

result_path = "results\\" + name_img
pyplot.imsave(result_path, rgb_image)
