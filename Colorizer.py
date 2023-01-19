import os
import keras
import numpy as np
from NeuralNetwork import create_model
from keras.utils import load_img, img_to_array
from skimage.color import rgb2lab
from skimage import color

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
input = keras.Input(shape=(None, None, 1))
model = create_model(input)

# Finish model
model = keras.Model(input, model)

model.compile(optimizer='Adam', loss='mse')
model.fit(X, Y, batch_size=1, epochs=200, verbose=1)
model.save('model.h5')

model.evaluate(X, Y, batch_size=1)

