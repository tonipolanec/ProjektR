import os
import cv2
import keras
import numpy as np
from matplotlib import pyplot
from keras.utils import load_img, img_to_array
from skimage.color import lab2rgb

#load the model
loaded_model = keras.models.load_model('model.h5')

test_folder = "dataset_bw/"
# Test images path
for img_name in os.listdir(test_folder):
    img = test_folder + "\\" + img_name
    img2 = load_img(img)
    img2 = img_to_array(img2) / 255
    ss = img2.shape

    img = load_img(img, target_size=(100, 100))
    arr_img = img_to_array(img) / 255

    X = np.array(arr_img)
    X = np.expand_dims(X, axis=2)
    X = np.resize(X, [1, 100, 100, 1])
    # X = np.reshape(X, (1, 100, 100, 1))

    # Predicting test image and producing output
    output = loaded_model.predict(X)
    output = np.reshape(output, (100, 100, 2))
    output = cv2.resize(output, (ss[1], ss[0]))

    # Combining output with input test image
    AB_img = output
    outputLAB = np.zeros(ss)
    outputLAB[:, :] = img2
    outputLAB[:, :, 1:] = AB_img
    outputLAB = (outputLAB * [100, 255, 255]) - [0, 128, 128]
    rgb_image = lab2rgb(outputLAB)
    # saving each image to results folder
    result_path = "results\\" + img_name
    pyplot.imsave(result_path, rgb_image)