import numpy as np
from PIL import Image
import cv2
from skimage import color
import tensorflow as tf


# Metoda koja učitava sliku i sprema ju u NumPy Array
def load(path):
    img = np.asarray(Image.open(path))
    if img.ndim == 2:
        img = np.tile(img[:, :, None], 3)
    return img


def resize(img):
    return np.asarray(Image.fromarray(img).resize(256, 256))


def preprocess(rgb_img):
    rgb_img_rs = resize(rgb_img)
    lab_img = color.rgb2lab(rgb_img)
    lab_img_rs = color.rgb2lab(rgb_img)

    l_img = lab_img[:, :, 0]
    l_img_rs = lab_img_rs[:, :, 0]

    tens = tf.Tensor(l_img)
    tens_rs = tf.Tensor(l_img)

    return tens, tens_rs

def postprocess(tens, out_ab, mode='bilinear'):

    HW_orig = tens.shape[2:]
    HW = out_ab.shape[2:]