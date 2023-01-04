import tensorflow as tf
from keras.applications import Xception
from tensorflow.python.keras.layers import Input, Conv2D, UpSampling2D, RepeatVector, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend


def create_model(input):
    model = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(input)
    model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(model)
    model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(model)
    model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(512, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)

    embeding = Reshape((28, 28, 1000))(RepeatVector(28 * 28)(Input(shape=((1000,)), name='embeding')))

    model = backend.concatenate([model, embeding], axis=3)
    model = Conv2D(256, (1, 1), activation='relu', padding='same')(model)

    model = Conv2D(256, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
    model = UpSampling2D((2, 2))(model)
    model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
    model = UpSampling2D((2, 2))(model)
    model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(16, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(2, (3, 3), activation='tanh', padding='same')(model)
    model = UpSampling2D((2, 2))(model)

    return model








