import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, InputLayer

def create_model(input):
    model = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(input)
    model = Conv2D(16, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(model)
    model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(model)
    model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
    model = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(model)
    # model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
    # model = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(model)
    model = UpSampling2D((2, 2))(model)
    # model = Conv2D(128, (3, 3), activation='relu', padding='same')(model)
    # model = UpSampling2D((2, 2))(model)
    model = Conv2D(64, (3, 3), activation='relu', padding='same')(model)
    model = UpSampling2D((2, 2))(model)
    model = Conv2D(32, (3, 3), activation='relu', padding='same')(model)
    model = UpSampling2D((2, 2))(model)
    model = Conv2D(16, (3, 3), activation='relu', padding='same')(model)
    model = UpSampling2D((2, 2))(model)
    model = Conv2D(2, (3, 3), activation='tanh', padding='same')(model)

    model = tf.reshape(model, (1, 112, 112, 2))
    model = tf.image.resize(model, [100, 100])
    model = tf.reshape(model, (1, 100, 100, 2))

    return model