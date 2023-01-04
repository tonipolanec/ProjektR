import os
from keras import Input, Model
from keras.applications import Xception
from keras.callbacks import TensorBoard
import numpy as np
from skimage.color import rgb2lab, gray2rgb
from keras.layers import Lambda
from keras import backend

from Evaluator import test_model
from NeuralNetwork import create_model
from generators import generate_datasets
from output import create_save_images

dataset_training_path = "resources/data_set/training"
number_of_images = len(os.walk(dataset_training_path).__next__()[1])

# pretrained model when dataset is too small
transfer_learning_model = Xception()
for layer in transfer_learning_model.layers:
    layer.trainable = False
train_transfer_learning_dataset, validation_transfer_learning_dataset = generate_datasets(dataset_training_path, 1)

train_dataset, validation_dataset = generate_datasets(dataset_training_path, 1)

encoder_input = Input(shape=(224, 224, 1,), name='luminance')
model_embedings = Input(shape=((1000,)), name='embeding')
outputs = create_model(encoder_input)
model = Model(inputs=[model_embedings, encoder_input], outputs=outputs)

callbacks = [TensorBoard(log_dir=".\\Tensorboard_logs", histogram_freq=1, profile_batch=100000000)]

model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])


# koristiti ImageProcessor?
def image_gen(dataset, transfer_learning_dataset, transfer_learning_model):
    for rgb_image, rgb_tl_image in zip(dataset, transfer_learning_dataset):
        lab_image = rgb2lab(rgb_image[0])
        luminance = lab_image[:, :, :, [0]]
        ab_components = lab_image[:, :, :, 1:] / 128
        tl_model_features = []
        lab_image_tl = rgb2lab(rgb_tl_image[0])
        luminance_tl = lab_image_tl[:, :, :, [0]]

        for i, sample in enumerate(luminance_tl):
            sample = gray2rgb(sample)
            sample = sample.reshape((1, 299, 299, 3))
            embedding = transfer_learning_model.predict(sample)
            tl_model_features.append(embedding)

        tl_model_features = np.array(tl_model_features)
        tl_model_features_shape_2d = backend.int_shape(Lambda(lambda x: x[:, 0, :], dtype='float32')(tl_model_features))
        tl_model_features = tl_model_features.reshape(tl_model_features_shape_2d)
        yield ([tl_model_features, luminance], ab_components)


history_log = model.fit(image_gen(train_dataset, train_transfer_learning_dataset, transfer_learning_model),
                        validation_data=image_gen(validation_dataset, validation_transfer_learning_dataset,
                                                  transfer_learning_model),
                        epochs=100,
                        steps_per_epoch=round(number_of_images * 0.8),
                        validation_steps=round(number_of_images * 0.2),
                        callbacks=callbacks)

model.save('convolution_xception.h5')

dataset_testing_path = "resources/data_set/testing"
save_results_path = "resources/results/image"
files = os.listdir(dataset_testing_path)

for index, file in enumerate(files):
    test, luminance_test, cur = test_model(dataset_testing_path, transfer_learning_model, model)
    create_save_images(test, luminance_test, cur, save_results_path, index)
