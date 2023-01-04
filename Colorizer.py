import os
from keras import Input, Model
from keras.applications import Xception
from keras.callbacks import TensorBoard
import numpy as np
from skimage.color import rgb2lab, gray2rgb
from keras.layers import Lambda
from keras import backend
from keras.utils import plot_model

from Evaluator import test_model
from NeuralNetwork import create_model
from generators import generate_datasets
from generators import generate_datasets_RGB
#from output import create_save_images

dataset_path = "./resources/original_images"
number_of_images = len(os.walk(dataset_path).__next__()[1])

train_dataset, validation_dataset = generate_datasets(dataset_path, 1)
train_dataset_RGB, validation_dataset_RGB = generate_datasets_RGB(dataset_path, 1)

encoder_input = Input(shape=(256, 256, 1,), name='luminance')
model_embedings = Input(shape=((1000,)), name='embeding')
outputs = create_model(encoder_input)
model = Model(inputs=[encoder_input], outputs=outputs)
#model = Model(inputs=[model_embedings, encoder_input], outputs=outputs)#

#plot_model(model, to_file='model.png')

callbacks = [TensorBoard(log_dir=".\\Tensorboard_logs", histogram_freq=1, profile_batch=100000000)]

model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

# Nije nam jasno što trebaju biti prva dva argumenta.
history_log = model.fit(train_dataset, #train_dataset_RGB,
                        validation_data=validation_dataset,
                        epochs=100,
                        steps_per_epoch=round(number_of_images * 0.8),
                        validation_steps=round(number_of_images * 0.2),
                        callbacks=callbacks)

model.save('convolution.h5')

# Ne znamo je li nam ovo uopće potrebno?
# save_results_path = "resources/results/image"
# files = os.listdir(dataset_path)

# for index, file in enumerate(files):
#    test, luminance_test, cur = test_model(, model)
#    create_save_images(test, luminance_test, cur, save_results_path, index)
