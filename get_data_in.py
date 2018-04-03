from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.backend import tf as ktf
from keras.preprocessing.image import ImageDataGenerator
import time
import tensorflow as tf

import os
import csv

STEER_CORRECTION = .2

height = 160
width = 320
NEW_HEIGHT = 120
NEW_WIDTH = 240
num_channels = 3
BATCH_SIZE = 256


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=256):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format
def my_resize_function(images):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(images, (96,96))

model = Sequential()
model.add(Lambda(lambda x: x/255.5 - 0.5, input_shape=(height, width, 3)))

model.add(Cropping2D(((50,10),(0,0)), input_shape=(height,width,3)))

#model.add(Lambda(lambda x: cv2.resize(x, (NEW_HEIGHT, NEW_WIDTH)) ) )
#, input_shape=(num_channels, height, width), output_shape=(ch, new_height, new_width))

#model.add(Lambda(lambda x: ImageDataGenerator(x, )))
#model.add(Lambda(lambda image: tf.image.resize_images(image, (NEW_HEIGHT, NEW_WIDTH))))
model.add(Lambda(my_resize_function))

model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2),
                 activation='elu'))

model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2),
                 activation='elu'  ))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
                 activation='elu'))
model.add(Flatten())

model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=.2,shuffle=True, epochs=EPOCHS, verbose=1)

model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3, steps_per_epoch=len(train_samples)//BATCH_SIZE)

model.save('model.h5')

"""
If the above code throw exceptions, try 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
"""

