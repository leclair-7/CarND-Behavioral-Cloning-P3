import os
import time
import csv

import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.backend import tf as ktf
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
from math import isclose
import random

STEER_CORRECTION = .2
PERCENTAGE_FLIPPED = .7
BRIGHTNESS_CHANGES = .6
width = 320
height = 160

NEW_WIDTH = 66
NEW_HEIGHT = 200

num_channels = 3
BATCH_SIZE = 128
EPOCHS = 3
ACTIVATION_FUNCTION = 'elu'

data = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)
def flipImages(images, steering_angle):
    flipped = np.copy(np.fliplr(images))   
    return (flipped, -1 * steering_angle)

def change_brightness(image):
    img = np.copy(image)
    image1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # uniform means all outcomes equally likely, 
    # defaults to [0,1)
    random_bright = .25 + np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def preprocess_image(image):
    '''
    crop
    resize 200 across 66 up/down
    convert to YUV
    '''
    print(image.shape)
    x,w = 25,image.shape[1]-25    
    #left to cut off, right to cut off
    y,h = 65, (image.shape[0] - 90)
    image = np.copy(image[y:y+h, x:x+w])
    print(image.shape)
    porky=input()
    image = cv2.resize(image, (200,66) )
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

train_samples, validation_samples = train_test_split(data, test_size=0.2)

def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+ batch_sample[0].split('/')[-1]
                left = './data/IMG/'+ batch_sample[1].split('/')[-1]
                right = './data/IMG/'+ batch_sample[2].split('/')[-1]
                
                #print(len(images), name)
                if name == "./data/IMG/center":
                    continue
                    
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                left_image  = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2RGB)

                #steering angles
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3]) + STEER_CORRECTION
                right_angle = float(batch_sample[3]) - STEER_CORRECTION
                
                coinFlip = random.random()
                if isclose(center_angle, 0.0):
                    if coinFlip > .7:
                        images.extend( (center_image,left_image, right_image ) )
                        steering_angles.extend( (center_angle, left_angle, right_angle) )
                else:
                    images.extend( (center_image,left_image, right_image ) )
                    steering_angles.extend( (center_angle, left_angle, right_angle) )
                if coinFlip < PERCENTAGE_FLIPPED:
                    flipcenter = flipImages(center_image,center_angle)
                    flipleft = flipImages(left_image,left_angle)
                    flipright = flipImages(right_image,right_angle)

                    images.append(flipcenter[0])
                    steering_angles.append(flipcenter[1])

                    images.append(flipleft[0])
                    steering_angles.append(flipleft[1])

                    images.append(flipright[0])
                    steering_angles.append(flipright[1])
                if coinFlip < BRIGHTNESS_CHANGES:
                    img_left_bright = change_brightness(left_image)
                    img_right_bright = change_brightness(right_image)

                    images.append(img_left_bright)
                    steering_angles.append(left_angle)

                    images.append(img_right_bright)
                    steering_angles.append(right_angle)
            X_train = np.array(images)
            y_train = np.array(steering_angles)

            X_train, y_train = shuffle(X_train, y_train)
            yield X_train[:BATCH_SIZE], y_train[:BATCH_SIZE]

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

def my_resize_function(images):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(images, (NEW_WIDTH,NEW_HEIGHT))

model = Sequential()
model.add(Lambda(lambda x: x/255.5 - 0.5, input_shape=(height, width, 3)))

model.add(Cropping2D(((70,25),(0,0)), input_shape=(height,width,3)))

#model.add(Lambda(lambda x: cv2.resize(x, (NEW_HEIGHT, NEW_WIDTH)) ) )
#, input_shape=(num_channels, height, width), output_shape=(ch, new_height, new_width))

#model.add(Lambda(lambda x: ImageDataGenerator(x, )))
#model.add(Lambda(lambda image: tf.image.resize_images(image, (NEW_HEIGHT, NEW_WIDTH))))
model.add(Lambda(my_resize_function))

model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2),
                 activation=ACTIVATION_FUNCTION))

model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2),
                 activation=ACTIVATION_FUNCTION  ))

model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2),
                 activation=ACTIVATION_FUNCTION))
#model.add()
'''
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2,2),
                 activation=ACTIVATION_FUNCTION))
'''
'''
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
                 activation=ACTIVATION_FUNCTION))
'''
model.add(Flatten())

#model.add(Dense(1184, activation=ACTIVATION_FUNCTION))
model.add(Dense(100, activation=ACTIVATION_FUNCTION))
model.add(Dense(50, activation=ACTIVATION_FUNCTION))
model.add(Dense(10, activation=ACTIVATION_FUNCTION))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=.2,shuffle=True, epochs=EPOCHS, verbose=1)

model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=EPOCHS, steps_per_epoch=len(train_samples)//BATCH_SIZE)

model.save('model.h5')

