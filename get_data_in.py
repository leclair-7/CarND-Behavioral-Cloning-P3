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
    1 crop
    2 resize 200 across 66 up/down
    3 convert to YUV
    '''
    x,w = 25,image.shape[1]-25    
    #left to cut off, right to cut off
    y,h = 65, (image.shape[0] - 90)
    image = np.copy(image[y:y+h, x:x+w])

    #print(image.shape)
    image = cv2.resize(image, (200,66) )
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def generator(samples, tags, batch_size=BATCH_SIZE):
    
    samples, tags = shuffle(samples, tags)

    assert len(samples) == len(tags)
    
    while 1:

        for offset in range(0, len(samples), batch_size):

            batch_samples, batch_tags = samples[offset:offset+batch_size], tags[offset:offset+batch_size]
            
            images = []
            steering_angles = []

            for bs in range(len(batch_samples)):
                '''
                print(batch_sample)
                p=input()
                '''
                an_image = cv2.cvtColor(cv2.imread(batch_samples[bs]), cv2.COLOR_BGR2RGB)
                
                images.append(  an_image  )
                steering_angles.append( batch_tags[bs] )
                
                if len(images) >= BATCH_SIZE:
                    break

                if coinFlip < PERCENTAGE_FLIPPED:
                    flipped_image,flipped_angle = flipImages(an_image,batch_tags[bs] )                    
                    
                    images.append( flipped_image )
                    steering_angles.append( flipped_angle )
                
                if len(images) >= BATCH_SIZE:
                    break

                if coinFlip < BRIGHTNESS_CHANGES:
                    brightness_changed_image = change_brightness(an_image)                    
                    images.append( brightness_changed_image  )
                    steering_angles.append( batch_tags[bs] )
            
            images = np.array(images)
            steering_angles = np.array(steering_angles)

            if len(images) >= BATCH_SIZE:
                yield images[:BATCH_SIZE], steering_angles[:BATCH_SIZE]
                samples, tags = shuffle(samples, tags)

data = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)

image_paths = []
steering_angles = []

for img_file_label in data:
    center = './data/IMG/'+ img_file_label[0].split('/')[-1]
    left = './data/IMG/'+ img_file_label[1].split('/')[-1]
    right = './data/IMG/'+ img_file_label[2].split('/')[-1]

    if center == "./data/IMG/center":
        continue

    center_angle = float(img_file_label[3])
    left_angle = float(img_file_label[3]) + STEER_CORRECTION
    right_angle = float(img_file_label[3]) - STEER_CORRECTION

    #steering angles
    coinFlip = random.random()
    if isclose(center_angle, 0.0):
        if coinFlip > .7:
            image_paths.extend( (center,left, right ) )
            steering_angles.extend( (center_angle, left_angle, right_angle) )
    else:
        image_paths.extend( (center,left, right ) )
        steering_angles.extend( (center_angle, left_angle, right_angle) )
    
image_paths = np.array(image_paths)
steering_angles = np.array(steering_angles)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
train_samples, test_samples, train_angles, test_angles = train_test_split(image_paths, steering_angles, test_size=0.2, random_state=42)

train_generator = generator(train_samples, train_angles, batch_size=BATCH_SIZE)
validation_generator = generator(test_samples, test_angles, batch_size=BATCH_SIZE)

model = Sequential()
model.add(Lambda(lambda x: x/255.5 - 0.5, input_shape=(height, width, 3)))

model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation=ACTIVATION_FUNCTION))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation=ACTIVATION_FUNCTION  ))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation=ACTIVATION_FUNCTION))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(2,2), activation=ACTIVATION_FUNCTION))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation=ACTIVATION_FUNCTION))

model.add(Flatten())

model.add(Dense(100, activation=ACTIVATION_FUNCTION))
model.add(Dense(50, activation=ACTIVATION_FUNCTION))
model.add(Dense(10, activation=ACTIVATION_FUNCTION))

#output layer... important !
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=.2,shuffle=True, epochs=EPOCHS, verbose=1)

model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(test_samples), nb_epoch=EPOCHS, steps_per_epoch=len(train_samples)//BATCH_SIZE)

model.save('model.h5')

