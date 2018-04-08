
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.backend import tf as ktf
from keras.preprocessing.image import ImageDataGenerator
import time
import tensorflow as tf
import sys
import cv2
import numpy as np
import csv
import random
from math import isclose
EPOCHS = 7
STEER_CORRECTION = .25



#starts out at 160 x 320
height = 160
width = 320
NEW_HEIGHT = 120
NEW_WIDTH = 240
num_channels = 3
PERCENTAGE_FLIPPED = .7
BRIGHTNESS_CHANGES = .6

data = []
with open('./data/driving_log.csv') as csv_in:
	reader = csv.reader(csv_in)
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

def readDataIn(data):
    images = []
    steering_angles = []
    
    for i in range(1,len(data)):
        
        data_point = data[i]

        name = './data/IMG/'+ data_point[0].split('/')[-1]
        left = './data/IMG/'+ data_point[1].split('/')[-1]
        righ = './data/IMG/'+ data_point[2].split('/')[-1]
        
        #print(len(images), name)
        if name == "./data/IMG/center":
            continue
            
        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        left_image  = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(cv2.imread(righ), cv2.COLOR_BGR2RGB)

        #steering angles
        center_angle = float(data_point[3])
        left_angle = float(data_point[3]) + STEER_CORRECTION
        right_angle = float(data_point[3]) - STEER_CORRECTION
        
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
      
    return [images, steering_angles]

print("got here..")    
images, steering_angles = readDataIn(data)
X_train = np.array(images)
y_train = np.array(steering_angles)
print("data uploaded", len(X_train))

'''

#X_train = [ktf.image.resize_images(image, (NEW_HEIGHT, NEW_WIDTH) ) for image in images]

#lucas = input()

import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
'''


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

model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2),
                 activation=ACTIVATION_FUNCTION))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(2,2),
                 activation=ACTIVATION_FUNCTION))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
                 activation=ACTIVATION_FUNCTION))
model.add(Flatten())

model.add(Dense(1184, activation=ACTIVATION_FUNCTION))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))

#output the steering angle
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=.2,shuffle=True, epochs=EPOCHS, verbose=1)

model.save('model.h5')


'''
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(X_train), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''