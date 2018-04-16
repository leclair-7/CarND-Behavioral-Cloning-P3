import os, datetime, time, csv
from math import isclose
import random, importlib

import numpy as np
import cv2

import tensorflow as tf

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Model parameters 
STEER_CORRECTION = .25
PERCENTAGE_FLIPPED = .8
BRIGHTNESS_CHANGES = .8
BATCH_SIZE = 128
EPOCHS = 3
ACTIVATION_FUNCTION = 'elu'
PROBABILITY_SKIP_ZERO_STEERING_ANGLE = .7

def bing():
    '''
    Function makes a sound - to be used when model training/testing is complete

    This is actually crucial for a busy person, start the NN training, and then this function
    is called (and makes the sound) when training's done
    '''
    pygame_spec = importlib.util.find_spec('pygame')
    if pygame_spec is not None:
        import pygame
        pygame.mixer.init()
        soundObj = pygame.mixer.Sound('beep1.ogg')
        soundObj.play()
        time.sleep(2)
        soundObj.stop()

def flipImages(images, steering_angle):
    flipped = np.copy(np.fliplr(images))   
    return (flipped, -1 * steering_angle)

def change_brightness(image):
    img = np.copy(image)
    image1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1,dtype=np.float64)
    # uniform means all outcomes equally likely, 
    # defaults to [0,1)
    random_bright = .4 + np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    # Data conversion and array slicing schemes -> Reference: the amazing Vivek Yadav
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    #(np slicing) --> Put any pixel that was made greater than 255 back to 255
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def preprocess_image(image):
    '''
    This function does the following:
    1 - Crops the image
    2 - Blurs the image slightly
    3 - Resizes the image to 64 across 64 up/down
    4 - Converts the image to YUV
    '''
    x,w = 25,image.shape[1]-25    
    #left to cut off, right to cut off
    y,h = 65, (image.shape[0] - 90)
    image = np.copy(image[y:y+h, x:x+w])
    image = cv2.GaussianBlur(image, (3,3), 0)    
    image = cv2.resize(image, (64,64),interpolation=cv2.INTER_AREA )
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
                an_image = cv2.imread(batch_samples[bs])
                
                #crop, blur, resize, convert to YUV
                an_image = preprocess_image(an_image)

                images.append(  an_image  )
                steering_angles.append( batch_tags[bs] )
                
                if len(images) >= BATCH_SIZE:
                    break

                #Adds any image with a  
                if abs(batch_tags[bs]) > .2:
                    flipped_image,flipped_angle = flipImages(an_image,batch_tags[bs] )                    
                    
                    images.append( flipped_image )
                    steering_angles.append( flipped_angle )
                
                if len(images) >= BATCH_SIZE:
                    break

                #Add a copy of each datapoint with different brightness
                # Helps the car drive in shadow situations and generalize
                brightness_changed_image = change_brightness(an_image)                    
                images.append( brightness_changed_image  )
                steering_angles.append( batch_tags[bs] )
            
            images = np.array(images)
            steering_angles = np.array(steering_angles)

            if len(images) >= BATCH_SIZE:
                yield images[:BATCH_SIZE], steering_angles[:BATCH_SIZE]
                samples, tags = shuffle(samples, tags)

def make_model():
    '''
    Outputs the model that we will use to control the car and predict the steering angle
    '''
    model = Sequential()
    #normalize and set input shape
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(64,64, 3)))

    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation=ACTIVATION_FUNCTION))
    #model.add(MaxPooling2D())
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation=ACTIVATION_FUNCTION  ))
    #model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation=ACTIVATION_FUNCTION))
    #model.add(MaxPooling2D())
    model.add(Dropout(.5))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation=ACTIVATION_FUNCTION))
    
    model.add(Flatten())

    model.add(Dense(100, activation=ACTIVATION_FUNCTION))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation=ACTIVATION_FUNCTION))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation=ACTIVATION_FUNCTION))

    #output layer... important !
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    return model

def load_train_data(using_custom):
    fp1 = './data/driving_log.csv'
    fp2 = './recovery_data/driving_log.csv'

    data = []
    
    fileToUpload = fp2 if using_custom else fp1
    with open(fileToUpload) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            data.append(line)

    image_paths = []
    steering_angles = []

    for img_file_label in data:    
           
        if using_custom:
            center = img_file_label[0].split('/')[-1]
            left = img_file_label[1].split('/')[-1]
            right = img_file_label[2].split('/')[-1]
        else:
            center = './data/IMG/'+ img_file_label[0].split('/')[-1]
            left = './data/IMG/'+ img_file_label[1].split('/')[-1]
            right = './data/IMG/'+ img_file_label[2].split('/')[-1]

        '''
        print(center)
        print(left)
        print(right)
        pause = input()
        '''
        
        if center == "./data/IMG/center":
            continue

        center_angle = float(img_file_label[3])
        left_angle = float(img_file_label[3]) + STEER_CORRECTION
        right_angle = float(img_file_label[3]) - STEER_CORRECTION
        
        coinFlip = random.random()
        if isclose(center_angle, 0.0):
            PROBABILITY_SKIP_ZERO_STEERING_ANGLE = .7 if using_custom else .7
            
            if coinFlip > PROBABILITY_SKIP_ZERO_STEERING_ANGLE:
                image_paths.extend( (center,left, right ) )
                steering_angles.extend( (center_angle, left_angle, right_angle) )
        else:
            image_paths.extend( (center,left, right ) )
            steering_angles.extend( (center_angle, left_angle, right_angle) )
        
    image_paths = np.array(image_paths)
    steering_angles = np.array(steering_angles)

    return image_paths, steering_angles

def train_model():
    '''
    This function trains the model and outputs the .h5 file

    A .h5 file saves the following (from the keras documentation):
        - the architecture of the model, allowing to re-create the model
        - the weights of the model
        - the training configuration (loss, optimizer)
        - the state of the optimizer, allowing to resume training exactly where you left off

    This function exploits this by saving the model and then training on the custom data.
    Since this data didn't always load successfully, it's good to save the first model as a checkpoint

    Given the amount of time for training, I have the program make a sound when it's done;
    a polling mechanism for humans
    '''
    for i in range(2):

        image_paths, steering_angles = None, None

        if i == 0:
            EPOCHS = 5
            image_paths, steering_angles = load_train_data(False)
            model = make_model()
        elif i == 1:
            EPOCHS = 3
            image_paths, steering_angles = load_train_data(True)
            model = load_model('model_udacity_only.h5')

        train_samples, test_samples, train_angles, test_angles = train_test_split(image_paths, steering_angles, test_size=0.2, random_state=42)
        train_generator = generator(train_samples, train_angles, batch_size=BATCH_SIZE)
        validation_generator = generator(test_samples, test_angles, batch_size=BATCH_SIZE)

        model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//BATCH_SIZE, \
        epochs=EPOCHS, \
        validation_data=validation_generator, validation_steps=len(test_samples)//BATCH_SIZE,\
        verbose = 1
         )

        if i ==0:
            model.save('model_udacity_only.h5')
        elif i==1:
            model.save("model.h5")
    bing()     

if __name__=='__main__':    
    train_model()