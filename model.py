
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.backend import tf as ktf
from keras.preprocessing.image import ImageDataGenerator
import time
import tensorflow as tf
#Commandline input if we want a histogram of steering angles
import sys
import cv2
import numpy as np
import csv

EPOCHS = 7
STEER_CORRECTION = .2

'''
training_file = "train.p"
validation_file= "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
'''

#starts out at 160 x 320
height = 160
width = 320
NEW_HEIGHT = 120
NEW_WIDTH = 240
num_channels = 3

lines = []

with open('./data/driving_log.csv') as csv_in:
	reader = csv.reader(csv_in)
	for line in reader:
		lines.append(line)

lines = lines

images = []
steering_angles = []
for line in lines:
	
	centImage_path = line[0].split('\\')[-1]
	leftImage_path = line[1].split('\\')[-1]
	righImage_path = line[2].split('\\')[-1]
	
	#current_path = './data/IMG/' + filename
	#cv2.imread reads it in as BGR...
	center_image = cv2.imread('./data/IMG/' + centImage_path)
	left_image   = cv2.imread('./data/IMG/' + leftImage_path)
	right_image  = cv2.imread('./data/IMG/' + righImage_path)
	image_flipped = np.copy(np.fliplr(center_image))
	
	images.extend((center_image, left_image, right_image, image_flipped))	
	
	steering_angle = float(line[3])
	
	#add a correction var?
	steering_angle_flipped = -steering_angle
	left_steering_angle = steering_angle + STEER_CORRECTION
	right_steering_angle = steering_angle - STEER_CORRECTION
	steering_angles.extend( (steering_angle,left_steering_angle,right_steering_angle,steering_angle_flipped) )
image = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in images]
X_train = np.array(images)
y_train = np.array(steering_angles)

'''
'''
from matplotlib import pyplot
x = np.arange(-1.5, 1.5, .01)
pyplot.hist(y_train,x)
pyplot.title('Steering Angles Histogram')
pyplot.show()

p = input("graph")

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

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
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

'''
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2),
                 activation='elu'))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(2,2),
                 activation='elu'))
'''
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
                 activation='elu'))
model.add(Flatten())
#model.add(Dense(1124, activation='elu'))

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