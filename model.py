
#Commandline input if we want a histogram of steering angles
import sys

import cv2
import numpy as np
import csv

EPOCHS = 7
STEER_CORRECTION = .2

#starts out at 160 x 320
NEW_HEIGHT = 120
NEW_WIDTH = 240

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
from matplotlib import pyplot
x = np.arange(-1.5, 1.5, .02)
pyplot.hist(y_train,x)
pyplot.title('Steering Angles Histogram')
pyplot.show()
'''
'''
Since this is a regression model minimize distance between how we drove the car
and what this model does 
'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.backend import tf as ktf
'''
'''
images = [ktf.image.resize_images(image, (NEW_HEIGHT, NEW_WIDTH) ) for image in images]


model = Sequential()
model.add(Lambda(lambda x: x/255.5 - 0.5, input_shape=(160, 320, 3)))

model.add(Cropping2D(((50,20),(0,0)), input_shape=(160,320,3)))

#model.add(Lambda(lambda x: my_resize_function(x)))


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
model.fit(X_train, y_train, validation_split=.2,shuffle=True, epochs=EPOCHS)

model.save('model.h5')




