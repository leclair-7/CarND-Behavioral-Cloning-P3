import cv2
import numpy as np
import csv

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
	center_image = cv2.imread('./data/IMG/' + centImage_path)
	left_image   = cv2.imread('./data/IMG/' + leftImage_path)
	right_image  = cv2.imread('./data/IMG/' + righImage_path)
	image_flipped = np.copy(np.fliplr(center_image))
	
	images.extend((center_image, left_image, right_image, image_flipped))	
	
	steering_angle = float(line[3])
	#add a correction var?
	steering_angle_flipped = -steering_angle
	steering_angles.extend( (steering_angle,steering_angle,steering_angle,steering_angle_flipped) )

X_train = np.array(images)
y_train = np.array(steering_angles)


'''
Since this is a regression model minimize distance between how we drove the car
and what this model does 

'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D

model = Sequential()
model.add(Lambda(lambda x: x/255.5 - 0.5, input_shape=(160, 320, 3)))

model.add(Cropping2D(((50,20),(0,0)), input_shape=(160,320,3)))

model.add(Conv2D(24, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))

model.add(Conv2D(36, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'  ))

model.add(Conv2D(48, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(2,2),
                 activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
                 activation='relu'))
model.add(Flatten())
#model.add(Dense(1124, activation='relu'))

model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))

#output the steering angle
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=.2,shuffle=True, epochs=7)

model.save('model.h5')




