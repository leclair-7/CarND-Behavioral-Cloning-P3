from keras.models import load_model
import os, datetime, time, csv, random, importlib
from math import isclose

import numpy as np
import cv2

import tensorflow as tf
from keras import regularizers

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, MaxPooling2D,Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
'''
After spending a ton of time on the car driving into the lake, we turn to
https://github.com/ksakmann/CarND-BehavioralCloning/blob/master/README.md
and use the following tips:

(already doing this) A random training example is chosen
The camera (left,right,center) is chosen randomly
Random shear: the image is sheared horizontally to simulate a bending road
(sort of doing this) Random crop: we randomly crop a frame out of the image to simulate the car being offset from the middle of the road (also downsampling the image to 64x64x3 is done in this step)
(sort of doing this) Random flip: to make sure left and right turns occur just as frequently
(sort of doing this) Random brightness: to simulate differnt lighting conditions

'''

# Model parameters 
STEER_CORRECTION = .25
PERCENTAGE_FLIPPED = .8
BRIGHTNESS_CHANGES = .8
BATCH_SIZE = 256
EPOCHS = 3
ACTIVATION_FUNCTION = 'elu'
PROBABILITY_SKIP_ZERO_STEERING_ANGLE = 0.0

def flipImages(images, steering_angle):
    flipped = np.copy(np.fliplr(images))   
    return (flipped, -1 * steering_angle)

def change_brightness(image):
    img = np.copy(image)
    image1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1,dtype=np.float64)
    # uniform means all outcomes equally likely, 
    # defaults to [0,1)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0) 
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    # Data conversion and array slicing schemes -> Reference: the amazing Vivek Yadav
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    #(np slicing) --> Put any pixel that was made greater than 255 back to 255
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image,steer):
    '''
    We decide to use shearing in the preprocessing
    '''
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    # https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713
    trans_range_x = 115
    trans_range_y = 40
    translate_x = trans_range_x * np.random.uniform() - trans_range_x/2
    translate_y = trans_range_y * np.random.uniform() - trans_range_y/2

    steer_ang = steer + (translate_x/trans_range_x * 2 * .2)    
    
    img2 = np.copy(image)
    #this better be 64x64x3
    rows,cols,channels = img2.shape

    Trans_M = np.float32([[1,0,translate_x],[0,1,translate_y]])
    image_tr = cv2.warpAffine(img2,Trans_M,(cols,rows))
    
    return image_tr,steer_ang

def preprocess_image(image):
    '''
    This function does the following:
    1 - Crops the image
    2 - Blurs the image slightly
    3 - Resizes the image to 64 across 64 up/down
    4 - Converts the image to YUV
    '''
    # top to cut off, bottom to cut off
    #x,w = 30,image.shape[1]-30
    
    #left to cut off, right to cut off
    #y,h = 55, (image.shape[0] - 90)
    #image = np.copy(image[y:y+h, x:x+w])
    #image = cv2.GaussianBlur(image, (3,3), 0)    
    #image = cv2.resize(image, (64,64),interpolation=cv2.INTER_AREA )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def random_crop(image,steering=0.0,tx_lower=-20,tx_upper=20,ty_lower=-2,ty_upper=2,rand=True):
    # we will randomly crop subsections of the image and use them as our data set.
    # also the input to the network will need to be cropped, but of course not randomly and centered.
    shape = image.shape
    col_start,col_end =abs(tx_lower),shape[1]-tx_upper
    horizon=60;
    bonnet=136
    if rand:
        tx= np.random.randint(tx_lower,tx_upper+1)
        ty= np.random.randint(ty_lower,ty_upper+1)
    else:
        tx,ty=0,0
    
    
    random_crop = image[horizon+ty:bonnet+ty,col_start+tx:col_end+tx,:]
    image = cv2.resize(random_crop,(64,64),cv2.INTER_AREA)
    # the steering variable needs to be updated to counteract the shift 
    if tx_lower != tx_upper:
        dsteering = -tx/(tx_upper-tx_lower)/3.0
    else:
        dsteering = 0
    steering += dsteering
    
    return image,steering

def bing():
    '''
    This function makes a sound - to be used when model training/testing is complete

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

def generator(samples, tags, bias, batch_size=BATCH_SIZE):
    
    samples, tags = shuffle(samples, tags)

    assert len(samples) == len(tags)
    
    while 1:

        for offset in range(0, len(samples), batch_size):            

            batch_samples, batch_tags = samples[offset:offset+batch_size], tags[offset:offset+batch_size]
            
            images = []
            steering_angles = []

            for bs in range(len(batch_samples)):              

                #pick center, left, or right randomly
                center_left_or_right = np.random.randint(3)


                an_image = preprocess_image( cv2.imread(batch_samples[bs][center_left_or_right]) )
                steering_angle = batch_tags[bs][center_left_or_right]

                


                threshold = np.random.uniform()
                '''
                if (abs(steering_angle) + bias) < threshold or abs(steering_angle) > 1.:
                    continue
                '''
                #an_image, steering_angle = trans_image(an_image, steering_angle)
                #if random.random() > .5:
                an_image, steering_angle = trans_image(an_image,steering_angle)

                an_image,steering_angle = random_crop(an_image,steering_angle)

                if random.random() > .5:
                    an_image, steering_angle = flipImages(an_image,steering_angle )

                
                #if random.random() > .5
                an_image = change_brightness(an_image)           
                          
                

                
                images.append(  an_image  )
                steering_angles.append( steering_angle )
                
                if len(images) >= BATCH_SIZE:
                    break
            
            images = np.array(images)
            steering_angles = np.array(steering_angles)

            #When we have enough images, we output a batch via yield
            if len(images) >= BATCH_SIZE:
                yield images[:BATCH_SIZE], steering_angles[:BATCH_SIZE]
                samples, tags = shuffle(samples, tags)
def validation_generator(samples, tags, batch_size=BATCH_SIZE):
    samples, tags = shuffle(samples, tags)

    assert len(samples) == len(tags)
    
    while 1:

        for offset in range(0, len(samples), batch_size):            

            batch_samples, batch_tags = samples[offset:offset+batch_size], tags[offset:offset+batch_size]
            
            images = []
            steering_angles = []

            for bs in range(len(batch_samples)):
                
                center_left_or_right = np.random.randint(3)

                an_image = preprocess_image( cv2.imread(batch_samples[bs][center_left_or_right]) )
                an_image = cv2.resize(an_image, (64,64),interpolation=cv2.INTER_AREA )

                steering_angle = batch_tags[bs][center_left_or_right]
                '''
                an_image = cv2.imread(batch_samples[bs])
                steering_angle = batch_tags[bs]               
                '''
                #crop, resize, convert to RGB
                #an_image = preprocess_image(an_image)
                

                images.append(  an_image  )
                steering_angles.append( steering_angle )
                
                if len(images) >= BATCH_SIZE:
                    break                               
            
            images = np.array(images)
            steering_angles = np.array(steering_angles)

            #When we have enough images, we output a batch via yield
            if len(images) >= BATCH_SIZE:
                yield images[:BATCH_SIZE], steering_angles[:BATCH_SIZE]
                samples, tags = shuffle(samples, tags)

def make_model():
    '''
    Outputs the model that we will use to control the car and predict the steering angle
    '''
    model = Sequential()
    #normalize and set input shape
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(64, 64, 3)))

    #model.add(Conv2D(3, kernel_size=(1, 1), strides=(1, 1), border_mode='same', name='color_conv') )
    model.add(Conv2D(32, kernel_size=(8,8), strides=(4,4),padding='same'))
    model.add(Activation(ACTIVATION_FUNCTION))
    model.add(Conv2D(64, kernel_size=(8,8), strides=(4,4),padding='same')) 
    model.add(Activation(ACTIVATION_FUNCTION))
    #model.add(Conv2D(45, kernel_size=(5,5), strides=(2, 2), kernel_regularizer=regularizers.l2(0.001)) )
    #model.add(Activation(ACTIVATION_FUNCTION))    
    model.add(Conv2D(128, kernel_size=(4,4), strides=(2,2),padding='same') )
    model.add(Conv2D(128, kernel_size=(2,2), strides=(1,1),padding='same') )
    model.add(Activation(ACTIVATION_FUNCTION))  
    
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(128) )
    model.add(Activation(ACTIVATION_FUNCTION))
    model.add(Dropout(0.5))

    
    model.add(Dense(128, activation=ACTIVATION_FUNCTION))
    
    #output layer... important !
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer=Adam(lr=1e-4, epsilon=1e-08), metrics=['accuracy'])
    return model

def load_train_data_folder(folder_number):
    '''
    This function puts the filepaths of the images, Class supplied
    or custom images, based on the using_custom parameter
    
    It returns an array of image filepaths and an associated array with the 
    the corresponding steering angles.

    Also it only keeps 70% of the images with steering angle = 0 because 
    it helps balance the data
    '''
    

    fp0 = './data/driving_log.csv'
    fp1 = './recovery_data/driving_log.csv'
    fp2 = './lap_counter_clock/driving_log.csv'
    fp3 = './challenge_course/driving_log.csv'
    
    folder_dict = {0:fp0, 1:fp1, 2:fp2}
    
    fileToUpload = folder_dict[folder_number]
    
    data = []
    
    with open(fileToUpload) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            data.append(line)

    image_paths = []
    steering_angles = []    

    for img_file_label in data[1:]:    
           
        if folder_number > 0:
            center = img_file_label[0].split('/')[-1]
            left = img_file_label[1].split('/')[-1]
            right = img_file_label[2].split('/')[-1]
        else:
            center = './data/IMG/'+ img_file_label[0].split('/')[-1]
            left = './data/IMG/'+ img_file_label[1].split('/')[-1]
            right = './data/IMG/'+ img_file_label[2].split('/')[-1]
        
        
        if center == "./data/IMG/center":
            continue
        

        center_angle = float(img_file_label[3])
        left_angle = float(img_file_label[3]) + STEER_CORRECTION
        right_angle = float(img_file_label[3]) - STEER_CORRECTION
        
        image_path_3 = (center, left, right ) 
        steering_3 = (center_angle,left_angle,right_angle)
        coinFlip = random.random()
        
        image_paths.append(image_path_3)
        steering_angles.append(steering_3)

    image_paths = image_paths
    steering_angles = steering_angles
    
    assert len(image_paths) == len(steering_angles)

    return image_paths, steering_angles

def load_train_data_folder_triples(folder_number):
    '''
    This function puts the filepaths of the images, Class supplied
    or custom images, based on the using_custom parameter
    
    It returns an array of image filepaths and an associated array with the 
    the corresponding steering angles.

    Also it only keeps 70% of the images with steering angle = 0 because 
    it helps balance the data
    '''
    

    fp0 = './data/driving_log.csv'
    fp1 = './recovery_data/driving_log.csv'
    fp2 = './lap_counter_clock/driving_log.csv'
    fp3 = './challenge_course/driving_log.csv'
    
    folder_dict = {0:fp0, 1:fp1, 2:fp2}
    
    fileToUpload = folder_dict[folder_number]
    
    data = []
    
    with open(fileToUpload) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            data.append(line)

    image_paths = []
    steering_angles = []    

    for img_file_label in data[1:]:    
           
        if folder_number > 0:
            center = img_file_label[0].split('/')[-1]
            left = img_file_label[1].split('/')[-1]
            right = img_file_label[2].split('/')[-1]
        else:
            center = './data/IMG/'+ img_file_label[0].split('/')[-1]
            left = './data/IMG/'+ img_file_label[1].split('/')[-1]
            right = './data/IMG/'+ img_file_label[2].split('/')[-1]
        
        
        if center == "./data/IMG/center":
            continue
        
        offset=1.0 
        dist=20.0    
        center_angle = float(img_file_label[3])
        left_angle = float(img_file_label[3]) + offset/dist * 360/( 2*np.pi) / 25.0
        right_angle = float(img_file_label[3]) - (offset/dist * 360/( 2*np.pi) / 25.0)
                   
        image_paths.append( (center, left, right ) )
        steering_angles.append( (center_angle, left_angle, right_angle) )

    image_paths = image_paths
    steering_angles = steering_angles
    
    assert len(image_paths) == len(steering_angles)

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
    image_paths, steering_angles = [], []

    for i in range(3):
        images_temp,steering_angles_temp = load_train_data_folder_triples(i)
        if i == 1:
            continue
        image_paths.extend(images_temp)
        steering_angles.extend(steering_angles_temp)
    
    image_paths = np.array(image_paths)
    steering_angles = np.array(steering_angles)

    
    EPOCHS = 6
    #model = make_model()    
    model = load_model("model_master6.h5")

    train_samples, validation_paths, test_samples, v_steering_angles = train_test_split(image_paths, steering_angles, test_size=0.1, random_state=42)
    
    #training set
    #train_generator = generator(train_samples, test_samples, bias, batch_size=BATCH_SIZE)
    
    #validation set
    val_generator = validation_generator(validation_paths, v_steering_angles, batch_size=BATCH_SIZE)


    print("Amount of datapoints in each set")
    for j in [train_samples, validation_paths, test_samples, v_steering_angles]:
        print(len(j))
    print()

    
    num_runs = 1
    #checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
    while 1:
        bias = 1. / (num_runs + 1.)

        model.fit_generator(generator(train_samples, test_samples, bias, batch_size=BATCH_SIZE), steps_per_epoch=80, \
        epochs=1, \
        validation_data=val_generator, validation_steps=10,\
        verbose = 1
        )
        '''
        #,
        #callbacks = [checkpoint]
        '''
        if num_runs >2 and num_runs%3==0:
            model.save("model_{}.h5".format(num_runs) )
        num_runs += 1
        if num_runs > EPOCHS:
            break

    model.save("model_master7.h5" )
    print( model.summary() )
    bing()     
start = time.time()
train_model()
print("It took {} to upload data and train".format(str(time.time()-start)))