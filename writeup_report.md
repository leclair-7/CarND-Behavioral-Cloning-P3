# **Behavioral Cloning**

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/recovery.png "Recovery"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/dataset.png "Steering Angles"
[image5]: ./examples/cropped-notcropped.png "Cropped not Cropped"
[image6]: ./examples/flipped-notflipped.png "Flipped Not Flipped"
[image7]: ./examples/brightness-original.png "Brightness Changed"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator, my drive.py file, and the model.h5 file the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The final model used consists of a convolutional neural network which takes in an input shape of 64 x 64 X 3, normalizes it using in a keras lambda layer followed by convolutional layers, drop out layers, and a layer that flattens the inputs to a 1-D array ending with fully connected layers.

I decided to use the ELU activation function because the literature suggests it works better for this purpose however in practice it appears to perform similarly to the RELU activation function.

#### 2. Attempts to reduce overfitting in the model

Over fitting was reduced by adding drop out layers in various places (see architecture section). Another way over fitting was reduced was the data augmentation was made to randomly distort images which led to a seemingly infinite number of images being send to the model.

#### 3. Model parameter tuning

The model used an Adam optimizer, with the learning rate reduced from 10**-3 to 10**-4. This was found when a blog post said the first thing to try to improve machine learning is to lower the learning rate.

#### 4. Appropriate training data

The training data used was the class supplied data which utilized the center, left, and right cameras. Each steering angle on the left had a steering correction number constant added to it, and on the right it was subtracted. Data was collected for recoveries (meaning the car gets close to driving into the lane or worse into the lake). One lap was driven in the opposite direction to help with model generalization.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first step was to use the convolutional network in the NVidia paper, “End to End Learning for Self-Driving Cars (NVidias)”. This model seemed appropriate because it answers the same question as this project's prompt. After training with the model it appeared to over fit tremendously, training times were slow, and most importantly, the car was driving into the lake.

To gauge the model’s progress as it was training, I split the data into a 10% validation set / 90% training set. I found that the loss reached its minimum after about five epochs. When I ran train the model for many epochs, on the order of 40, I noticed that when the car was driving, it changed steering angles many times per second. This appears to be what over-fitting looks like for this task. The model was too complex (which goes hand-in-hand with over-fitting). As a result, I reduced model complexity by removing some layers and reducing other layers' sizes. Removing two of the convolutional layers greatly decreased training time. Parameter tuning involved copious amounts of trial and error. During the trial and error the image size was reduced to decrease training time, cropping was done to use the most relevant parts of the image, the road.

![alt text][image5]

To combat the over-fitting, I added dropout layers.

Then I lowered the training rate from 10^-3 to 10^-4. This made the car drive smoother and made a change in the number of epochs more noticeable.

Initially the training runs that consisted of the class supplied data set resulted in satisfactory results that is, almost successful. However, the car touched the line after two different curves. After trying drop out layers in different places, epoch numbers, and pooling layers, I decided to do recovery laps. This consists of recording the car when it is in an undesirable place such as close to a lane marker, and then turning the car back to the center of the lane. This was done for two laps, and then I recorded one lap of the car going counterclockwise around the race course. This was to help the car generalize and also to make the curve drives smoother. The recovery laps trained model failed on submission which led me on the seemingly more robust approach of randomly augmenting each image before it is sent to the model. This is described at the beginning of section 3.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (created by the make_model() function in model.py) consisted of a convolution neural network with the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input / normalize 	| 64x64x3 image    								|
| Convolution 8x8x32   	| 4x4 stride, Same Padding 					|
| Convolution 8x8x64   	| 4x4 stride, Same Padding 					|
| Convolution 4x4x64   	| 2x2 stride, Same Padding						|
| Convolution 2x2x128  	| 1x1 stride, Same Padding 					|
| Flatten				| flatten to a 1D vector						|
| Dropout      			| keep probability = .5							|
| Fully connected		| 128 outputs 									|
| ELU	      			| 												|
| Dropout      			| keep probability = .5							|
| Fully connected		| 128 outputs 									|
| Fully connected		| 1 output (Steering Angle) 					|


#### 3. Creation of the Training Set & Training Process

To augment the data I randomly translated the image about the x axis and y axis), randomly cropped the image, randomly changed the brightness, and flipped half of the images. In total I had 7,444 images of the car driving counterclockwise. Surprisingly enough, this is less than the 24,109 image files in the given training data.

Since most frames involve a steering angle = 0, the data at the outset was heavily biased toward making the car steer straight. This was corrected for by use of the cropping function which changes the steering angle based on how much it changed the input image. This lets the model learn how to steer curves synthetically instead of the laborious task of collecting recovery data. Recovery data was attempter with keyboard controls and by using the mouse, neither of which worked. In my first submission, I decided to disregard about 70% of the image whose steering angles was 0. I then collected data/increased model complexity, used multiple weekends of effort. The first approach did not work except in very specific situations.  

The dataset without the zero steering angles removed is displayed below. The advantage of changing the steering angle with respect to how the crop manipulation is generated is that the model learns more situations where it is close to the lane line.
![alt text][image4]

Brightness Change:
![alt text][image7]

After each batch, the data was reshuffled in the generator function. The generator function used is a lazy evaluator; it makes the training data on demand. When a generator was not used the computer crashed due to either the RAM being full or on occasion, the GPU gives out of memory error.
