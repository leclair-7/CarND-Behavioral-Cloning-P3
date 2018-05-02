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
[image4]: ./examples/Steering-angle-filters.png "Steering Angles"
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

Over fitting was reduced by adding drop out layers in various places (see architecture section). Another way over fitting was reduced was to tune the file upload process. That is, only 30% of the images with steering angle equal to zero were allowed to be uploaded. This made the data more balanced while reducing hard drive accesses.

#### 3. Model parameter tuning

The model used an Adam optimizer, with the learning rate 10**-4. This was found when a blog post said the first thing to try to improve machine learning is to lower the learning rate.

#### 4. Appropriate training data

The training data used was the class supplied data which utilized the center, left, and right cameras. Each steering angle on the left had a steering correction number constant added to it, and on the right it was subtracted. Data was collected for recoveries (meaning the car gets close to driving into the lane or worse into the lake). One lap was drawn in the opposite direction to help with model generalization.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first step was to use the convolutional network in the NVidia paper, “End to End Learning for Self-Driving Cars (NVidias)”. This model seemed appropriate because it answers the same question as this project's prompt. After training with the model it appeared to over fit tremendously, training times were slow, and most importantly, the car was driving into the lake.

To gauge the model’s progress as it was training, I split the data into a 20% validation set / 80% training set. I found that the loss reached its minimum after about five epochs. When I ran train the model for many epochs, on the order of 40, I noticed that when the car was driving, it changed steering angles many times per second. This appears to be what over-fitting looks like for this task. The model was too complex (which goes hand-in-hand with over-fitting). As a result, I reduced model complexity by removing some layers and reducing other layers' sizes. Removing two of the convolutional layers greatly decreased training time. Parameter tuning involved copious amounts of trial and error. During the trial and error the image size was reduced to decrease training time, cropping was done to use the most relevant parts of the image, the road.

![alt text][image5]

To combat the over-fitting, I added dropout layers.

Then I lowered the training rate from 10^-3 to 10^-4. This made the car drive smoother and made a change in the number of epochs more noticeable.

Initially the training runs that consisted of the class supplied data set resulted in satisfactory results that is, almost successful. However, the car touched the line after two different curves. After trying drop out layers in different places, epoch numbers, and pooling layers, I decided to do recovery laps. This consists of recording the car when it is in an undesirable place such as close to a lane marker, and then turning the car back to the center of the lane. This was done for two laps, and then I recorded one lap of the car going counterclockwise around the race course. This was to help the car generalize and also to make the curve drives smoother.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (created by the make_model() function in model.py) consisted of a convolution neural network with the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input / normalize 	| 64x64x3 image    								|
| Convolution 5x5x24   	| 1x1 stride, Valid Padding 					|
| Convolution 5x5x36   	| 2x2 stride, Valid Padding 					|
| Convolution 3x3x64   	| 2x2 stride, Valid Padding						|
| Dropout      			| keep probability = .5							|
| Convolution 3x3x64  	| 2x2 stride, Valid Padding 					|
| Flatten				| flatten to a 1D vector						|
| Fully connected		| 100 outputs 									|
| ELU	      			| 												|
| Dropout      			| keep probability = .5							|
| Fully connected		| 50 outputs 									|
| ELU	      			| 												|
| Dropout      			| keep probability = .5							|
| Fully connected		| 10 outputs 									|
| Fully connected		| 1 output (Steering Angle) 					|


#### 3. Creation of the Training Set & Training Process

To augment the data I appended an image copy with the brightness modified by random amount, and on more higher numbered steering angles, I flipped the image so the model would be trained with the higher number of images where it needs to steer at a higher angle such as when it gets close to a curve. Each image was loaded, cropped, blurred, resized to 64 x64, and then converted to YUV. In total I had 25,740 recovery images. Surprisingly enough, this is more than the 24,109 image files in the given training data.

Since most frames involve a steering angle = 0, the data at the outset was heavily biased toward making the car steer straight. Given this fact, I decided to disregard about 70% of the image whose steering angles was 0. Below is a plot of the datasets that were ultimately sent to the generator to generate batches

![alt text][image4]

I also used a function that changes the brightness of every image randomly, on some images I created a copy (these were only ones with a considerable turning steering angle) which is greater than .2.

![alt text][image7]

After each batch, the data was reshuffled in the generator function. It generator function was used which is a lazy evaluator and it makes the training data on demand. When a generator was not use the computer would crash do to either the RAM being full or on occasion, the GPU gives out of memory error.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from being on the left lane to being back in the center.

![alt text][image1]
