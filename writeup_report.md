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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The final model used consists of a convolution on network which takes in an input shape of 64 x 64 X 3, normalizes it in a keras lambda layer followed by three convolutional layers, and drop out layer, another convolutional later, the array is then flattened and a fully connected three fully connected layers follow.

I decided to use the ELU activation function however, it did not do better than a RELU layer.

#### 2. Attempts to reduce overfitting in the model

Over fitting was reduced in this model by adding a drop out layers in various places (see architecture section). Another way overfitting was reduced was to tune the file upload process. That is, only 30% of the images with steering angle equal to zero were allowed to be uploaded. This made the data more balanced while reducing hard drive accesses.

#### 3. Model parameter tuning

The model used an adam optimizer, with the learning rate 10**-4. This was found when a blog post said the first thing to try to improve machine learning is to lower the learning rate.

#### 4. Appropriate training data

The training data used was the class supply data, or utilize the center lane, and left and right sides of the road, each steering angle in the left had a steering correction number constant added to it, and on the right it was subtracted. I also collected manual data for recoveries and I drove one lap in the opposite direction to help the model generalize.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first step was to use the convolutional network as used in the Nvidia paper, “End to End Learning for Self-Driving Cars (Nvidia)”. This model seemed appropriate because it answers the same question as the project prompt. After beginning to train with the model it appeared to overfit tremendously, training times were slow, and most importantly, the car was driving into the lake.

To gauge the model’s progress as it was training, I split the data into a 20% validation set an 80% training set. I found that the loss reach its minimum after about five epochs. When I ran train the model for many epochs, on the order of 40, I noticed that when the car was driving, it changed steering angles many times per second. This appears to be what overfitting looks like for this task, the model being too complex (which goes hand-in-hand with overfitting). As a result, I reduced the image size, experimented with cropping until what seem like the most important parts of the image were maintained. The model was reduced in size because a convolution with stride = (2,2) reduces the size the length and width of the layer’s output.

I also used a function that changes the brightness of every image randomly, on some images I created a copy (these were only ones with a considerable turning steering angle) which is greater than .2.

To combat the overfitting, I added the dropout layers.

Then I lowered the training rate from 10^-3 to 10^-4. This made the car drive smoother and made a change in the number of epochs more noticable.

After

Initially the trainings that consisted of the class supply data set, this resulted in satisfactory results, however, the car touch the lien lime after two different curves. After testing different drop out layers, epic numbers, and pulling layers, I decided to do recovery laps. This consists of recording the car when it is in an undesirable place such as close to a lane marker, or on Elaine marker, and then turning the car back to the center of the lane. This was done for two laps, and then I recorded one lap of the car going counterclockwise around the race course. This was to help the car generalize and also to make the curves drive smoother.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (created by the make_model() function in model.py) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input / normalize 	| 64x64x3 image    								|
| Convolution 5x5x24   	| 2x2 stride, Valid Padding 					|
| Convolution 5x5x36   	| 2x2 stride, Valid Padding 					|
| Convolution 3x3x64   	| 2x2 stride, Valid Padding						|
| Dropout      			| keep probability = .5							|
| Convolution 3x3x64  	| 2x2 stride, Valid Padding 					|
| Flatten				| flatten to a 1D vector						|
| Fully connected		| 100 outputs 									|
| ReLU	      			| 												|
| Dropout      			| keep probability = .5							|
| Fully connected		| 50 outputs 									|
| ReLU	      			| 												|
| Dropout      			| keep probability = .5							|
| Fully connected		| 10 outputs 									|
| Fully connected		| 1 output (Steering Angle) 					|

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To augment the data set in the generator function I attended an image copy with the brightness modified by random amount, and on more higher numbered steering Ingles, I flip the image so the model would be train with the hair representation of situations where it needs to steer at a higher angle such as when it gets close to a curve. Each image was loaded, cropped, blurred, resized to 64 x64, and then converted to YUV. In total I had 25,740 recovery images. Surprisingly enough, this is more than the 24,109 image files in the given training data.

After each batch, the data was reshuffled in the generator function. It generator function was used which is a lazy evaluator and it makes the training data on demand. When a generator was not use the computer would crash do to either the RAM being full or on occasion, the GPU gives out of memory error.


To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]


After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set.
