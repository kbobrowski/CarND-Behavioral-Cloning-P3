**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/center.jpg "Center driving"
[image2]: ./pictures/recovery1.jpg "Recovery Image"
[image3]: ./pictures/recovery2.jpg "Recovery Image"
[image4]: ./pictures/recovery3.jpg "Recovery Image"
[image5]: ./pictures/hist1.png "Histogram"
[image6]: ./pictures/hist2.jpg "Histogram"

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

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 128 (model.py lines 89-108) 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 90). 

#### 2. Attempts to reduce overfitting in the model

Although overfitting was small, (training loss slightly smaller than validation loss), dropout layers were introduced to reduce it further. Model was also trained on a track driven in an opposite direction.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so initially the learning rate was not tuned manually (model.py line 108). For fine-tuning of the model (e.g. after introducing new data) learning rate was decreased to 0.0001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a well-known architecture and initial data set, and then refine the architecture and collect more data points based on the results from the previous model. Whenever possible, previous model was used as a starting point for the next optimization.

My first step was to use a convolution neural network model similar to the NVIDIA architecture for self-driving cars, which was designed for similar case.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

To combat the overfitting, I modified the model so that it includes dropout.

The final step was to run the simulator to see how well the car was driving around track one. The driving was not smooth, and there were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected additional data, including standard laps and recovery laps (for the latter case recording was only active while returning to the center of the lane). I also increased depth of the last CNN layer to 128. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, with maximum speed, as can be seen in the following video:

[![video](https://img.youtube.com/vi/wFuqlYk8aHU/0.jpg)](https://www.youtube.com/watch?v=wFuqlYk8aHU)

Moreover, it is quite robust to recover after manual interruption:

[![video](https://img.youtube.com/vi/M23Jr4QRMnQ/0.jpg)](https://www.youtube.com/watch?v=M23Jr4QRMnQ)

#### 2. Final Model Architecture

The final model architecture (model.py lines 89-108) consisted of a convolution neural network with the following layers and layer sizes (0.5 dropout was applied to all the layers):

1) 160x320x3 normalization layer
2) 66x320x3 cropping layer
3) 31x158x24 convolution layer (5x5 filter, 2x2 subsampling)
4) 14x77x36 convolution layer (5x5 filter, 2x2 subsampling)
5) 5x37x48 convolution layer (5x5 filter, 2x2 subsampling)
6) 3x35x64 convolution layer (3x3 filter, 1x1 subsampling)
7) 1x33x128 convolution layer (3x3 filter, 1x1 subsampling)
8) 1x4224 fully-connected layer
9) 1x256 fully-connected layer
10) 1x64 fully-connected layer
11) 1x1 output layer

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![recovery1][image2]

![recovery2][image3]

![recovery3][image4]

Plotting histogram of the steering angle, it is clear that majority of the samples are straight driving, which may introduce bias:

![hist1][image5]

To decrease small steering angles, samples were filtered with a condition to exclude 50% of the steering angles below 0.05, which resulted in the following histogram:

![hist2][image6]

To augment the data sat, flipping images and angles was introduced, as it produces valid sample. Flipping is done randomly within the generator.

After the collection process, I had 45555 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. Preprocessing consisted of normalizing and cropping the image. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by increase of validation loss after 5 epochs. For the initial training I used an adam optimizer so that manually training the learning rate wasn't necessary. Subsequent refinements (e.g. including more data) used learning rate of 0.0001.
