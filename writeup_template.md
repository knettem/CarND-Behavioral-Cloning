# **Behavioral Cloning** 

**Behavioral Cloning Project**

The main objective of this project is to clone human driving behavior using a Deep neural network. To achieve this we need to use a Car simulator provided by Udacity.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Center.jpg "center_image"
[image2]: ./examples/center_recovery.jpg "Recovery Image"
[image3]: ./examples/right_recovery1.jpg "Recovery Image"
[image4]: ./examples/right_recovery2.jpg "Recovery Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 recoreded video in autonomous mode. which is based on images created by the model.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I used the generator code which were provided in the class room to load and preprocess the data on the fly. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 convolutional neural networks with filter sizes ranging from 3x3 to 5x5. (model.py lines 50-54) 
All the convolutional layers include RELU activation layers to introduce nonlinearity. (model.py lines 50-54) and the data is normalized in the model using a Keras lambda layer (code line 50 in model.py). 

I used image cropping keras Cropping2D layer where the image cropped to include the area consisiting of the road and road signs. 

#### 2. Attempts to reduce overfitting in the model

Dropout layers added in between Actiavation layers values ranging rom 0.2 to 0.5 in order to reduce overfitting but the vehicle wentoff each time dropout layer was added. So, finally i removed dropout layer from my model. The model architecture i used NVIDIA model and it doesnt use dorpouts.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 68). The Adam optimizer is good because it automatically adjusts the learning rate over epochs.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVIDIA Architecture. I thought this model might be appropriate because it was implemented in keras. This model should work well without adding extra layers to model like dropout and maxpooling.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added more data to process the model. Collecting more data can help improve a model when the model is overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track epsecially around the bridges and sharp corners to improve the driving behavior in these cases, I used recovery lap to recover the car from sides and sharp turns. Added more dataset for these cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

Layer 1: adds the normalization using Keras lamda layer.
Layer 2: has the cropped image layer.
Layer 3: Convolutional layer with kernal 5x5, depth as 24 , strides 2,2 and RELU activation layer
Layer 4: Convolutional layer with kernal 5x5 ,depth as 36 , strides 2,2 and RELU activation layer
Layer 5: Convolutional layer, kernal 5x5 , depth as 48, strides 2,2 and RELU activation layer
Layer 6: Convolutional layer, kernal 3x3 , depth as 64 and RELU activation layer
Layer 7: Convolutional layer, kernal 3x3 , depth as 64 and RELU activation layer
Layer 8: Flattening layer
Layer 9: Fully connected layer
Layer10: Fully connected layer
Layer11: Fully connected layer
Output:  Dense(1)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay in middle of the road. The below images show what a recovery looks like starting from the right side

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.
Some samples are not recorded when the car moving to out of line in recovery lap so that we can avoid the bad behaviour into the model.

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by constant value of validatin loss. I used an adam optimizer so that manually training the learning rate wasn't necessary. The generator was set to take batches of 32 samples and model.fit_generator is configured. 