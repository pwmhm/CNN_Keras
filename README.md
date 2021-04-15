# CNN_Keras
Implementation of Keras API (Tensorflow 2.4.1) to classify images of dogs and cats

# Dependencies : <br/>
- SKLearn train-test-split
- Tensorflow 2.4.1
- CSV

# Getting started
Before using this project, make sure you have your dataset images downloaded and stored in 
a folder called Dataset. Dataset should have two directories, test and train. store your unlabeled
data in test, and your labeled data in train. Rename the test data to "(number)" and the train data 
to "cat_(number)" or "dog_number". 

If you don't want to bother with all of that, download the dataset here https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition .

# Preprocessing the images
The images on the cats vs dogs dataset are not of the same size, so it is imperative that we first resize the image before
we do any training at all. We do this by running the dataset.py. Note that this is only need to be run one time.
***Make sure you have sufficient RAM and HDD space***.

# Training the model
To train the model, go ahead to the main file and run it. Modify the parameters as you see fit.

# Labeled testing
There is no labeled testing data from the dataset, so you need to divide the train data provided. This is already done 
on line 11-12 on Main.py. First we split the main data by 80% Train, 20% Test, and then we split the training data by
80% Training and 20% Validation. Modify this as you please.

The testing on labeled data is done on the same script as training the data. On training, the model will save on the
best accuracy, and on testing it will load the previously saved model.

# Unlabeled testing (Predicting)
After training, you might want to test the model on unlabeled data. You can do this by running test.py, 
but make sure you rename the model_name to your desired model. The script will test all images already processed
by Dataset.py stored on /Dataset/test so, but only shows 49 images to analyze.
