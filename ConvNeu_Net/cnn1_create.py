import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import (Dense, Activation, Flatten,
BatchNormalization, Conv2D, MaxPool2D)
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras import applications as app
#from keras.applications.vgg16 import preprocess_input
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#%matplotlib inline 

#organize our data into directories
os.chdir('/home/machio_b/Documents/Dog_Cats/train') #move to the directory with the data
if os.path.isdir('train/dog') is False: #check to see if dir is already in place
    #if not we proceed to create 3 directories with 2 sub-dir in each 
    # train, test, valid -> cat, dog
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')
#from the 25000 photos all random, we move just a small random number to the directories
    for i in random.sample(glob.glob('cat*'), 200): #1000 samples into the train set
        shutil.move(i, 'train/cat')
    for i in random.sample(glob.glob('dog*'), 200):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 70): #200 samples into the valid set
        shutil.move(i, 'valid/cat')
    for i in random.sample(glob.glob('dog*'), 70):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 15): #100 samples into the test set
        shutil.move(i, 'test/cat')
    for i in random.sample(glob.glob('dog*'), 15):
        shutil.move(i, 'test/dog')      #the remaining amount of data remains in the directory
        #the data remaining data can be deleted at will

os.chdir('../../')

#having obtained the data that we require and organizing it -> time to process the data

#create variables to have the location of the data sets
#specify the relative path / also it can be specified absolutely
train_path = '/home/machio_b/Documents/Dog_Cats/train/train'
test_path = '/home/machio_b/Documents/Dog_Cats/train/test'
valid_path = '/home/machio_b/Documents/Dog_Cats/train/valid'
    
#generate batches of image tensor data from the respective directories - returns 
# a directory iterator -> an infinitely repeating

train_batches = ImageDataGenerator(preprocessing_function=app.vgg16.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(120,120), classes=['cat', 'dog'], 
    batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=app.vgg16.preprocess_input)\
    .flow_from_directory(directory=valid_path, target_size=(120,120), classes=['cat', 'dog'],
    batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=app.vgg16.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(244,244), classes=['cat', 'dog'],
    batch_size=10, shuffle=False) #default-> data is shuffed but since we need the data 
    # in te confusion.... we need it in an unshuffled form

#flow_from_dir -. creates a directory iterator that enetrates batches of normalized
# tensor image dta from the respective dirs
# target_size -> resizes the image to teh specified size -> size is determined by the
# input size the neural net expects

#visulaize the data 

img, labels = next(train_batches) #generate a batch of images and labels from the training set

#plot the processed images

# def plotImages(images_arr):
#     fig, axes = plt.subplots(1,10, figsize=(20,20))
#     axes = axes.flatten()

#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# plotImages(img)
# print(labels)