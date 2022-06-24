import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications import imagenet_utils

from sklearn.metrics import confusion_matrix

import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

#function downloads a mobilenet network for inference -> we see how well 
# it performes in classifying images as per imageNet class
mobile = tf.keras.applications.mobilenet.MobileNet()
#mobile net was trained by ImageNet just as VGG16

#prep_image fxn -> gets the image file and preps it in the formart 
#required by the model

def prepare_image(file):
    img_path = '/MobileNet-samples/'
    img = tf.keras.utils.load_img(img_path + file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

preprocessed_image = prepare_image('1.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
