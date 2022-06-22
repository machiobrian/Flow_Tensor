#throws in the error -> To enable them in other operations, rebuild TensorFlow with
#  the appropriate compiler flags.-> solve with =>
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense

#dense-> most common connects an every i/p to o/p -> fully connected
#activation function -. defines a neurons o/p given i/p

#the actual sequential model we are building
# its a linear stack of layers
# """the first dense layer is basically the first hidden layer
# the input layer is defined by the input_shape param ->tells the 
# dense layer what kinda input to expect -
# .. Keras defines the i/p layer automatically"""
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
]) #units = nodes/ neurons -> input_shape is only defined for first hidden layer
# softmax for output layer of the neural net -> predict a multinominal 
# probability distribution
model.summary()

