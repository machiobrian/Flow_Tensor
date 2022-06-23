from cnn1_create import train_batches, valid_batches
import tensorflow as tf
import keras
from keras.activations import relu, softmax
from keras.models import Sequential #one input/output tensor
from keras.layers import (Dense, MaxPool2D, Conv2D, Flatten)
from keras.optimizers import Adam


model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
    input_shape=(120,120,3)),
    MaxPool2D(pool_size=(2,2), strides=2), #cut by half
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2,2), strides=2),
    Flatten(), #to 1 dimensional
    Dense(units=2, activation='softmax') #sftmax -> gives probability of 
])

#first layer ->  2 dimensional convolution layer, with 32 output filters of kernel
#size of 3x3 using the relu activation function -> o/p filters are chosen arbitrarily 
# the  input shape is the shape of our image data, enable zero padding by 
# specifying padding as zero

#MaxPooling -> reuces the dimensionality of our data

#model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',
 metrics=['accuracy'])

 #incase we have two classes, we configure our output layer to have a single o/p 
 # and use binary_crossentropy, therefore configuring the last layer as sigmoid 
 # as our activation function -> Illustrate below

# model2 = Sequential([
#     Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
#     input_shape=(224,224,3)),
#     MaxPool2D(pool_size=(2,2), strides=2),
#     Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
#     MaxPool2D(pool_size=(2,2), strides=2),
#     Flatten(),
#     Dense(units=1, activation='sigmoid')
# ])



#model2.summary()

#TRAINING

model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=10,
    verbose=2
)


#steps_per_epoch -> how many batches of samples should be passed to the model 
# before declaring one epoch complete -> we have 1000, train samples, batch size of 10 
# therefore the steps are 100

# changes batch size to 5 in cnn1_create.py


