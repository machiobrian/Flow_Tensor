import tensorflow as tf
from keras.models import load_model #needed if we are to load
# a previously installed model

new_model = load_model('models/medical_trial_model.h5')

#new_model.summary()

# """the save function, saves:
# model architecture -> allows model recreation
# model weights -> if already trained
# training configuration -> loss, optimizer
# state of optimizer -> allows resumption of model where we 
# left if"""

new_model.get_weights()