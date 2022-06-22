#generate data from an imaginary clinical trial

import numpy as np
from random import randint, random

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

#empty lists
train_labels = []
train_samples = []

test_labels = []
test_samples = []

#generate dummy data
for i in range(50):
    #5% of young individuals not experiencing any side-effect
    random_young = randint(13,64)
    train_samples.append(random_young)
    train_labels.append(1)

    #5% of older individuals w/no side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    #95% of young individuals not experiencing side-ffect
    random_young = randint(13,64)
    train_samples.append(random_young)
    train_labels.append(0)

    #95% -> older individuals -> w/side effect
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

# for i in train_samples:
#     print(i)

#convert the trained samples and samples into a numpy array
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

#scale down the data
#1. specify the range that we want
scaler = MinMaxScaler(feature_range=(0,1))
scaled_trained_samples = scaler.fit_transform(train_samples.reshape(-1,1))
#the (-1,1) is a formality since the fit_transform does not accept 1D data by default

#print the rescaled data
# for k in scaled_trained_samples:
#     print(k)




