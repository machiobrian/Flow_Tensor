#prediction -> Inference

from statistics import mode
from simple_dataset import train_samples, train_labels
#we are going to use this same training dataset as our testing dataset
from train_simple_dset import model, scaled_trained_samples
import os.path as path
predictions = model.predict(x=scaled_trained_samples, batch_size=10, verbose=0)
for i in predictions:
    print(i)
    #for each sample in our test set -> [!exp side effect,experiecning side effect]

#calling the save() function ... if it does not exist, create new and save... 
# if it does overwrite and save
if path.isfile('models/medical_trial_model.h5') is False:
    model.save('models/medical_trial_model.h5')
