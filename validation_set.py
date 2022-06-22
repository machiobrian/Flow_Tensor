#check for overfitting and underfitting issues

#validation split -> get a portion of validation set 
from train_simple_dset import model
from simple_dataset import scaled_trained_samples, train_labels

model.fit(x=scaled_trained_samples,y=train_labels, validation_split=0.1,
        batch_size=10, epochs=30, shuffle=True, verbose=2)