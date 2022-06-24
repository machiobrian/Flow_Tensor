#how the model that we trained in build_train.py holds up in predicting 
# on images of test cats and dogs -> using a CNN for inference - making educated guesses

from cnn1_create import test_batches, img
from build_train_CNN import model
import numpy as np

test_image, test_labels = next(test_batches ) #extract a batch of images and the 
                            #corresponding labels from the test_set
plotImages(img)                                
test_batches.classes

predictions = model.predict(x=test_batches, steps=len(test_batches),verbose=0)
np.round(predictions)