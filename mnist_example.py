#change the directory to run from the virtual environment of the conda_tfv
import tensorflow as tf
#from tensorflow import keras -> not needed anymore since keras is a standalone
import matplotlib.pyplot as plt #create figures and add different images on the figures
import numpy as np
#from tensorflow.keras.layers import Conv2D, Input, Dense, Maxpool2D, BatchNormalization, Flatten, GlobalAveragePool2D
from keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAveragePooling2D

def display_some_examples(examples, labels):
    #randomly choose some images from the data sets and them inside the figure
    plt.figure(figsize=(10,10))

    #plot 25 images - 5x5
    for i in range(25):
        idx = np.random.randint(34, examples.shape[0]-1)
        #choose images from index 34 to 60,000 -1 = 59,999
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout() #automatically adds more space to see the images and labels in
                    #a much better way
        plt.imshow(img, cmap='gray') #inform plt the images its receiving are gray scale
    plt.show()


#if the script is run, run the part but when imprted dont run this part
# creates a sweparate behaviour when importing and when running the scripts
if __name__=="__main__":

    #start by importing the data set
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
    #load the data and put them in a directory
    #x -the data set and y -the labels

    #print the shape of the data sets that we just downloaded
    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)
    #each data set is 28x28 pixel -> also, lets plot some of the examples and corresponding
    #labels


    display_some_examples(x_test, y_test) #call the function and pass it the parameters
    