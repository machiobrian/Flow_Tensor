from setup4training import model
from simple_dataset import scaled_trained_samples, train_labels

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


model.summary()

#compile the model and make it ready for training
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

#training occurs when we make a call -> the .fit function
model.fit(x=scaled_trained_samples, y=train_labels, batch_size=10, epochs=30,
shuffle=True, verbose=2)