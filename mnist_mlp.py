'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 25

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0, :,:])
plt.imshow(x_train[0, :,:], cmap='gray')
plt.show()

x_train = x_train.reshape(60000, 784)

x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(y_test)
print(y_test.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""2 hidden layers each with 256 neurons"""
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

def confusionMatrix(Model):

	probMatrix = Model.predict(x_test)
	print(probMatrix.shape)

	predictedVals = np.zeros((1, 10000))
	print(predictedVals.shape)

	realVals = np.zeros((10000, 1))

	# Get the matrix for the test values
	for row in range(10000):
		for col in range(10):
			if y_test[(row, col)] == 1.0:
				realVals[(row, 0)] = col

	# Normalize the probMatrix
	probMatrix = normalize(probMatrix, axis=1, norm='l1')

	# Get the matrix for the predicated test values
	for row in range(probMatrix.shape[0]):
		maxIndex = 0
		maxVal = 0
		for col in range(probMatrix.shape[1]):
			if probMatrix[(row, col)] > maxVal:
				maxVal = probMatrix[(row, col)]
				maxIndex = col 
		predictedVals[(0, row)] = maxIndex
		maxIndex = 0
		maxVal = 0

	predictedVals = np.transpose(predictedVals)
	predictedVals = predictedVals.reshape((10000, ))
	predictedVals = predictedVals.astype(np.uint8)

	realVals = realVals.reshape((10000, ))
	realVals = realVals.astype(np.uint8)

	return confusion_matrix(realVals, predictedVals)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



