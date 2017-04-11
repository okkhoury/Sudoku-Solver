from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import skimage 
from skimage import io


# put this code in a function to make it cleaner
def loadModel():
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")

	# evaluate loaded model on test data
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	return loaded_model

loaded_model = loadModel()

# This is just a test image I found online
zero = io.imread("zero.jpeg")

def predictImageVal(numImage):
 
 	# Make sure that image is in the correct format. (1, 784)
	numImage = skimage.color.rgb2grey(numImage)
	numImage = np.resize(numImage, (28, 28)) # All images have to be input as 28x28
	numImage = numImage.flatten(order='C') # This changes the size to (, 784)
	numImage = np.resize(numImage, (784, 1)) # Puts image into correct shape: (1, 784)
	numImage = np.transpose(numImage)

	# pass formatted image into neural net and get prediction matrix
	predMatrix = loaded_model.predict(numImage)
	print(predMatrix)

	# Search the probability matrix to find the classifier with the highest probability
	maxVal = 0
	maxIndex = 0
	counter = 0
	for col in range(predMatrix.shape[1]):
		if predMatrix[(0, col)] > maxVal:
			maxVal = predMatrix[(0, col)]
			maxIndex = counter
		counter += 1
	return maxIndex

print(predictImageVal(zero))

