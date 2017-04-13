from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import skimage 
from skimage import io
import matplotlib.pyplot as plt
from skimage import transform

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
zero = io.imread("sevenHand.jpg")

# need to add a line of code to resize and scale the image to 28x28, so the the CNN can predict it
def predictImageVal(numImage):

	#Convert image to gray scale, resize it to 28x28, convert type to ubyte
	numImage = skimage.color.rgb2grey(numImage)
	numImage = transform.resize(numImage, (28,28))
	numImage = skimage.img_as_ubyte(numImage)

	#Our images are black text / white background, the model needs white text / black background. These lines invert the black/white
	invertedImg = np.zeros((28,28))
	invertedImg[numImage < 100] = 255
	invertedImg[numImage >= 100] = 0

	#Forms the image to the correct format to be read by the model
	invertedImg = invertedImg.flatten(order='C')  
	invertedImg = np.resize(invertedImg, (784,1))    # This changes the size to (, 784)
	invertedImg = np.transpose(invertedImg)          # Puts image into correct shape: (1, 784)

	# pass formatted image into neural net and get prediction matrix
	predMatrix = loaded_model.predict(invertedImg)
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

